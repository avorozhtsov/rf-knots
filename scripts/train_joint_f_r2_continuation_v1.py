"""Resume selected R2 from frozen v4 endpoint into a separate F training run."""

import argparse
import fcntl
import json
import math
import os
import random
import time
from pathlib import Path

import torch

import train_joint_f_r2_r4_v1 as base
import train_joint_f_r2_r4_bounded_v2 as bounded
from train_joint_f_r2_r4_smoothed_v3 import ERROR_MASS, MARGIN, smoothed_pairwise_loss

PARENT = Path("artifacts/joint-f-r2-r4-smoothed-v4-R2/step-114561.pt")
PARENT_SHA = "1ffe7346e0725fcb98125ebf03ab1faaaf7048985d69eeb00300e2828cca6946"
PARENT_MANIFEST = PARENT.parent / "manifest.json"
PROTOCOL = Path("research/314-r2-only-f-continuation-v1.md")
START_STEP = 114561
BLOCK = 128


def setup_training():
    bounded.pairwise_margin_loss = smoothed_pairwise_loss
    base.CONFIG["F_loss"] = "exact-context-smoothed-pairwise-softplus"
    base.CONFIG["F_margin"] = MARGIN
    base.CONFIG["F_error_mass"] = ERROR_MASS
    base.CONFIG["checkpoint_interval"] = BLOCK
    base.train_step = bounded.train_step


def inputs():
    paths = (base.BANK / "manifest.json", base.BANK_AUDIT, PARENT,
             PARENT_MANIFEST, PROTOCOL, Path(__file__),
             Path("scripts/train_joint_f_r2_r4_v1.py"),
             Path("scripts/train_joint_f_r2_r4_bounded_v2.py"),
             Path("scripts/train_joint_f_r2_r4_smoothed_v3.py"),
             Path("scripts/train_joint_f_r2_r4_smoothed_v4.py"),
             Path("scripts/mcts_tournament_signals_v1.py"),
             Path("scripts/tournament_term_sampler_v1.py"),
             Path("scripts/pretrained_raster_policy.py"),
             Path("scripts/mcts_training_supervision.py"))
    hashes = {str(path): base.sha(path) for path in paths}
    if (hashes[str(base.BANK / "manifest.json")] != base.BANK_SHA
            or hashes[str(base.BANK_AUDIT)] != base.BANK_AUDIT_SHA
            or hashes[str(PARENT)] != PARENT_SHA):
        raise ValueError("frozen bank, audit, or parent differs")
    parent_manifest = base.read(PARENT_MANIFEST)
    if (parent_manifest["optimizer_steps"] != START_STEP
            or parent_manifest["checkpoints"][-1]["sha256"] != PARENT_SHA
            or parent_manifest["contract"]["arm"] != "R2"):
        raise ValueError("parent manifest differs")
    return hashes


def make_objects(saved):
    net, objective = base.load_actor()
    net.encoder.graph_rounds = 2
    head = base.AuxiliaryCostHead()
    net.to("mps")
    objective.to("mps")
    head.to("mps")
    parameters = [*net.parameters(), *objective.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=base.CONFIG["lr"],
                                  weight_decay=base.CONFIG["weight_decay"])
    net.load_state_dict(saved["model"], strict=True)
    objective.load_state_dict(saved["objective"], strict=True)
    head.load_state_dict(saved["auxiliary_head"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if not optimizer.state:
        raise ValueError("missing AdamW state")
    rng = random.Random()
    rng.setstate(saved["rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.mps.set_rng_state(saved["mps_rng"])
    return net, objective, head, optimizer, rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=BLOCK)
    args = parser.parse_args()
    if args.out.exists() != args.resume or not 1 <= args.stop_after <= BLOCK:
        raise ValueError("new output without --resume; existing with --resume; 1..128 steps")
    if (not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"):
        raise RuntimeError("native MPS without CPU fallback required")
    torch.set_num_threads(1)
    setup_training()
    frozen = inputs()
    order = base.bank_order(base.read(base.BANK / "manifest.json"))
    contract = {"schema": "rf-knots-R2-F-continuation-v1", "arm": "R2",
                "parent_step": START_STEP, "block_steps": BLOCK,
                "config": base.CONFIG.copy(), "inputs": frozen,
                "eligible_roots": order, "role": "native-MPS F-dominant training"}
    contract_sha = base.digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        manifest = base.read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "optimizer_steps": START_STEP,
            "checkpoints": [], "new_active_seconds": 0.0,
            "new_cpu_seconds": 0.0, "stage_draws": None}
        if (manifest["contract"] != json.loads(json.dumps(contract))
                or manifest["contract_sha256"] != contract_sha
                or manifest["optimizer_steps"] != (manifest["checkpoints"][-1]["step"]
                                                  if manifest["checkpoints"] else START_STEP)):
            raise ValueError("resume contract/checkpoint state differs")
        for entry in manifest["checkpoints"]:
            if base.sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("checkpoint SHA differs")
        entry = manifest["checkpoints"][-1] if manifest["checkpoints"] else None
        saved = torch.load(args.out / entry["file"] if entry else PARENT,
                           map_location="cpu", weights_only=False)
        if (saved["step"] != manifest["optimizer_steps"]
                or saved["contract_sha256"] != (contract_sha if entry else
                                                 base.read(PARENT_MANIFEST)["contract_sha256"])):
            raise ValueError("loaded checkpoint identity differs")
        if manifest["stage_draws"] is None:
            manifest["stage_draws"] = saved["stage_draws"].copy()
        if saved["stage_draws"] != manifest["stage_draws"]:
            raise ValueError("stage draws differ")
        net, objective, head, optimizer, rng = make_objects(saved)
        start = manifest["optimizer_steps"]
        end = start + args.stop_after
        manifest.update(status="training", pid=os.getpid())
        base.atomic(manifest_path, manifest)
        pending_wall = pending_cpu = 0.0
        for step in range(start, end):
            if any(base.sha(Path(path)) != expected for path, expected in frozen.items()):
                raise ValueError("pinned input changed")
            started_wall, started_cpu = time.monotonic(), time.process_time()
            net.train(); objective.train(); head.train()
            root, group, group_index = base.selected_group(order, step)
            metrics = base.train_step(net, objective, head, optimizer, group, step)
            torch.mps.synchronize()
            pending_wall += time.monotonic() - started_wall
            pending_cpu += time.process_time() - started_cpu
            manifest["stage_draws"][root["stage"]] += 1
            if ((step + 1) % BLOCK == 0 or step + 1 == end):
                if (not math.isfinite(metrics["F_loss"]) or metrics["F_loss"] < 0
                        or not math.isfinite(metrics["F_encoder_grad_norm"])
                        or metrics["F_encoder_grad_norm"] <= 1e-12):
                    raise ValueError("pathological F metrics")
                checkpoint = {
                    "contract_sha256": contract_sha, "step": step + 1,
                    "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "objective": {k: v.detach().cpu() for k, v in objective.state_dict().items()},
                    "auxiliary_head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
                    "stage_draws": manifest["stage_draws"].copy(), "metrics": metrics,
                    "root_index": root["root_index"], "group_index": group_index}
                path = args.out / f"step-{step + 1:06d}.pt"
                partial = path.with_suffix(".partial")
                torch.save(checkpoint, partial)
                partial.replace(path)
                manifest["checkpoints"].append({"file": path.name,
                                                 "sha256": base.sha(path), "step": step + 1})
                manifest["optimizer_steps"] = step + 1
                manifest["new_active_seconds"] += pending_wall
                manifest["new_cpu_seconds"] += pending_cpu
                pending_wall = pending_cpu = 0.0
                manifest.update(metrics=metrics, pid=os.getpid())
                base.atomic(manifest_path, manifest)
                print(json.dumps({"optimizer_steps": step + 1,
                                  "new_active_seconds": manifest["new_active_seconds"],
                                  "F_loss": metrics["F_loss"],
                                  "F_encoder_grad_norm": metrics["F_encoder_grad_norm"]}), flush=True)
        manifest.update(status="checkpointed", pid=None)
        base.atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
