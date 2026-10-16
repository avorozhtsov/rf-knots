"""Frozen-actor, fixed-b16 head ablations on two exploratory root splits."""

import argparse
import fcntl
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import ACTOR, ACTOR_SHA, ROOTS, load_actor
from budgeted_cost_value_head_v1 import BudgetedCostValueHead, value_loss
from build_mcts_training_supervision import atomic, sha
from raster_log_training import digest
from root_group_holdout_value_gate_v1 import evaluate, partition, records_for
from train_root_budget16_value_head_v1 import encode_roots


LABELS = Path("artifacts/root-budget16-multioutput-labels-v1/labels.json")
PROTOCOL = Path("research/270-root-value-head-ablation-v1.md")
TESTS = Path("tests/test_root_value_head_ablation_v1.py")
VARIANTS = {
    "narrow": {"width": 32, "weight_decay": 0.01, "sampler": "attempt"},
    "decay": {"width": 128, "weight_decay": 0.10, "sampler": "attempt"},
    "grouped": {"width": 128, "weight_decay": 0.01, "sampler": "root_pair"},
}
CONFIG = {"seed": 20261004, "steps": 512, "checkpoint_interval": 64,
          "batch_size": 64, "lr": 0.001, "grad_clip": 1.0,
          "device": "mps", "local_simulations": 16, "reference_simulations": 1024}


def read(path):
    return json.loads(path.read_text())


def grouped_lookup(examples):
    """96 root pairs x 20 equally weighted lambda/seed attempt indices."""
    positions = defaultdict(list)
    for index, row in enumerate(examples):
        positions[row["root_index"] // 2].append((row["lambda"], row["seed"], index))
    if len(positions) != 96 or any(len(group) != 20 for group in positions.values()):
        raise ValueError("fit root-pair coverage differs")
    return torch.tensor([[index for _, _, index in sorted(positions[root])]
                         for root in sorted(positions)], dtype=torch.long)


def draw_batch(rng, count, lookup, sampler):
    if sampler == "attempt":
        return torch.randint(count, (CONFIG["batch_size"],), generator=rng)
    if sampler == "root_pair":
        selected = torch.randperm(lookup.shape[0], generator=rng)[:CONFIG["batch_size"]]
        chosen_attempts = torch.randint(20, (CONFIG["batch_size"],), generator=rng)
        return lookup[selected, chosen_attempts]
    raise ValueError("unknown sampler")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--split-salt", type=int, choices=(20261005, 20261006), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical dependency missing")
    sys.path.insert(0, str(pgx_src))
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("native MPS required without CPU fallback")
    if sha(ACTOR) != ACTOR_SHA or sha(LABELS) != "57e7fa1d021ddc54aaeb220d68e415e6ab91d6765749b8087bd1eda8ebd77201":
        raise ValueError("actor or labels changed")
    labels = read(LABELS)
    if any(sha(Path(path)) != digest for path, digest in labels["inputs"].items()):
        raise ValueError("label provenance changed")
    roots = read(ROOTS)
    heldout = partition(roots, args.split_salt)
    fit, test = records_for(roots, labels["labels"], heldout)
    variant = VARIANTS[args.variant]
    inputs = {str(path): sha(path) for path in (
        ACTOR, LABELS, ROOTS, PROTOCOL, TESTS, Path(__file__),
        Path("scripts/root_group_holdout_value_gate_v1.py"),
        Path("scripts/train_root_budget16_value_head_v1.py"),
        Path("scripts/budgeted_cost_value_head_v1.py"),
        Path("scripts/bootstrap_cost_value_teacher_v1.py"),
    )}
    contract = {"schema": "rf-knots-root-value-head-ablation-v1",
                "inputs": inputs, "actor_sha256": ACTOR_SHA,
                "config": {**CONFIG, **variant, "variant": args.variant,
                           "split_salt": args.split_salt},
                "holdout_base_indices": heldout}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        state = read(manifest_path) if args.resume else {
            "schema": contract["schema"], "contract": contract,
            "contract_sha256": contract_sha, "status": "preparing",
            "optimizer_steps": 0, "checkpoints": []}
        if (state["contract"] != contract or state["contract_sha256"] != contract_sha
                or state["optimizer_steps"] != len(state["checkpoints"]) * CONFIG["checkpoint_interval"]):
            raise ValueError("ablation resume contract/coverage differs")
        for index, entry in enumerate(state["checkpoints"]):
            if (entry["step"] != (index + 1) * CONFIG["checkpoint_interval"]
                    or sha(args.out / entry["file"]) != entry["sha256"]):
                raise ValueError("ablation checkpoint SHA differs")
        if state["status"] == "completed":
            print("already completed", flush=True)
            return
        atomic(manifest_path, state)
        torch.set_num_threads(1)
        started_wall, started_cpu = time.monotonic(), time.process_time()
        net, _ = load_actor()
        net.to("mps").eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        features = encode_roots(net, roots)
        encoded = {name: torch.stack([features[(r["root_index"], name)] for r in fit])
                   for name in ("knot", "layout", "isotopy")}
        cc_left = torch.tensor([r["cc_left"] for r in fit], dtype=torch.float32, device="mps")
        moves_left = torch.tensor([r["moves_left"] for r in fit], dtype=torch.float32, device="mps")
        lam = torch.tensor([r["lambda"] for r in fit], dtype=torch.float32, device="mps")
        solved = torch.tensor([r["solved"] for r in fit], dtype=torch.float32, device="mps")
        cc = torch.tensor([float("nan") if r["cc_remaining"] is None else r["cc_remaining"]
                           for r in fit], dtype=torch.float32, device="mps")
        moves = torch.tensor([float("nan") if r["moves_remaining"] is None else r["moves_remaining"]
                              for r in fit], dtype=torch.float32, device="mps")
        lookup = grouped_lookup(fit)
        head = BudgetedCostValueHead(width=variant["width"],
                                     reference_simulations=CONFIG["reference_simulations"]).to("mps")
        param_count = sum(p.numel() for p in head.parameters())
        optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["lr"],
                                      weight_decay=variant["weight_decay"])
        rng = torch.Generator(device="cpu")
        rng.manual_seed(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])
        torch.mps.manual_seed(CONFIG["seed"])
        start = state["optimizer_steps"]
        if start:
            entry = state["checkpoints"][-1]
            payload = torch.load(args.out / entry["file"], map_location="cpu", weights_only=False)
            if payload["contract_sha256"] != contract_sha or payload["step"] != start:
                raise ValueError("resume checkpoint identity differs")
            head.load_state_dict(payload["head"])
            optimizer.load_state_dict(payload["optimizer"])
            rng.set_state(payload["draw_rng"])
            torch.set_rng_state(payload["torch_rng"])
            torch.mps.set_rng_state(payload["mps_rng"])
        state.update(status="training", pid=os.getpid(), parameter_count=param_count)
        atomic(manifest_path, state)
        end = min(CONFIG["steps"], start + (args.stop_after or CONFIG["steps"]))
        for step in range(start, end):
            indices = draw_batch(rng, len(fit), lookup, variant["sampler"]).to("mps")
            output = head({name: tensor[indices] for name, tensor in encoded.items()},
                          cc_left=cc_left[indices], moves_left=moves_left[indices],
                          cc_weight=lam[indices],
                          local_simulations=torch.full((len(indices),), 16.0, device="mps"))
            losses = value_loss(output, solved=solved[indices], cc=cc[indices],
                                moves=moves[indices], cc_left=cc_left[indices],
                                moves_left=moves_left[indices])
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite value loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CONFIG["grad_clip"],
                                                  error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % CONFIG["checkpoint_interval"] == 0:
                torch.mps.synchronize()
                payload = {"contract_sha256": contract_sha, "step": step + 1,
                           "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                           "optimizer": optimizer.state_dict(), "draw_rng": rng.get_state(),
                           "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state()}
                checkpoint_path = args.out / f"step-{step + 1:04d}.pt"
                temporary = checkpoint_path.with_suffix(".partial")
                torch.save(payload, temporary)
                temporary.replace(checkpoint_path)
                state["checkpoints"].append({"file": checkpoint_path.name,
                                             "sha256": sha(checkpoint_path), "step": step + 1})
                state.update(optimizer_steps=step + 1, value_loss=float(losses["total"].detach()),
                             solve_loss=float(losses["solve"].detach()),
                             cost_loss=float(losses["cost"].detach()),
                             gradient_norm_before_clip=float(norm),
                             train_cpu_seconds=time.process_time() - started_cpu,
                             train_wall_seconds=time.monotonic() - started_wall)
                atomic(manifest_path, state)
                print(json.dumps({"salt": args.split_salt, "variant": args.variant,
                                  "step": step + 1, "loss": state["value_loss"]}), flush=True)
        if state["optimizer_steps"] == CONFIG["steps"]:
            net.cpu().eval()
            report = evaluate(net, head, roots, fit, test)
            report.update(schema="rf-knots-root-value-head-ablation-report-v1",
                          contract_sha256=contract_sha, variant=args.variant,
                          split_salt=args.split_salt, parameter_count=param_count,
                          checkpoint_sha256=state["checkpoints"][-1]["sha256"],
                          train_cpu_seconds=state["train_cpu_seconds"],
                          train_wall_seconds=state["train_wall_seconds"])
            atomic(args.out / "report.json", report)
            state.update(status="completed", pid=None,
                         report_sha256=sha(args.out / "report.json"))
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
