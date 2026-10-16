"""Matched head-to-head policy imitation with equal-cost path diversity."""

import argparse
import fcntl
import json
import os
import random
import time
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import atomic, digest, sha
from mcts_training_supervision import verified_success_traces
from pretrained_raster_policy import policy_log_probabilities
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness
from rf_knots.site_proposals import proposal_candidates

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
BANK = Path("artifacts/qrun-equal-cost-bank-v1/bank.json")
PARENT = Path("artifacts/qrun-like-policy-curriculum-rung1-v1-envfix/step-1024.pt")
PROTOCOL = Path("research/292-qrun-equal-cost-diversity-pilot-v1.md")
ROSTER_SHA = "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506"
PARENT_SHA = "03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8"
BANK_SHA = "ebfc9f467a783255e48aa2023e8060cd0309519521ec3bbd515ce774fe7b1264"
CONFIG = {"steps": 320, "batch_roots": 4, "checkpoint_interval": 32,
          "seed": 20261006, "lr": 5e-5, "weight_decay": 0.01,
          "grad_clip": 1.0, "device": "mps", "alternative_probability": 0.5}


def read(path):
    return json.loads(path.read_text())


def build_bank():
    roster = read(ROSTER)
    alternatives = read(BANK)
    if ([len(stage) for stage in roster["stages"]] != [140, 286, 287, 287]
            or alternatives["schema"] != "rf-knots-qrun-equal-cost-bank-v1"
            or len(alternatives["rows"]) != 1000):
        raise ValueError("TRAIN roster or equal-cost bank differs")
    layers = []
    index = 0
    for stage in roster["stages"]:
        layer = []
        for root in stage:
            row = alternatives["rows"][index]
            if (row["root_index"] != index or row["stage"] != root["stage"]
                    or row["generator_index"] != root["generator_index"]):
                raise ValueError("equal-cost root order differs")
            reference = UnknotWitness.from_dict(root["witness"])
            reference.verify()
            pair = []
            for weight_index, instance_dict in enumerate(root["instances"]):
                instance = M.Instance(**{**instance_dict,
                                         "word": tuple(instance_dict["word"])})
                ref_rows = verified_success_traces(instance, [reference])[0]["rows"]
                alternative = row["alternatives"][weight_index]
                alt_rows = None
                if alternative is not None:
                    witness = UnknotWitness.from_dict(alternative)
                    witness.verify()
                    if (instance.cc_weight * witness.crossing_changes + witness.moves
                            != instance.cc_weight * reference.crossing_changes + reference.moves):
                        raise ValueError("alternative objective cost differs")
                    alt_rows = verified_success_traces(instance, [witness])[0]["rows"]
                pair.append((instance, ref_rows, alt_rows))
            if tuple(p[0].cc_weight for p in pair) != (10, 1000):
                raise ValueError("paired objectives differ")
            layer.append(pair)
            index += 1
        layers.append(layer)
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if (args.out.exists() != args.resume or args.stop_after < 0
            or args.stop_after % CONFIG["checkpoint_interval"]):
        raise ValueError("new output without --resume; existing with --resume; stop at checkpoint")
    if (not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"):
        raise RuntimeError("require native MPS without CPU fallback")
    torch.set_num_threads(1)
    pins = (ROSTER, BANK, PARENT, PROTOCOL, Path(__file__),
            Path("scripts/pretrained_raster_policy.py"),
            Path("scripts/mcts_training_supervision.py"),
            Path("src/rf_knots/site_proposals.py"))
    inputs = {str(path): sha(path) for path in pins}
    if (inputs[str(ROSTER)] != ROSTER_SHA or inputs[str(BANK)] != BANK_SHA
            or inputs[str(PARENT)] != PARENT_SHA):
        raise ValueError("pinned TRAIN bank or parent changed")
    contract = {"schema": "rf-knots-qrun-equal-cost-pilot-v1", "arm": args.arm,
                "config": CONFIG, "inputs": inputs,
                "role": "verified equal-cost imitation pilot, not cost improvement"}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        manifest = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "optimizer_steps": 0,
            "checkpoints": [], "wall_seconds_committed": 0.0,
            "cpu_seconds_committed": 0.0, "stage_draws": [0, 0, 0, 0],
            "alternative_draws": 0, "alternative_available_draws": 0}
        if (manifest["contract"] != json.loads(json.dumps(contract))
                or manifest["contract_sha256"] != contract_sha
                or manifest["optimizer_steps"] != 32 * len(manifest["checkpoints"])):
            raise ValueError("resume contract/steps/checkpoints differ")
        for index, entry in enumerate(manifest["checkpoints"], 1):
            if entry["step"] != index * 32 or sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("committed checkpoint changed")
        if manifest["optimizer_steps"] == CONFIG["steps"]:
            manifest["status"] = "completed"
            atomic(manifest_path, manifest)
            return
        bank = build_bank()
        net, objective = load_actor()
        parent = torch.load(PARENT, map_location="cpu", weights_only=False)
        if parent["step"] != 1024:
            raise ValueError("parent endpoint differs")
        net.load_state_dict(parent["model"], strict=True)
        objective.load_state_dict(parent["objective"], strict=True)
        net.to("mps")
        optimizer = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"],
                                      weight_decay=CONFIG["weight_decay"])
        seed = CONFIG["seed"]
        objective_rng = random.Random(seed)
        choice_rng = random.Random(seed + 1)
        position_rng = random.Random(seed + 2)
        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)
        draws = list(manifest["stage_draws"])
        if manifest["optimizer_steps"]:
            entry = manifest["checkpoints"][-1]
            saved = torch.load(args.out / entry["file"], map_location="cpu",
                               weights_only=False)
            if saved["contract_sha256"] != contract_sha or saved["step"] != entry["step"]:
                raise ValueError("resume checkpoint contract differs")
            net.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            objective_rng.setstate(saved["objective_rng"])
            choice_rng.setstate(saved["choice_rng"])
            position_rng.setstate(saved["position_rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.mps.set_rng_state(saved["mps_rng"])
            if saved["stage_draws"] != draws:
                raise ValueError("resume stage counters differ")
        start = manifest["optimizer_steps"]
        manifest.update(status="training", pid=os.getpid())
        atomic(manifest_path, manifest)
        end = min(CONFIG["steps"], start + (args.stop_after or CONFIG["steps"]))
        wall_start, cpu_start = time.monotonic(), time.process_time()
        for step in range(start, end):
            net.train()
            requests, targets = [], []
            for stage in range(4):
                root_index = draws[stage] % len(bank[stage])
                draws[stage] += 1
                weight_index = objective_rng.randrange(2)
                coin = choice_rng.randrange(2)
                instance, ref_rows, alt_rows = bank[stage][root_index][weight_index]
                if alt_rows is not None:
                    manifest["alternative_available_draws"] += 1
                use_alternative = args.arm == "B" and coin == 0 and alt_rows is not None
                manifest["alternative_draws"] += int(use_alternative)
                trace = alt_rows if use_alternative else ref_rows
                state, action = trace[position_rng.randrange(len(trace))]
                actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | {action}))
                requests.append((instance, state, actions))
                targets.append(actions.index(action))
            optimizer.zero_grad(set_to_none=True)
            logp = policy_log_probabilities(net, requests)
            loss = -logp[torch.arange(len(requests), device=logp.device), targets].mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite imitation loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), CONFIG["grad_clip"],
                                                  error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % CONFIG["checkpoint_interval"] == 0:
                torch.mps.synchronize()
                saved = {"contract_sha256": contract_sha, "step": step + 1,
                         "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                         "objective": {k: v.detach().cpu() for k, v in objective.state_dict().items()},
                         "optimizer": optimizer.state_dict(),
                         "objective_rng": objective_rng.getstate(),
                         "choice_rng": choice_rng.getstate(),
                         "position_rng": position_rng.getstate(),
                         "torch_rng": torch.get_rng_state(),
                         "mps_rng": torch.mps.get_rng_state(),
                         "stage_draws": draws.copy(),
                         "alternative_draws": manifest["alternative_draws"],
                         "alternative_available_draws": manifest["alternative_available_draws"]}
                checkpoint_path = args.out / f"step-{step + 1:04d}.pt"
                partial = checkpoint_path.with_suffix(".partial")
                torch.save(saved, partial)
                partial.replace(checkpoint_path)
                manifest["checkpoints"].append({"file": checkpoint_path.name,
                                                "sha256": sha(checkpoint_path),
                                                "step": step + 1})
                manifest["optimizer_steps"] = step + 1
                manifest["wall_seconds_committed"] += time.monotonic() - wall_start
                manifest["cpu_seconds_committed"] += time.process_time() - cpu_start
                manifest["stage_draws"] = draws.copy()
                manifest.update(loss=float(loss.detach()), grad_norm=float(grad))
                atomic(manifest_path, manifest)
                wall_start, cpu_start = time.monotonic(), time.process_time()
                print(json.dumps({"arm": args.arm, "optimizer_steps": step + 1,
                                  "stage_draws": draws, "alternative_draws":
                                  manifest["alternative_draws"], "loss": manifest["loss"]}),
                      flush=True)
        if manifest["optimizer_steps"] == CONFIG["steps"]:
            manifest.update(status="completed", pid=None)
            atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
