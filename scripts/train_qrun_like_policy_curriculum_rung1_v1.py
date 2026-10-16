"""Resumable 1000-root verified-solution policy curriculum with rehearsal."""

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
ROSTER_SHA = "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506"
PARENT = Path("artifacts/qrun-like-policy-bootstrap-rung0-v2/step-0256.pt")
PARENT_SHA = "54ecbf5c7b3bb211d4dd42d0e8e254342e1748575000701d5600e5d81c2d6d43"
PROTOCOL = Path("research/284-qrun-curriculum-rung1-v1.md")
CONFIG = {"steps": 1024, "batch_roots": 4, "checkpoint_interval": 32,
          "seed": 20261005, "lr": 5e-5, "weight_decay": 0.01,
          "grad_clip": 1.0, "device": "mps",
          "phase_quotas_per_8_batches": [[16, 8, 4, 4], [8, 8, 8, 8],
                                         [4, 8, 8, 12], [4, 4, 8, 16]]}


def read(path):
    return json.loads(path.read_text())


def schedule(phase):
    if phase == 0:
        return ((0, 0, 1, 2), (0, 0, 1, 3)) * 4
    if phase == 1:
        return ((0, 1, 2, 3),) * 8
    if phase == 2:
        return ((0, 1, 2, 3), (1, 2, 3, 3)) * 4
    return ((0, 2, 3, 3), (1, 2, 3, 3)) * 4


def build_bank():
    roster = read(ROSTER)
    if [len(stage) for stage in roster["stages"]] != [140, 286, 287, 287]:
        raise ValueError("roster stage sizes differ")
    seen = set()
    bank = []
    for stage in roster["stages"]:
        layer = []
        for root in stage:
            identity = tuple(root["query_group"])
            if identity in seen:
                raise ValueError("duplicate TRAIN root")
            seen.add(identity)
            witness = UnknotWitness.from_dict(root["witness"])
            witness.verify()
            paired = []
            for instance_dict in root["instances"]:
                instance = M.Instance(**{**instance_dict,
                                         "word": tuple(instance_dict["word"])})
                trace = verified_success_traces(instance, [witness])[0]
                paired.append((instance, trace["rows"]))
            if tuple(pair[0].cc_weight for pair in paired) != (10, 1000):
                raise ValueError("resource pair differs")
            layer.append(paired)
        bank.append(layer)
    if len(seen) != 1000:
        raise ValueError("expected 1000 distinct roots")
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    pins = (ROSTER, PARENT, PROTOCOL, Path(__file__),
            Path("scripts/pretrained_raster_policy.py"),
            Path("scripts/mcts_training_supervision.py"),
            Path("src/rf_knots/site_proposals.py"))
    inputs = {str(path): sha(path) for path in pins}
    if inputs[str(ROSTER)] != ROSTER_SHA or inputs[str(PARENT)] != PARENT_SHA:
        raise ValueError("roster or mixed256 parent changed")
    contract = {"schema": "rf-knots-qrun-like-policy-curriculum-rung1-v1",
                "config": CONFIG, "inputs": inputs,
                "role": "verified-solution imitation curriculum, not F-dominant R2/R4"}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        manifest = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "optimizer_steps": 0,
            "checkpoints": [], "seconds_committed": 0.0,
            "stage_draws": [0, 0, 0, 0], "distinct_roots_seen": 0}
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
        if parent["step"] != 256:
            raise ValueError("parent endpoint differs")
        net.load_state_dict(parent["model"], strict=True)
        objective.load_state_dict(parent["objective"], strict=True)
        net.to("mps")
        optimizer = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"],
                                      weight_decay=CONFIG["weight_decay"])
        rng = random.Random(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])
        torch.mps.manual_seed(CONFIG["seed"])
        draws = list(manifest["stage_draws"])
        if manifest["optimizer_steps"]:
            entry = manifest["checkpoints"][-1]
            saved = torch.load(args.out / entry["file"], map_location="cpu",
                               weights_only=False)
            if saved["contract_sha256"] != contract_sha or saved["step"] != entry["step"]:
                raise ValueError("resume checkpoint contract differs")
            net.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            rng.setstate(saved["rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.mps.set_rng_state(saved["mps_rng"])
            if saved["stage_draws"] != draws:
                raise ValueError("resume stage counters differ")
        start = manifest["optimizer_steps"]
        manifest.update(status="training", pid=os.getpid())
        atomic(manifest_path, manifest)
        end = min(CONFIG["steps"], start + (args.stop_after or CONFIG["steps"]))
        started = time.monotonic()
        for step in range(start, end):
            net.train()
            phase = step // 256
            stages = schedule(phase)[step % 8]
            requests, targets = [], []
            for stage in stages:
                root_index = draws[stage] % len(bank[stage])
                draws[stage] += 1
                instance, trace = bank[stage][root_index][rng.randrange(2)]
                state, action = trace[rng.randrange(len(trace))]
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
                         "optimizer": optimizer.state_dict(), "rng": rng.getstate(),
                         "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
                         "stage_draws": draws.copy()}
                checkpoint_path = args.out / f"step-{step + 1:04d}.pt"
                partial = checkpoint_path.with_suffix(".partial")
                torch.save(saved, partial)
                partial.replace(checkpoint_path)
                manifest["checkpoints"].append({"file": checkpoint_path.name,
                                                "sha256": sha(checkpoint_path),
                                                "step": step + 1})
                manifest["optimizer_steps"] = step + 1
                manifest["seconds_committed"] += time.monotonic() - started
                manifest["stage_draws"] = draws.copy()
                manifest["distinct_roots_seen"] = sum(min(draws[i], len(bank[i]))
                                                      for i in range(4))
                manifest.update(loss=float(loss.detach()), grad_norm=float(grad))
                atomic(manifest_path, manifest)
                started = time.monotonic()
                print(json.dumps({"optimizer_steps": step + 1,
                                  "distinct_roots_seen": manifest["distinct_roots_seen"],
                                  "stage_draws": draws, "loss": manifest["loss"]}),
                      flush=True)
        if manifest["optimizer_steps"] == CONFIG["steps"]:
            manifest.update(status="completed", pid=None)
            atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
