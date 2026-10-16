"""First resumable Q-run-like policy bootstrap from 140 verified TRAIN paths."""

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
PARENT = Path("artifacts/mcts-tournament-nine-arm-pilot-v1/F-step-0144.pt")
PARENT_SHA = "0deb9418ea71a432644b7370c84879cf887536dcd3757280466e73ed749ff227"
CONFIG = {"steps": 256, "batch_roots": 4, "checkpoint_interval": 16,
          "seed": 20261004, "lr": 1e-4, "weight_decay": 0.01,
          "grad_clip": 1.0, "device": "mps"}


def read(path):
    return json.loads(path.read_text())


def build_bank():
    roster = read(ROSTER)
    rows = roster["stages"][0]
    if len(rows) != 140 or len({tuple(r["query_group"]) for r in rows}) != 140:
        raise ValueError("first-rung root coverage differs")
    bank = []
    for root in rows:
        witness = UnknotWitness.from_dict(root["witness"])
        witness.verify()
        paired = []
        for instance_dict in root["instances"]:
            instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
            trace = verified_success_traces(instance, [witness])[0]
            paired.append((instance, trace["rows"]))
        if tuple(pair[0].cc_weight for pair in paired) != (10, 1000):
            raise ValueError("L10/L1000 resource pair differs")
        bank.append(paired)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    if (not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"):
        raise RuntimeError("require native MPS without CPU fallback")
    torch.set_num_threads(1)
    pins = (ROSTER, PARENT, Path(__file__),
            Path("research/279-qrun-like-policy-bootstrap-rung0-v1.md"),
            Path("scripts/pretrained_raster_policy.py"),
            Path("scripts/mcts_training_supervision.py"),
            Path("src/rf_knots/site_proposals.py"))
    inputs = {str(path): sha(path) for path in pins}
    if inputs[str(ROSTER)] != ROSTER_SHA or inputs[str(PARENT)] != PARENT_SHA:
        raise ValueError("roster or F144 checkpoint changed")
    contract = {"schema": "rf-knots-qrun-like-policy-bootstrap-rung0-v1",
                "config": CONFIG, "inputs": inputs,
                "role": "verified-solution policy warmup, not F-dominant R2/R4"}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        manifest = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "optimizer_steps": 0, "checkpoints": [],
            "seconds_committed": 0.0}
        if (manifest["contract"] != json.loads(json.dumps(contract))
                or manifest["contract_sha256"] != contract_sha
                or manifest["optimizer_steps"] != 16 * len(manifest["checkpoints"])):
            raise ValueError("resume contract/steps/checkpoints differ")
        for index, entry in enumerate(manifest["checkpoints"], 1):
            if entry["step"] != index * 16 or sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("committed checkpoint changed")
        if manifest["optimizer_steps"] == CONFIG["steps"]:
            manifest["status"] = "completed"
            atomic(manifest_path, manifest)
            return
        bank = build_bank()
        net, objective = load_actor()
        parent = torch.load(PARENT, map_location="cpu", weights_only=False)
        if parent["arm"] != "F" or parent["step"] != 144:
            raise ValueError("F parent identity differs")
        net.load_state_dict(parent["model"], strict=True)
        objective.load_state_dict(parent["objective"], strict=True)
        net.to("mps")
        optimizer = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"],
                                      weight_decay=CONFIG["weight_decay"])
        rng = random.Random(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])
        torch.mps.manual_seed(CONFIG["seed"])
        if manifest["optimizer_steps"]:
            entry = manifest["checkpoints"][-1]
            saved = torch.load(args.out / entry["file"], map_location="cpu", weights_only=False)
            if saved["contract_sha256"] != contract_sha or saved["step"] != entry["step"]:
                raise ValueError("resume checkpoint contract differs")
            net.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            rng.setstate(saved["rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.mps.set_rng_state(saved["mps_rng"])
        start = manifest["optimizer_steps"]
        manifest.update(status="training", pid=os.getpid())
        atomic(manifest_path, manifest)
        end = min(CONFIG["steps"], start + (args.stop_after or CONFIG["steps"]))
        checkpoint_started = time.monotonic()
        for step in range(start, end):
            net.train()
            requests, targets = [], []
            for offset in range(CONFIG["batch_roots"]):
                root_index = (step * CONFIG["batch_roots"] + offset) % len(bank)
                instance, trace = bank[root_index][rng.randrange(2)]
                state, action = trace[rng.randrange(len(trace))]
                actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | {action}))
                requests.append((instance, state, actions))
                targets.append(actions.index(action))
            optimizer.zero_grad(set_to_none=True)
            logp = policy_log_probabilities(net, requests)
            chosen = logp[torch.arange(len(requests), device=logp.device), targets]
            loss = -chosen.mean()
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
                         "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state()}
                checkpoint_path = args.out / f"step-{step + 1:04d}.pt"
                partial = checkpoint_path.with_suffix(".partial")
                torch.save(saved, partial)
                partial.replace(checkpoint_path)
                manifest["checkpoints"].append({"file": checkpoint_path.name,
                                                "sha256": sha(checkpoint_path),
                                                "step": step + 1})
                manifest["optimizer_steps"] = step + 1
                manifest["seconds_committed"] += time.monotonic() - checkpoint_started
                manifest.update(loss=float(loss.detach()), grad_norm=float(grad),
                                distinct_roots_seen=min(140, 4 * (step + 1)))
                atomic(manifest_path, manifest)
                checkpoint_started = time.monotonic()
                print(json.dumps({"optimizer_steps": step + 1, "loss": manifest["loss"],
                                  "distinct_roots_seen": manifest["distinct_roots_seen"]}), flush=True)
        if manifest["optimizer_steps"] == CONFIG["steps"]:
            manifest.update(status="completed", pid=None)
            atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
