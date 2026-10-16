"""Continuous single-writer R2 optimizer blocks with disk/numerical gates."""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from build_mcts_training_supervision import atomic, sha

OUT = Path("artifacts/additive-root-only-r2-v10")
TRAINER = Path("scripts/train_additive_root_only_r2_v10.py")
PROTOCOL = Path("research/375-continuous-r2-mcts-learning-contract-v1.md")
BLOCK = 1024
MIN_FREE = 12 * 1024**3


def read():
    return json.loads((OUT / "manifest.json").read_text())


def check(state):
    if (state["status"] != "checkpointed" or state["contract"]["arm"] != "R2"
            or state["contract"]["block_steps"] != BLOCK
            or state["contract"]["config"]["continuation_mode"]
            != "continuous-until-explicit-or-resource-gate"):
        raise ValueError("continuing optimizer contract differs")
    for path, expected in state["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned optimizer input changed: {path}")
    entry = state["checkpoints"][-1]
    path = OUT / entry["file"]
    if (entry["step"] != state["optimizer_steps"] or sha(path) != entry["sha256"]):
        raise ValueError("latest checkpoint SHA/step differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["step"] != entry["step"]
            or saved["contract_sha256"] != state["contract_sha256"]
            or not saved["optimizer"]["state"]
            or not all(key in saved for key in ("rng", "torch_rng", "mps_rng"))):
        raise ValueError("optimizer or RNG missing from full checkpoint")
    metrics = state["metrics"]
    f = metrics["F_encoder_grad_norm"]
    if (not math.isfinite(metrics["F_loss"]) or metrics["F_loss"] < 0
            or not math.isfinite(f) or f <= 1e-6
            or metrics["aux_scale"] * metrics["aux_encoder_grad_norm"] > .100001 * f
            or metrics["rehearsal_scale"] * metrics["rehearsal_encoder_grad_norm"] > .200001 * f):
        raise ValueError("F/aux/rehearsal numerical gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-blocks", type=int, default=0,
                        help="operational check; zero means keep running")
    args = parser.parse_args()
    if args.max_blocks < 0:
        raise ValueError("negative block limit")
    env = os.environ.copy()
    env.update(PYTORCH_ENABLE_MPS_FALLBACK="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH="src:scripts:/Users/artemvorozhtsov/projects/pgx-mcts-bench/src")
    with (OUT / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocks = 0
        while True:
            state = read()
            check(state)
            reason = ("user_stop_request" if (OUT / "STOP_REQUEST").exists()
                      else "storage_gate" if shutil.disk_usage(OUT).free < MIN_FREE
                      else "operational_limit" if args.max_blocks and blocks >= args.max_blocks
                      else None)
            if reason:
                atomic(OUT / "scheduler-status.json", {
                    "status": reason, "step": state["optimizer_steps"],
                    "free_bytes": shutil.disk_usage(OUT).free,
                    "timestamp": time.time()})
                print(json.dumps({"status": reason}), flush=True)
                return
            atomic(OUT / "scheduler-status.json", {
                "status": "running", "pid": os.getpid(),
                "step": state["optimizer_steps"], "timestamp": time.time(),
                "scheduler_sha256": sha(Path(__file__))})
            subprocess.run([sys.executable, str(TRAINER), "--out", str(OUT),
                            "--resume", "--stop-after", str(BLOCK)], env=env, check=True)
            updated = read()
            check(updated)
            if updated["optimizer_steps"] != state["optimizer_steps"] + BLOCK:
                raise ValueError("optimizer block advancement differs")
            blocks += 1
            with (OUT / "block-history.jsonl").open("a") as history:
                history.write(json.dumps({"step": updated["optimizer_steps"],
                                          "new_active_seconds": updated["new_active_seconds"],
                                          "metrics": updated["metrics"],
                                          "timestamp": time.time()}) + "\n")


if __name__ == "__main__":
    main()
