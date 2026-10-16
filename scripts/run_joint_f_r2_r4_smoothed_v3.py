"""Serial, resumable native-MPS scheduler for smoothed F R2/R4."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_joint_f_r2_r4_smoothed_v3.py"
SOURCE_SHA = "8fcede9f5d09266f37b7ed6c5589bd1a93177b0c94b82d1357c62dce77c96f6e"
OUT = ROOT / "artifacts/joint-f-r2-r4-smoothed-v3"
BLOCK = 128
TARGET_SECONDS = 12 * 3600


def manifest(arm):
    path = ROOT / f"artifacts/joint-f-r2-r4-smoothed-v3-{arm}/manifest.json"
    data = json.loads(path.read_text())
    if data["contract"]["arm"] != arm:
        raise ValueError("arm identity differs")
    if data["contract"]["config"]["v3_source_sha256"] != SOURCE_SHA:
        raise ValueError("smoothed trainer source pin differs")
    entry = data["checkpoints"][-1]
    checkpoint = path.parent / entry["file"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("latest checkpoint SHA differs")
    if data["optimizer_steps"] != entry["step"]:
        raise ValueError("uncommitted optimizer steps")
    metrics = data["metrics"]
    if not math.isfinite(metrics["F_loss"]) or metrics["F_loss"] < 0:
        raise ValueError("smoothed F loss invalid")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-seconds", type=float, default=TARGET_SECONDS)
    args = parser.parse_args()
    if hashlib.sha256(TRAINER.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("smoothed trainer source changed")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            states = {arm: manifest(arm) for arm in ("R2", "R4")}
            steps = {arm: states[arm]["optimizer_steps"] for arm in states}
            if abs(steps["R2"] - steps["R4"]) > BLOCK:
                raise ValueError("arms differ by more than one scheduled block")
            if steps["R2"] == steps["R4"] and all(
                states[arm]["active_seconds"] >= args.target_seconds for arm in states
            ):
                print(json.dumps({"status": "complete", "steps": steps,
                                  "active_seconds": {arm: states[arm]["active_seconds"]
                                                     for arm in states}}), flush=True)
                return
            arm = "R2" if steps["R2"] <= steps["R4"] else "R4"
            command = [sys.executable, str(TRAINER), "--arm", arm,
                       "--out", str(ROOT / f"artifacts/joint-f-r2-r4-smoothed-v3-{arm}"),
                       "--resume", "--stop-after", str(BLOCK)]
            print(json.dumps({"status": "block_start", "arm": arm,
                              "steps_before": steps, "command": command,
                              "time": time.time()}), flush=True)
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
            updated = manifest(arm)
            if updated["optimizer_steps"] != steps[arm] + BLOCK:
                raise ValueError("block optimizer steps incomplete")
            print(json.dumps({"status": "block_complete", "arm": arm,
                              "steps": updated["optimizer_steps"],
                              "active_seconds": updated["active_seconds"],
                              "F_loss": updated["metrics"]["F_loss"],
                              "time": time.time()}), flush=True)


if __name__ == "__main__":
    main()
