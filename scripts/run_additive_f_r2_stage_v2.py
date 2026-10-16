"""Single-writer native-MPS R2 blocks to two new stage-aware active hours."""

import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from build_mcts_training_supervision import sha

OUT = Path("artifacts/additive-f-r2-stage-v2")
TRAINER = Path("scripts/train_additive_f_r2_stage_v2.py")
TRAINER_SHA = "b128530abb0ce30163873b21f9961a2f07a72de58d96960d3043dfb0cd1c58da"
PROTOCOL = Path("research/322-additive-r2-stage-aware-continuation-v2.md")
PROTOCOL_SHA = "88be6cb0a063ff33abb34f8c171e91a0f8564eb1242539e28e4bc0ee6c9c18c0"
CONTRACT_SHA = "14ebfc498faf18a7ff40ed55d6c3b2d8adc5e48fac7973e960df9b282d4bcf56"
TARGET_NEW_ACTIVE_SECONDS = 7200.0


def read():
    return json.loads((OUT / "manifest.json").read_text())


def check(manifest):
    if sha(TRAINER) != TRAINER_SHA or sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("frozen trainer/protocol changed")
    if (manifest["contract_sha256"] != CONTRACT_SHA
            or manifest["contract"]["config"]["sampler"] != "additive-stage-aware-v2"
            or manifest["status"] != "checkpointed"
            or manifest["optimizer_steps"] != manifest["checkpoints"][-1]["step"]):
        raise ValueError("stage-aware checkpoint contract differs")
    last = manifest["checkpoints"][-1]
    if sha(OUT / last["file"]) != last["sha256"]:
        raise ValueError("latest checkpoint SHA differs")
    metrics = manifest["metrics"]
    f = metrics["F_encoder_grad_norm"]
    if (not math.isfinite(metrics["F_loss"]) or metrics["F_loss"] < 0
            or not math.isfinite(f) or f <= 1e-12
            or metrics["aux_scale"] * metrics["aux_encoder_grad_norm"] > .100001 * f
            or metrics["rehearsal_scale"] * metrics["rehearsal_encoder_grad_norm"]
            > .200001 * f):
        raise ValueError("F/aux/rehearsal gradient gate failed")


def main():
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = "src:scripts:/Users/artemvorozhtsov/projects/pgx-mcts-bench/src"
    with (OUT / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            manifest = read()
            check(manifest)
            if manifest["new_active_seconds"] >= TARGET_NEW_ACTIVE_SECONDS:
                print(json.dumps({"status": "target-reached",
                                  "step": manifest["optimizer_steps"],
                                  "new_active_seconds": manifest["new_active_seconds"]}),
                      flush=True)
                return
            command = [sys.executable, str(TRAINER), "--out", str(OUT),
                       "--resume", "--stop-after", "128"]
            started = time.monotonic()
            subprocess.run(command, env=env, check=True)
            updated = read()
            check(updated)
            if updated["optimizer_steps"] != manifest["optimizer_steps"] + 128:
                raise ValueError("optimizer step advancement differs")
            print(json.dumps({"step": updated["optimizer_steps"],
                              "new_active_seconds": updated["new_active_seconds"],
                              "block_wall_seconds": time.monotonic() - started}),
                  flush=True)


if __name__ == "__main__":
    main()
