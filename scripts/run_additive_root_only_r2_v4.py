"""Single-writer native-MPS R2 F-first blocks to six new active hours."""

import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from build_mcts_training_supervision import sha

OUT = Path("artifacts/additive-root-only-r2-v4")
TRAINER = Path("scripts/train_additive_root_only_r2_v4.py")
TRAINER_SHA = "7cad3166d1f35f8794c705af68b25b2a6f91614de5528c36d325162e08c4d749"
PROTOCOL = Path("research/353-continuous-r2-f-mcts-rehearsal-v4.md")
PROTOCOL_SHA = "207bdb72d8f7daaa6e8ba4ba6a88191f890ca24196c7568e2d30b930d3b87e1a"
CONTRACT_SHA = "0e8550cd457392b2965eac3e6e60102b2159ea4fc2efa9f3c43b180b62ce015b"
TARGET_NEW_ACTIVE_SECONDS = 21600.0
MIN_FREE_BYTES = 10 * 1024**3


def read():
    return json.loads((OUT / "manifest.json").read_text())


def check(manifest):
    if sha(TRAINER) != TRAINER_SHA or sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("frozen trainer/protocol changed")
    if (manifest["contract_sha256"] != CONTRACT_SHA
            or manifest["contract"]["config"]["sampler"] !=
            "retained-stage-90/new-root-only-10"
            or manifest["status"] != "checkpointed"
            or manifest["optimizer_steps"] != manifest["checkpoints"][-1]["step"]):
        raise ValueError("R2 v4 checkpoint contract differs")
    last = manifest["checkpoints"][-1]
    if sha(OUT / last["file"]) != last["sha256"]:
        raise ValueError("latest checkpoint SHA differs")
    metrics = manifest["metrics"]
    f = metrics["F_encoder_grad_norm"]
    if (not math.isfinite(metrics["F_loss"]) or metrics["F_loss"] < 0
            or not math.isfinite(f) or f <= 1e-6
            or metrics["aux_scale"] * metrics["aux_encoder_grad_norm"]
            > .100001 * f
            or metrics["rehearsal_scale"] * metrics["rehearsal_encoder_grad_norm"]
            > .200001 * f):
        raise ValueError("F/aux/rehearsal gradient gate failed")
    if shutil.disk_usage(OUT).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 10 GiB free; no further training block")


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
