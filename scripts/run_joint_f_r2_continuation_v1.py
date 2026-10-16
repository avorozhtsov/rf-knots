"""Single-writer 128-step R2 MPS blocks until two new active hours are saved."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from build_mcts_training_supervision import sha

OUT = Path("artifacts/joint-f-r2-continuation-v1")
TRAINER = Path("scripts/train_joint_f_r2_continuation_v1.py")
TRAINER_SHA = "8cd72290774112c2cc6eb0dc7f7d0960c38350c3620f9d8dbf99ad49d11442f5"
PROTOCOL = Path("research/314-r2-only-f-continuation-v1.md")
PROTOCOL_SHA = "97c5a81167086b572216974247a1d0fb40f61a65d9b66f027b2e1c30c716e6e6"
CONTRACT_SHA = "5507279e87245faa8f461c8645ac6f9534c91171d1377a67ae2609b8b2b14dff"
TARGET_NEW_ACTIVE_SECONDS = 7200.0


def read():
    return json.loads((OUT / "manifest.json").read_text())


def check(manifest):
    if sha(TRAINER) != TRAINER_SHA or sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("frozen trainer/protocol changed")
    if manifest["contract_sha256"] != CONTRACT_SHA:
        raise ValueError("R2 continuation contract changed")
    if (manifest["status"] != "checkpointed" or
            manifest["optimizer_steps"] != manifest["checkpoints"][-1]["step"]):
        raise ValueError("R2 continuation not at a committed checkpoint")
    last = manifest["checkpoints"][-1]
    if sha(OUT / last["file"]) != last["sha256"]:
        raise ValueError("R2 continuation checkpoint SHA changed")
    metrics = manifest["metrics"]
    if (metrics["F_loss"] < 0 or metrics["F_encoder_grad_norm"] <= 1e-12 or
            metrics["aux_scale"] * metrics["aux_encoder_grad_norm"] >
            .100001 * metrics["F_encoder_grad_norm"] or
            metrics["rehearsal_scale"] * metrics["rehearsal_encoder_grad_norm"] >
            .200001 * metrics["F_encoder_grad_norm"]):
        raise ValueError("F or gradient ratio gate failed")


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
                print(json.dumps({"status": "target-reached", "step": manifest["optimizer_steps"],
                                  "new_active_seconds": manifest["new_active_seconds"]}), flush=True)
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
                              "block_wall_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
