"""Single-writer 128-step MPS blocks to two new checkpointed active hours."""

import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from build_mcts_training_supervision import sha

OUT = Path("artifacts/additive-root-only-r2-v3")
TRAINER = Path("scripts/train_additive_root_only_r2_v3.py")
TRAINER_SHA = "7cef6247131b74c22c66dec6640a109a95b49741dbeeb16d9f07cbf1ab8fdcf5"
PROTOCOL = Path("research/345-additive-root-only-r2-training-v3.md")
PROTOCOL_SHA = "39f7bd227b21e5c06c892d2c0cae40b24ea054ca637cc589d91292ef94e5131e"
CONTRACT_SHA = "b4306f11a248f456b6649eb7af190b4c062d0fb69f1fed4be978128ed2cc9980"
TARGET_NEW_ACTIVE_SECONDS = 7200.0


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
        raise ValueError("R2 root-only checkpoint contract differs")
    last = manifest["checkpoints"][-1]
    if sha(OUT / last["file"]) != last["sha256"]:
        raise ValueError("latest checkpoint SHA differs")
    metrics = manifest["metrics"]
    f = metrics["F_encoder_grad_norm"]
    if (not math.isfinite(metrics["F_loss"]) or metrics["F_loss"] < 0
            or not math.isfinite(f) or f <= 1e-12
            or metrics["aux_scale"] * metrics["aux_encoder_grad_norm"]
            > .100001 * f
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
