"""Single-writer short R2 v9 blocks with a 24-GiB disk gate."""

from pathlib import Path

import run_additive_root_only_r2_v4 as scheduler

scheduler.OUT = Path("artifacts/additive-root-only-r2-v9")
scheduler.TRAINER = Path("scripts/train_additive_root_only_r2_v9.py")
scheduler.TRAINER_SHA = "5e9d2410ce5dbdc5a8b67e59e36e5b45328e2340d8edcfcabc08d87358750e71"
scheduler.PROTOCOL = Path("research/368-r2-v9-short-storage-gated-continuation.md")
scheduler.PROTOCOL_SHA = "bdc1b263f5cde38c021b4dc2531e9a417ef088dff246490c9418bd71f6b8652b"
scheduler.CONTRACT_SHA = "57cd9361ac453ab4a6ad04dc58a5c6dc9e2ef63730afd0a860807d7c676daeb8"
scheduler.TARGET_NEW_ACTIVE_SECONDS = 1800.0
scheduler.MIN_FREE_BYTES = 24 * 1024**3


if __name__ == "__main__":
    scheduler.main()
