"""Single-writer bounded R2 v8 blocks with a 24-GiB disk gate."""

from pathlib import Path

import run_additive_root_only_r2_v4 as scheduler

scheduler.OUT = Path("artifacts/additive-root-only-r2-v8")
scheduler.TRAINER = Path("scripts/train_additive_root_only_r2_v8.py")
scheduler.TRAINER_SHA = "8bcaa8bfbadda3dcb849c9b9850c049847f38b1a71f67fc7dac8bd9e34ead1bd"
scheduler.PROTOCOL = Path("research/366-r2-v8-storage-bounded-continuation.md")
scheduler.PROTOCOL_SHA = "fa19c749a1f84ac3a644b1e220c89592b9d020e122ca0499fd35491701d654c0"
scheduler.CONTRACT_SHA = "4163732cb6705f2a2bdcd2cc3506f479f8b3234cf214a91f0ea844d1cba1001e"
scheduler.TARGET_NEW_ACTIVE_SECONDS = 3600.0
scheduler.MIN_FREE_BYTES = 24 * 1024**3


if __name__ == "__main__":
    scheduler.main()
