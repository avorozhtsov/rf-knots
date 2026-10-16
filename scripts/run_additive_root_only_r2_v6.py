"""Single-writer one-hour R2 v6 blocks with a 24-GiB disk gate."""

from pathlib import Path

import run_additive_root_only_r2_v4 as scheduler

scheduler.OUT = Path("artifacts/additive-root-only-r2-v6")
scheduler.TRAINER = Path("scripts/train_additive_root_only_r2_v6.py")
scheduler.TRAINER_SHA = "7272abb1731a3e06867a14539af3a051e144b2f821bc17ea16affcf90b1b2b49"
scheduler.PROTOCOL = Path("research/361-r2-v6-low-disk-continuation.md")
scheduler.PROTOCOL_SHA = "467914049b219bd6c4fa1f8e0a869acfaa84874b8658bb8e4720ee1e86dbf5b3"
scheduler.CONTRACT_SHA = "6b5fa76e5f2884be68a48bb97ceffbf3141bc285cad81eac49fe24d145083a82"
scheduler.TARGET_NEW_ACTIVE_SECONDS = 3600.0
scheduler.MIN_FREE_BYTES = 24 * 1024**3


if __name__ == "__main__":
    scheduler.main()
