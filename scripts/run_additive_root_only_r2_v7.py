"""Single-writer bounded R2 v7 blocks with a 24-GiB disk gate."""

from pathlib import Path

import run_additive_root_only_r2_v4 as scheduler

scheduler.OUT = Path("artifacts/additive-root-only-r2-v7")
scheduler.TRAINER = Path("scripts/train_additive_root_only_r2_v7.py")
scheduler.TRAINER_SHA = "e014f6cd745163e7460f2f6f41d35eaeef0c71db347f712f13f5742609784d14"
scheduler.PROTOCOL = Path("research/364-r2-v7-storage-bounded-continuation.md")
scheduler.PROTOCOL_SHA = "2cea6701f69354dfd985c2877c009b06eb2827934b31a86e3b1408b13b241fa3"
scheduler.CONTRACT_SHA = "4ec7097399f7b5ee32e5acd0e445c588a75f2fe844a2a7ee471136d9a2b51e93"
scheduler.TARGET_NEW_ACTIVE_SECONDS = 3600.0
scheduler.MIN_FREE_BYTES = 24 * 1024**3


if __name__ == "__main__":
    scheduler.main()
