"""Single-writer two-hour R2 v5 blocks with a 20-GiB disk gate."""

from pathlib import Path

import run_additive_root_only_r2_v4 as scheduler

scheduler.OUT = Path("artifacts/additive-root-only-r2-v5")
scheduler.TRAINER = Path("scripts/train_additive_root_only_r2_v5.py")
scheduler.TRAINER_SHA = "ddb2bf2af3aef2ec071086709868d120eaee37c221b50cc31a52ca02eee71717"
scheduler.PROTOCOL = Path("research/359-r2-f-continuation-v5-storage-gate.md")
scheduler.PROTOCOL_SHA = "f22ac81f1becb7a20a96bc1f7534ac65674c4d98470d3e20c472cfd1b13d95bd"
scheduler.CONTRACT_SHA = "94ea340f01d430de6a2bd748e9cd7d205cec9a71fb0534c3e62983766dafaebe"
scheduler.TARGET_NEW_ACTIVE_SECONDS = 7200.0
scheduler.MIN_FREE_BYTES = 20 * 1024**3


if __name__ == "__main__":
    scheduler.main()
