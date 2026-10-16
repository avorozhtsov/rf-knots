"""Storage-safe continuation of v3 with one checkpoint per 128 steps."""

import hashlib
from pathlib import Path

import train_joint_f_r2_r4_bounded_v2 as bounded
from train_joint_f_r2_r4_smoothed_v3 import (
    ERROR_MASS, MARGIN, smoothed_pairwise_loss,
)


if __name__ == "__main__":
    bounded.pairwise_margin_loss = smoothed_pairwise_loss
    bounded.base.PROTOCOL = Path("research/308-f-v4-storage-safe-continuation.md")
    bounded.base.CONFIG["F_loss"] = "exact-context-smoothed-pairwise-softplus"
    bounded.base.CONFIG["F_margin"] = MARGIN
    bounded.base.CONFIG["F_error_mass"] = ERROR_MASS
    bounded.base.CONFIG["v2_source_sha256"] = hashlib.sha256(
        Path(bounded.__file__).read_bytes()).hexdigest()
    bounded.base.CONFIG["v3_source_sha256"] = hashlib.sha256(
        Path("scripts/train_joint_f_r2_r4_smoothed_v3.py").read_bytes()).hexdigest()
    bounded.base.CONFIG["v4_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    bounded.base.CONFIG["checkpoint_interval"] = 128
    bounded.base.train_step = bounded.train_step
    bounded.base.main()
