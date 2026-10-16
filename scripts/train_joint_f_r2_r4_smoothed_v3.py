"""Label-smoothed exact-context F repair of the matched R2/R4 trainer."""

import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

import train_joint_f_r2_r4_bounded_v2 as bounded


MARGIN = 0.5
ERROR_MASS = 0.02


def smoothed_pairwise_loss(logp, positive, negative, positions,
                           pos_mass, neg_mass):
    result = None
    for positive_action, positive_weight in positive:
        for negative_action, negative_weight in negative:
            z = MARGIN + logp[positions[negative_action]] \
                - logp[positions[positive_action]]
            pair = (1 - ERROR_MASS) * F.softplus(z) + ERROR_MASS * F.softplus(-z)
            weight = positive_weight * negative_weight / (pos_mass * neg_mass)
            result = pair * weight if result is None else result + pair * weight
    return result


if __name__ == "__main__":
    bounded.pairwise_margin_loss = smoothed_pairwise_loss
    bounded.base.PROTOCOL = Path("research/306-f-smoothed-pairwise-repair-v1.md")
    bounded.base.CONFIG["F_loss"] = "exact-context-smoothed-pairwise-softplus"
    bounded.base.CONFIG["F_margin"] = MARGIN
    bounded.base.CONFIG["F_error_mass"] = ERROR_MASS
    bounded.base.CONFIG["v2_source_sha256"] = hashlib.sha256(
        Path(bounded.__file__).read_bytes()).hexdigest()
    bounded.base.CONFIG["v3_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    bounded.base.train_step = bounded.train_step
    bounded.base.main()
