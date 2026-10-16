"""Separate, untrained value head for a frozen raster policy/encoder checkpoint.

The three encoder vectors are produced by the existing shared raster encoder.
This module does not turn a checkpoint into a trained value model by itself.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def estimate_local_simulations(root_remaining, path, *, prior_pseudocount=10.0):
    """Expected local visits, including the current selected simulation.

    Each path entry is (all sibling visit counts, normalized policy priors,
    selected child index), ordered from root toward the selected node. This
    freezes current visit proportions; actual adaptive MCTS may reallocate.
    """
    if type(root_remaining) is not int or root_remaining < 0:
        raise ValueError("root_remaining must be a nonnegative integer")
    if not math.isfinite(prior_pseudocount) or prior_pseudocount <= 0:
        raise ValueError("positive finite prior pseudocount required")
    if root_remaining == 0:
        return 0.0
    share = 1.0
    for visits, priors, selected in path:
        if (
            len(visits) != len(priors)
            or len(visits) == 0
            or type(selected) is not int
            or not 0 <= selected < len(visits)
            or any(type(n) is not int or n < 0 for n in visits)
            or any(not math.isfinite(p) or p < 0 for p in priors)
            or not math.isclose(sum(priors), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError("invalid tree visits, priors, or selected child")
        share *= (visits[selected] + prior_pseudocount * priors[selected]) / (
            sum(visits) + prior_pseudocount
        )
    return 1.0 + (root_remaining - 1) * share


def optimistic_local_simulations(
    root_remaining, expected_local, *, minimum=2, additive=0.0, relative=0.0
):
    """Selection-only budget credit, never a calibrated visit forecast.

    The credit cannot exceed the root's remaining simulations. Keep the
    unbiased `expected_local` separately for calibration and reporting.
    """
    if type(root_remaining) is not int or root_remaining < 0:
        raise ValueError("root_remaining must be a nonnegative integer")
    if (
        type(minimum) is not int
        or minimum < 0
        or not math.isfinite(expected_local)
        or not math.isfinite(additive)
        or not math.isfinite(relative)
        or not 0 <= expected_local <= root_remaining
        or additive < 0
        or relative < 0
    ):
        raise ValueError("invalid optimistic budget parameters")
    if root_remaining == 0:
        return 0.0
    return min(
        float(root_remaining),
        max(float(minimum), expected_local * (1.0 + relative) + additive),
    )


class BudgetedCostValueHead(nn.Module):
    def __init__(self, embedding_dim=64, width=128, reference_simulations=1024):
        super().__init__()
        if embedding_dim < 1 or width < 1 or reference_simulations < 1:
            raise ValueError("invalid value-head dimensions or reference budget")
        self.embedding_dim = embedding_dim
        self.reference_simulations = reference_simulations
        self.body = nn.Sequential(
            nn.Linear(3 * embedding_dim + 4, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )

    def forward(self, encoded, *, cc_left, moves_left, cc_weight, local_simulations):
        vectors = [encoded[name] for name in ("knot", "layout", "isotopy")]
        if any(v.ndim != 2 or v.shape[1] != self.embedding_dim for v in vectors):
            raise ValueError("expected three shared-encoder vectors")
        if any(v.shape != vectors[0].shape for v in vectors):
            raise ValueError("encoder vector batch shapes differ")
        batch, _ = vectors[0].shape
        context = torch.stack(
            [
                torch.as_tensor(x, dtype=vectors[0].dtype, device=vectors[0].device)
                for x in (cc_left, moves_left, cc_weight, local_simulations)
            ],
            dim=-1,
        )
        if context.shape != (batch, 4) or not torch.isfinite(context).all():
            raise ValueError("budget context must be finite with one row per state")
        if (
            (context[:, 0] < 0).any()
            or (context[:, 1] < 0).any()
            or (context[:, 2] <= 0).any()
            or (context[:, 3] < 1).any()
        ):
            raise ValueError("invalid remaining resources, objective, or local budget")
        features = torch.stack(
            (
                torch.log1p(context[:, 0]) / math.log1p(32),
                torch.log1p(context[:, 1]) / math.log1p(512),
                torch.log(context[:, 2]) / math.log(1000),
                torch.log1p(context[:, 3]) / math.log1p(self.reference_simulations),
            ),
            dim=-1,
        )
        raw = self.body(torch.cat((*vectors, features), dim=-1))
        return {
            "solve_logit": raw[:, 0],
            "p_solve": torch.sigmoid(raw[:, 0]),
            "cc_remaining": torch.sigmoid(raw[:, 1]) * context[:, 0],
            "moves_remaining": torch.sigmoid(raw[:, 2]) * context[:, 1],
        }


def value_loss(prediction, *, solved, cc, moves, cc_left, moves_left):
    """A censored failure trains success probability, never the paired costs."""
    logit = prediction["solve_logit"]
    solved = torch.as_tensor(solved, dtype=logit.dtype, device=logit.device)
    cc = torch.as_tensor(cc, dtype=logit.dtype, device=logit.device)
    moves = torch.as_tensor(moves, dtype=logit.dtype, device=logit.device)
    cc_left = torch.as_tensor(cc_left, dtype=logit.dtype, device=logit.device)
    moves_left = torch.as_tensor(moves_left, dtype=logit.dtype, device=logit.device)
    if any(x.shape != logit.shape for x in (solved, cc, moves, cc_left, moves_left)):
        raise ValueError("value-label batch shapes differ")
    if (
        not torch.isfinite(solved).all()
        or not torch.all((solved == 0) | (solved == 1))
        or (cc_left < 0).any()
        or (moves_left < 0).any()
    ):
        raise ValueError("invalid success/resource labels")
    success = solved.bool()
    if success.any() and (
        not torch.isfinite(cc[success]).all()
        or not torch.isfinite(moves[success]).all()
        or (cc[success] < 0).any()
        or (moves[success] < 0).any()
        or (cc[success] > cc_left[success]).any()
        or (moves[success] > moves_left[success]).any()
    ):
        raise ValueError("successful path costs exceed node-relative resources")
    solve_loss = F.binary_cross_entropy_with_logits(logit, solved)
    if not success.any():
        return {"total": solve_loss, "solve": solve_loss, "cost": logit.new_zeros(())}
    cc_error = F.smooth_l1_loss(
        prediction["cc_remaining"][success] / cc_left[success].clamp_min(1),
        cc[success] / cc_left[success].clamp_min(1),
    )
    move_error = F.smooth_l1_loss(
        prediction["moves_remaining"][success] / moves_left[success].clamp_min(1),
        moves[success] / moves_left[success].clamp_min(1),
    )
    cost_loss = cc_error + move_error
    return {"total": solve_loss + cost_loss, "solve": solve_loss, "cost": cost_loss}
