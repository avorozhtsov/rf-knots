"""GPU-compatible metric objective for global knot and diagram embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def witness_priority_weight(metric: str, upper: float, semantic_length: int) -> float:
    """Priority for long CC=0/1 witnesses requested by the training contract."""
    if metric != "cc" or upper > 1:
        return 1.0
    if semantic_length > 20:
        return 8.0
    if semantic_length > 10:
        return 4.0
    if semantic_length > 5:
        return 2.0
    return 1.0


@dataclass(frozen=True)
class MetricBatch:
    metric: tuple[str, ...]
    lower: Tensor
    upper: Tensor
    witness_semantic_length: Tensor


class MetricEmbeddingObjective(nn.Module):
    """Interval metric loss over caller-supplied global embeddings.

    The network remains free to produce representations in any order. This
    objective compares state pairs, not a prescribed next action or trajectory.
    Separate heads preserve knot/CC and diagram-isotopy geometry.
    """

    def __init__(self, minimum_scale: float = 0.25):
        super().__init__()
        self.minimum_scale = minimum_scale
        self.cc_scale_raw = nn.Parameter(torch.tensor(0.5))
        self.rm_scale_raw = nn.Parameter(torch.tensor(0.5))

    def scale(self, metric: str) -> Tensor:
        raw = self.cc_scale_raw if metric == "cc" else self.rm_scale_raw
        return F.softplus(raw) + self.minimum_scale

    def forward(
        self,
        left: dict[str, Tensor],
        right: dict[str, Tensor],
        batch: MetricBatch,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if len(batch.metric) != batch.lower.shape[0]:
            raise ValueError("metric batch length mismatch")
        total = batch.lower.new_zeros(())
        diagnostics = {}
        active = 0
        for metric, head in (("cc", "knot"), ("rm", "isotopy")):
            indices = [index for index, value in enumerate(batch.metric) if value == metric]
            if not indices:
                continue
            index = torch.as_tensor(indices, device=batch.lower.device)
            cosine = (left[head][index] * right[head][index]).sum(-1)
            predicted = self.scale(metric) * (1.0 - cosine)
            lower = batch.lower[index]
            upper = batch.upper[index]
            lengths = batch.witness_semantic_length[index]
            weights = torch.ones_like(predicted)
            if metric == "cc":
                eligible = upper <= 1
                weights = torch.where(eligible & (lengths > 5), 2.0, weights)
                weights = torch.where(eligible & (lengths > 10), 4.0, weights)
                weights = torch.where(eligible & (lengths > 20), 8.0, weights)
            violation = F.relu(lower - predicted).square() + F.relu(predicted - upper).square()
            metric_loss = (violation * weights).sum() / weights.sum().clamp_min(1.0)
            total = total + metric_loss
            diagnostics[f"{metric}_loss"] = metric_loss.detach()
            diagnostics[f"{metric}_mean_predicted"] = predicted.mean().detach()
            active += 1
        if active == 0:
            raise ValueError("metric batch contains neither cc nor rm rows")
        return total / active, diagnostics
