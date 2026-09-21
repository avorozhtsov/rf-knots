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


CC_GEOMETRIES = ("chord", "angular", "log_affinity")


class MetricEmbeddingObjective(nn.Module):
    """Interval metric loss over caller-supplied global embeddings.

    The network remains free to produce representations in any order. This
    objective compares state pairs, not a prescribed next action or trajectory.
    Separate heads preserve knot/CC and diagram-isotopy geometry.
    """

    def __init__(
        self,
        minimum_scale: float = 0.25,
        cc_geometry: str = "chord",
        cc_rank_weight: float = 0.0,
        cc_rank_margin: float = 0.25,
        witness_priority: bool = True,
    ):
        super().__init__()
        if cc_geometry not in CC_GEOMETRIES:
            raise ValueError(f"unknown CC geometry: {cc_geometry}")
        if cc_rank_weight < 0:
            raise ValueError("CC rank weight must be non-negative")
        if cc_rank_margin <= 0:
            raise ValueError("CC rank margin must be positive")
        self.minimum_scale = minimum_scale
        self.cc_geometry = cc_geometry
        self.cc_rank_weight = cc_rank_weight
        self.cc_rank_margin = cc_rank_margin
        self.witness_priority = witness_priority
        self.cc_scale_raw = nn.Parameter(torch.tensor(0.5))
        self.rm_scale_raw = nn.Parameter(torch.tensor(0.5))

    def scale(self, metric: str) -> Tensor:
        raw = self.cc_scale_raw if metric == "cc" else self.rm_scale_raw
        return F.softplus(raw) + self.minimum_scale

    def cc_distance(self, cosine: Tensor) -> Tensor:
        """Convert cosine similarity to a non-negative CC distance coordinate."""
        if self.cc_geometry == "chord":
            return 1.0 - cosine
        # Keep nonlinear coordinates accurate under CUDA/MPS mixed precision.
        cosine = cosine.float() if cosine.dtype in (torch.float16, torch.bfloat16) else cosine
        if self.cc_geometry == "angular":
            # Subtract the clamping floor so identical vectors still map to zero.
            epsilon = torch.finfo(cosine.dtype).eps
            ceiling = 1.0 - epsilon
            return (
                torch.acos(cosine.clamp(-ceiling, ceiling)) - torch.acos(cosine.new_tensor(ceiling))
            ).clamp_min(0.0)
        affinity = ((1.0 + cosine) / 2.0).clamp_min(torch.finfo(cosine.dtype).eps)
        return -torch.log(affinity)

    def calibrate_cc_scale(
        self, reference_cosine: float = 0.4, reference_distance: float = 1.0
    ) -> None:
        """Initialize scale so a reference cosine represents the requested CC."""
        if not -1.0 < reference_cosine < 1.0:
            raise ValueError("reference cosine must lie strictly inside (-1, 1)")
        if reference_distance <= 0:
            raise ValueError("reference distance must be positive")
        with torch.no_grad():
            coordinate = self.cc_distance(self.cc_scale_raw.new_tensor(reference_cosine))
            desired_scale = reference_distance / coordinate
            shifted = desired_scale - self.minimum_scale
            if shifted <= 0:
                raise ValueError("reference requires a scale below minimum_scale")
            self.cc_scale_raw.copy_(torch.log(torch.expm1(shifted)))

    def cc_ranking_loss(self, predicted: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
        """Hinge loss for every batch pair with provably ordered CC intervals."""
        ordered = upper[:, None] < lower[None, :]
        if not ordered.any():
            return predicted.new_zeros(())
        separation = predicted[None, :] - predicted[:, None]
        return F.relu(self.cc_rank_margin - separation[ordered]).mean()

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
            coordinate = self.cc_distance(cosine) if metric == "cc" else 1.0 - cosine
            predicted = self.scale(metric) * coordinate
            lower = batch.lower[index]
            upper = batch.upper[index]
            lengths = batch.witness_semantic_length[index]
            weights = torch.ones_like(predicted)
            if metric == "cc" and self.witness_priority:
                eligible = upper <= 1
                weights = torch.where(eligible & (lengths > 5), 2.0, weights)
                weights = torch.where(eligible & (lengths > 10), 4.0, weights)
                weights = torch.where(eligible & (lengths > 20), 8.0, weights)
            violation = F.relu(lower - predicted).square() + F.relu(predicted - upper).square()
            metric_loss = (violation * weights).sum() / weights.sum().clamp_min(1.0)
            if metric == "cc" and self.cc_rank_weight:
                rank_loss = self.cc_ranking_loss(predicted, lower, upper)
                metric_loss = metric_loss + self.cc_rank_weight * rank_loss
                diagnostics["cc_rank_loss"] = rank_loss.detach()
            total = total + metric_loss
            diagnostics[f"{metric}_loss"] = metric_loss.detach()
            diagnostics[f"{metric}_mean_predicted"] = predicted.mean().detach()
            active += 1
        if active == 0:
            raise ValueError("metric batch contains neither cc nor rm rows")
        return total / active, diagnostics
