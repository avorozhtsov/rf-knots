"""Length-scalable full-braid encoder used by embedding calibration."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualWordBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.pointwise = nn.Conv1d(width, 2 * width, 1)
        self.output = nn.Conv1d(width, width, 1)

    def forward(self, state: Tensor, mask: Tensor) -> Tensor:
        update = self.norm(state).transpose(1, 2)
        update = self.depthwise(update)
        update = F.glu(self.pointwise(update), dim=1)
        update = self.output(update).transpose(1, 2)
        return (state + update) * mask.unsqueeze(-1)


class ScalableBraidEmbeddingEncoder(nn.Module):
    """Read an entire variable-length braid and emit three global embeddings."""

    def __init__(
        self,
        *,
        width: int = 96,
        depth: int = 4,
        embedding_dim: int = 64,
        maximum_generator: int = 1023,
        maximum_strands: int = 1024,
    ):
        super().__init__()
        if depth < 1 or width < 8 or embedding_dim < 2:
            raise ValueError("invalid encoder dimensions")
        self.maximum_generator = maximum_generator
        self.maximum_strands = maximum_strands
        self.tokens = nn.Embedding(2 * maximum_generator + 1, width, padding_idx=0)
        self.strands = nn.Embedding(maximum_strands + 1, width)
        self.position_projection = nn.Linear(8, width, bias=False)
        self.blocks = nn.ModuleList(ResidualWordBlock(width) for _ in range(depth))
        self.pool_projection = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(width, embedding_dim)
                for name in ("knot", "isotopy", "layout")
            }
        )

    def _token_ids(self, word: Tensor) -> Tensor:
        magnitude = word.abs()
        if torch.any(magnitude > self.maximum_generator):
            raise ValueError("braid generator exceeds configured vocabulary")
        return torch.where(word > 0, 2 * magnitude - 1, 2 * magnitude)

    @staticmethod
    def _position_features(length: int, device: torch.device) -> Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32)
        denominator = max(length, 1)
        angle = 2 * math.pi * position / denominator
        return torch.stack(
            [
                torch.sin(angle),
                torch.cos(angle),
                torch.sin(2 * angle),
                torch.cos(2 * angle),
                torch.sin(4 * angle),
                torch.cos(4 * angle),
                position / denominator,
                torch.ones_like(position),
            ],
            dim=-1,
        )

    def forward(self, word: Tensor, lengths: Tensor, strands: Tensor) -> dict[str, Tensor]:
        if word.ndim != 2 or lengths.ndim != 1 or strands.ndim != 1:
            raise ValueError("bad padded braid batch shape")
        if torch.any(strands > self.maximum_strands):
            raise ValueError("strand count exceeds configured embedding")
        mask = torch.arange(word.shape[1], device=word.device)[None, :] < lengths[:, None]
        state = self.tokens(self._token_ids(word))
        state = state + self.position_projection(
            self._position_features(word.shape[1], word.device)
        )[None, :, :]
        state = state + self.strands(strands)[:, None, :]
        state = state * mask.unsqueeze(-1)
        for block in self.blocks:
            state = block(state, mask)
        divisor = lengths.clamp_min(1).unsqueeze(-1)
        mean = state.sum(1) / divisor
        maximum = state.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        first = state[:, 0]
        pooled = self.pool_projection(torch.cat((mean, maximum, first), dim=-1))
        return {name: F.normalize(head(pooled), dim=-1) for name, head in self.heads.items()}
