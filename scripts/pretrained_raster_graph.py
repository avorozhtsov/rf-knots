"""Unified size-independent network with actual pretrained local Q raster weights.

Only the supplied tile trunk is pretrained. Graph/global/policy layers are new.
This research module intentionally has no uncalibrated value output.
"""

import torch
from pretrained_raster_tiles import head_context
from torch import nn
from torch.nn import functional as F

from rf_knots.site_head import SiteHead
from rf_knots.strand_graph_encoder import (
    CircularMultiscaleContext,
    StrandMessageBlock,
    crossing_adjacency,
)

GLOBALS = ("knot", "layout", "isotopy")


class RasterGraphEncoder(nn.Module):
    def __init__(self, tile_encoder, *, width=64, embedding_dim=64, graph_rounds=2):
        super().__init__()
        if width < 2 or embedding_dim < 2 or graph_rounds < 1:
            raise ValueError("invalid graph integration dimensions")
        self.tiles = tile_encoder
        self.width, self.embedding_dim, self.graph_rounds = width, embedding_dim, graph_rounds
        self.project = nn.Sequential(
            nn.Linear(2 * tile_encoder.width + 8, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.message = StrandMessageBlock(width)
        self.context = CircularMultiscaleContext(width)
        self.pool = nn.Sequential(nn.Linear(3 * width, width), nn.LayerNorm(width), nn.GELU())
        self.heads = nn.ModuleDict({name: nn.Linear(width, embedding_dim) for name in GLOBALS})

    def encode_sites(self, word, lengths, strands):
        if (
            word.ndim != 2
            or word.shape[0] < 1
            or word.shape[1] < 1
            or lengths.shape != word.shape[:1]
            or strands.shape != lengths.shape
        ):
            raise ValueError("invalid padded braid shapes")
        if (
            word.is_floating_point()
            or lengths.is_floating_point()
            or strands.is_floating_point()
            or word.is_complex()
            or lengths.is_complex()
            or strands.is_complex()
            or word.dtype == torch.bool
            or lengths.dtype == torch.bool
            or strands.dtype == torch.bool
        ):
            raise ValueError("word/length/strand geometry must be integer")
        if (lengths < 0).any() or (lengths > word.shape[1]).any() or (strands < 1).any():
            raise ValueError("invalid active lengths/strands")
        mask = torch.arange(word.shape[1], device=word.device)[None, :] < lengths[:, None]
        clean = word.masked_fill(~mask, 0)
        encoded = []
        for row, length, n in zip(clean, lengths.tolist(), strands.tolist(), strict=True):
            active = row[:length]
            local = self.tiles(active, n)
            geometry = head_context(active, n, 0, rows=self.tiles.rows)[:, 8:]
            projected = self.project(torch.cat((local, geometry), dim=1))
            encoded.append(F.pad(projected, (0, 0, 0, word.shape[1] - length)))
        state = torch.stack(encoded) * mask.unsqueeze(-1)
        adjacency = crossing_adjacency(clean, lengths)
        for _ in range(self.graph_rounds):
            state = self.message(state, adjacency, mask)
        state = self.context(state, mask, lengths, -1)
        mean = state.sum(1) / lengths.clamp_min(1).unsqueeze(-1)
        maximum = state.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
        minimum = state.masked_fill(~mask.unsqueeze(-1), torch.inf).amin(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
        pooled = self.pool(torch.cat((mean, maximum, minimum), dim=-1))
        globals_ = {name: F.normalize(head(pooled), dim=-1) for name, head in self.heads.items()}
        return {
            "tokens": state,
            "mask": mask,
            **globals_,
            "word": clean.clone(),
            "lengths": lengths.clone(),
            "strands": strands.clone(),
        }

    def forward(self, word, lengths, strands):
        encoded = self.encode_sites(word, lengths, strands)
        return {name: encoded[name] for name in GLOBALS}


class RasterGraphPolicy(nn.Module):
    def __init__(self, tile_encoder, *, width=64, embedding_dim=64, graph_rounds=2):
        super().__init__()
        self.encoder = RasterGraphEncoder(
            tile_encoder, width=width, embedding_dim=embedding_dim, graph_rounds=graph_rounds
        )
        self.budget_dim = 5
        self.position = nn.Sequential(nn.Linear(width + 8, width), nn.LayerNorm(width), nn.GELU())
        self.head = SiteHead(
            width=width, embedding_dim=embedding_dim, budget_dim=5, structural_generators=True
        )

    def score_encoded(self, encoded, sites, site_mask, budgets, *, heads=None):
        """Reuse an unchanged encoded word; no local/global backbone recomputation."""
        state = encoded["tokens"]
        batch, padded, _ = state.shape
        if heads is not None and (heads.shape != (batch,) or heads.is_floating_point()):
            raise ValueError("require one integer head position per batch row")
        if heads is None:
            phase = state.new_zeros((batch, padded, 8))
        else:
            phases = []
            for row, length, n, head in zip(
                encoded["word"],
                encoded["lengths"].tolist(),
                encoded["strands"].tolist(),
                heads.tolist(),
                strict=True,
            ):
                phase_row = head_context(row[:length], n, head)[:, :8]
                phases.append(F.pad(phase_row, (0, 0, 0, padded - length)))
            phase = torch.stack(phases)
        positioned = self.position(torch.cat((state, phase), dim=-1)) * encoded["mask"].unsqueeze(
            -1
        )
        logits = self.head(
            positioned,
            encoded["mask"],
            {n: encoded[n] for n in GLOBALS},
            sites,
            site_mask,
            budgets=budgets,
            strands=encoded["strands"],
            word=encoded["word"],
        )
        effective = self.head.effective_site_mask(encoded["mask"], sites, site_mask)
        return logits, effective

    def forward(self, word, lengths, strands, sites, site_mask, budgets, *, heads=None):
        encoded = self.encoder.encode_sites(word, lengths, strands)
        logits, effective = self.score_encoded(encoded, sites, site_mask, budgets, heads=heads)
        return {"logits": logits, "effective": effective, **{n: encoded[n] for n in GLOBALS}}
