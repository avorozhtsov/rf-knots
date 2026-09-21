"""Size-independent Artin-braid encoder; an experimental, opt-in architecture.

Crossings are vertices. Each strand-position segment joins consecutive crossings
incident on that position, including the closure seam. No array is indexed by
the total number of strands: even inactive strands cost only scalar features.
These are diagram adjacency edges, not assertions of knot equivalence.

The graph trunk has shared weights across message-passing rounds. Increasing the
round count changes compute/receptive field, not checkpoint dimensions. Finite
depth and fixed-size global pooling remain capacity bottlenecks; accepting a
large braid is not evidence of solving it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder


def crossing_adjacency(word: Tensor, lengths: Tensor) -> Tensor:
    """Return previous-left/right, next-left/right crossing indices [B,L,4].

    Work/storage O(sum(lengths) + B*padded_length), independent of strand count.
    Topology is discrete CPU preprocessing (not differentiated). Each crossing
    touches lane abs(g) and abs(g)+1; the crossing vertex mixes both ports. A
    lane with a single incident crossing forms a self-edge. Empty/padded sites
    also use self-edges and are masked by the caller.
    """
    batch, padded = word.shape
    rows = word.detach().cpu().tolist()
    sizes = lengths.detach().cpu().tolist()
    neighbours = torch.arange(padded).view(1, padded, 1).expand(batch, padded, 4).clone()
    for b, (row, length) in enumerate(zip(rows, sizes, strict=True)):
        lanes: dict[int, list[tuple[int, int]]] = {}
        for j, signed_generator in enumerate(row[:length]):
            g = abs(signed_generator)
            for side in (0, 1):
                lanes.setdefault(g + side, []).append((j, side))
        for crossings in lanes.values():
            for i, (j, side) in enumerate(crossings):
                neighbours[b, j, side] = crossings[i - 1][0]
                neighbours[b, j, 2 + side] = crossings[(i + 1) % len(crossings)][0]
    return neighbours.to(device=word.device)


class StrandMessageBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.update = nn.Sequential(
            nn.Linear(5 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, state: Tensor, adjacency: Tensor, mask: Tensor) -> Tensor:
        batch, length, width = state.shape
        normalized = self.norm(state)
        offsets = torch.arange(batch, device=state.device).view(batch, 1, 1) * length
        messages = F.embedding(adjacency + offsets, normalized.reshape(batch * length, width))
        update = self.update(torch.cat((normalized, messages.flatten(2)), dim=-1))
        return (state + update) * mask.unsqueeze(-1)


class CircularMultiscaleContext(nn.Module):
    """Shared-weight circular dilations; O(L log L), no storage-seam anchor.

    Automatic depth is per example, NOT the batch's longest word. The effective
    neighbourhood grows exponentially; no claim of an exact knot invariant is
    made. This is a trunk block, not an additive output/value adapter.
    """

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Parameter(torch.empty(5, width))
        nn.init.normal_(self.depthwise, std=5**-0.5)
        self.update = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, state: Tensor, mask: Tensor, lengths: Tensor, rounds: int) -> Tensor:
        batch, padded, width = state.shape
        # Even when padded differs, extra automatic rounds are identities on a
        # short row. Parameters and arithmetic on its active rounds are shared.
        count = (padded - 1).bit_length() if rounds == -1 else rounds
        position = torch.arange(padded, device=state.device)[None, :, None]
        offsets = torch.arange(-2, 3, device=state.device)[None, None, :]
        base = torch.arange(batch, device=state.device)[:, None, None] * padded
        for level in range(count):
            dilation = 1 << level
            index = (position + dilation * offsets) % lengths.clamp_min(1)[:, None, None]
            normalized = self.norm(state).reshape(batch * padded, width)
            neighbourhood = F.embedding(index + base, normalized)
            update = self.update((neighbourhood * self.depthwise).sum(2))
            active = (lengths > dilation) if rounds == -1 else (lengths > 0)
            state = (state + update * active[:, None, None]) * mask.unsqueeze(-1)
        return state


class HierarchicalWindowContext(nn.Module):
    """Shared-weight relative windows at exponentially growing radii.

    This is an opt-in trunk block for long words.  A token at position ``p``
    receives the mean of circular windows centred at ``p`` with radii
    ``window_size * 2**level``.  The same projection is reused at every level;
    the level count therefore changes compute/receptive field, not checkpoint
    dimensions.  Indices are reduced modulo each row's true length, never the
    batch padding width, so a storage seam cannot become a learned landmark.

    It is deliberately local-to-global rather than a single pooled vector:
    policy sites keep their per-position state while each level supplies a
    progressively wider relative summary.  The implementation is O(L * levels)
    for a fixed number of sampled offsets per level, with no LxL attention.
    """

    def __init__(self, width: int, window_size: int = 8, levels: int = 3) -> None:
        super().__init__()
        if width < 2 or window_size < 1 or levels < 1:
            raise ValueError("invalid hierarchical window dimensions")
        self.width = int(width)
        self.window_size = int(window_size)
        self.levels = int(levels)
        self.norm = nn.LayerNorm(width)
        self.relative_projection = nn.Sequential(
            nn.Linear(width + 2, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))

    def forward(self, state: Tensor, mask: Tensor, lengths: Tensor) -> Tensor:
        batch, padded, width = state.shape
        if width != self.width:
            raise ValueError("state width does not match hierarchical window width")
        if mask.shape != (batch, padded) or lengths.shape != (batch,):
            raise ValueError("invalid hierarchical window shapes")
        if padded < 1:
            return state
        normalized = self.norm(state)
        positions = torch.arange(padded, device=state.device, dtype=torch.long)
        row = torch.arange(batch, device=state.device, dtype=torch.long)[:, None]
        base = row * padded
        chunks: list[Tensor] = []
        # A fixed odd stencil per level approximates the full window while
        # keeping the cost linear.  Relative phase is represented explicitly,
        # so the same local pattern at another word position is distinguishable
        # when the policy asks for it, without changing the global knot meaning.
        for level in range(self.levels):
            radius = self.window_size * (1 << level)
            stride = max(1, radius // 4)
            offsets = torch.arange(-radius, radius + 1, stride, device=state.device)
            indices = (positions[None, :, None] + offsets[None, None, :]) % lengths.clamp_min(1)[:, None, None]
            gathered = F.embedding(
                indices + base[:, :, None], normalized.reshape(batch * padded, width)
            )
            summary = gathered.mean(dim=2)
            # The offset extent is a relative fraction of the actual word; this
            # is not an absolute seam coordinate and remains finite for L=1.
            phase = offsets.to(normalized.dtype) / lengths.clamp_min(1).to(normalized.dtype)[:, None, None]
            phase = phase.expand(batch, padded, offsets.numel())
            phase_features = torch.stack(
                (phase.mean(dim=-1), phase.abs().mean(dim=-1)), dim=-1
            )
            chunks.append(self.relative_projection(torch.cat((summary, phase_features), dim=-1)))
        update = torch.stack(chunks, dim=0).mean(dim=0)
        return (state + self.output(update)) * mask.unsqueeze(-1)


class StrandGraphBraidEncoder(ScalableBraidEmbeddingEncoder):
    """One parameter set for any finite word length / Artin strand count.

    The tiny sign vocabulary has only padding/positive/negative. Unclipped
    logarithmic distances to boundaries and relative position replace lookup
    tables of generator IDs and strand counts. Local *integer* relative buckets
    retain exact adjacency/commutation thresholds. Float features need not be
    injective at astronomically large indices; topology is built from integers.
    """

    def __init__(
        self,
        *,
        width: int = 96,
        depth: int = 4,
        embedding_dim: int = 64,
        graph_rounds: int = 2,
        context_rounds: int = 0,
        window_size: int = 0,
        hierarchy_levels: int = 0,
        normalized_knot: bool = False,
    ):
        if graph_rounds < 0:
            raise ValueError("graph_rounds must be nonnegative")
        if context_rounds < -1:
            raise ValueError("context_rounds must be -1 (automatic), zero, or positive")
        if window_size < 0 or hierarchy_levels < 0:
            raise ValueError("window_size and hierarchy_levels must be nonnegative")
        if bool(window_size) != bool(hierarchy_levels):
            raise ValueError("window_size and hierarchy_levels must be enabled together")
        # Reuse the tested necklace blocks and pooling, without large dummy
        # vocabularies. Historical encoders/checkpoints are completely untouched.
        super().__init__(
            width=width,
            depth=depth,
            embedding_dim=embedding_dim,
            maximum_generator=1,
            maximum_strands=1,
            positional_geometry="necklace",
        )
        del self.strands
        del self.position_projection
        self.maximum_generator = None
        self.maximum_strands = None
        self.structural_projection = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.Linear(width, width)
        )
        self.strand_message = StrandMessageBlock(width)
        self.graph_rounds = graph_rounds
        self.context_rounds = context_rounds
        self.window_size = int(window_size)
        self.hierarchy_levels = int(hierarchy_levels)
        self.normalized_knot = normalized_knot
        if context_rounds:
            self.multiscale_context = CircularMultiscaleContext(width)
        if hierarchy_levels:
            self.window_context = HierarchicalWindowContext(
                width, window_size=window_size, levels=hierarchy_levels
            )

    def _encode(self, word: Tensor, lengths: Tensor, strands: Tensor):
        sites, mask, globals_ = self._encode_raw(word, lengths, strands)
        if self.normalized_knot:
            from rf_knots.embedding_normalization import fast_normalize_global_view

            views = [
                fast_normalize_global_view(tuple(row[:length]), n)
                for row, length, n in zip(
                    word.detach().cpu().tolist(),
                    lengths.detach().cpu().tolist(),
                    strands.detach().cpu().tolist(),
                    strict=True,
                )
            ]
            padded = max(1, max(len(row) for row, _ in views))
            normalized = word.new_tensor(
                [list(row) + [0] * (padded - len(row)) for row, _ in views]
            )
            sizes = lengths.new_tensor([len(row) for row, _ in views])
            counts = strands.new_tensor([n for _, n in views])
            # Only knot geometry is invariant under these moves. Isotopy-distance
            # and layout heads, site indices and masks keep the raw diagram.
            globals_["knot"] = self._encode_raw(normalized, sizes, counts)[2]["knot"]
        return sites, mask, globals_

    def _encode_raw(self, word: Tensor, lengths: Tensor, strands: Tensor):
        if (
            word.ndim != 2
            or word.shape[1] < 1
            or lengths.shape != word.shape[:1]
            or strands.shape != lengths.shape
        ):
            raise ValueError("bad padded braid batch shape")
        if torch.any(lengths < 0) or torch.any(lengths > word.shape[1]) or torch.any(strands < 1):
            raise ValueError("invalid lengths or strand counts")
        mask = torch.arange(word.shape[1], device=word.device)[None, :] < lengths[:, None]
        magnitude = word.abs()
        if torch.any(mask & ((magnitude == 0) | (magnitude >= strands[:, None]))):
            raise ValueError("active Artin generators must satisfy 1 <= abs(g) < strands")
        # Ignore padding contents, not merely the final padded outputs.
        word = word.masked_fill(~mask, 0)
        magnitude = word.abs()
        dtype = self.tokens.weight.dtype
        g = magnitude.to(dtype)
        n = strands.to(dtype)[:, None].expand_as(g)
        features = torch.stack(
            (
                torch.log1p(n),
                torch.log1p(g),
                torch.log1p((n - g).clamp_min(0)),
                g / (n - 1).clamp_min(1),
                torch.log1p(lengths.to(dtype))[:, None].expand_as(g),
            ),
            dim=-1,
        )
        ids = torch.where(word == 0, 0, torch.where(word > 0, 1, 2))
        halo = self._necklace_halo(word.shape[1], lengths, self.blocks[0].radius)
        state = (
            self.tokens(ids)
            + self.structural_projection(features)
            + self._strand_features(word, strands, halo)
        ) * mask.unsqueeze(-1)
        adjacency = crossing_adjacency(word, lengths) if self.graph_rounds else None
        for block in self.blocks:
            state = block.forward_necklace(state, mask, halo)
        if self.context_rounds:
            state = self.multiscale_context(state, mask, lengths, self.context_rounds)
        for _ in range(self.graph_rounds):
            state = self.strand_message(state, adjacency, mask)
        if self.hierarchy_levels:
            state = self.window_context(state, mask, lengths)
        mean = state.sum(1) / lengths.clamp_min(1).unsqueeze(-1)
        maximum = state.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
        minimum = state.masked_fill(~mask.unsqueeze(-1), torch.inf).amin(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
        pooled = self.pool_projection(torch.cat((mean, maximum, minimum), dim=-1))
        globals_ = {name: F.normalize(head(pooled), dim=-1) for name, head in self.heads.items()}
        # Local sites already carry their own topology and whole-word context
        # through SiteHead; no seam-dependent absolute phase enters their logits.
        return state, mask, globals_
