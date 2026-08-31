"""Length-scalable full-braid encoder used by embedding calibration.

Two positional geometries live here.

``positional_geometry="legacy"`` is the shipped encoder, unchanged to the bit:
absolute Fourier features of ``j / L`` projected into the token stream, and a
zero-padded depthwise convolution over the word.  Every existing checkpoint,
call site and test targets this path, so it is the default and its parameter
set, initialisation RNG draw order and ``state_dict`` keys are frozen.

``positional_geometry="necklace"`` implements note 22 §6.4, which observes that
the two axes of the raster have different geometry and that giving them the same
encoding is a correctness error rather than a tuning choice:

* **word position is a genuine circle.**  ``braid.py`` is explicit that the
  storage seam carries no meaning and that conjugation is exactly a rotation of
  it, so the trunk convolves *around* the word (each braid wrapping at its own
  length, never at the batch padding width) and the emitted tokens carry a
  rotary encoding at ``theta = 2 * pi / L``.  Normalising the frequencies by
  ``L`` is what lets one set of weights score a 20-letter and a 254-letter word.
* **strand index is not a circle** unless ``cyclic_band_generators`` is on.  It
  is an interval with two real boundaries, and what matters along it is a hard
  threshold -- far commutation is legal exactly at ``|g - h| >= 2`` and the
  braid relation acts at ``|g - h| = 1``.  A clipped relative bucket represents
  that exactly where a frequency ladder only approximates it, so the strand axis
  gets buckets of ``g - h`` over the convolution's own window plus absolute
  (never strand-count-normalised) distances to strand ``1`` and strand ``n-1``.
* **with ``cyclic_band_generators`` the diagram really is a torus.**  The seam
  generator ``b_n`` joins strand ``n`` to strand ``1``, the symmetry is
  ``U(1) x U(1)``, and the strand axis switches to the second half of a
  factorised two-axis rotary at ``theta = 2 * pi / n``.  The flag switches the
  geometry; neither branch is hard-coded.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_GEOMETRIES = ("legacy", "necklace")


class ResidualWordBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.pointwise = nn.Conv1d(width, 2 * width, 1)
        self.output = nn.Conv1d(width, width, 1)

    @property
    def radius(self) -> int:
        return int(self.depthwise.kernel_size[0]) // 2

    def forward(self, state: Tensor, mask: Tensor) -> Tensor:
        update = self.norm(state).transpose(1, 2)
        update = self.depthwise(update)
        update = F.glu(self.pointwise(update), dim=1)
        update = self.output(update).transpose(1, 2)
        return (state + update) * mask.unsqueeze(-1)

    def forward_necklace(self, state: Tensor, mask: Tensor, halo: Tensor) -> Tensor:
        """Same weights, but the word convolves around its own closure.

        ``halo`` names, for every column of a ``radius``-extended word, the
        position of the row it should read: ``(j - radius) mod len_b``.  Feeding
        the ordinary depthwise convolution that extended tensor is what makes
        the block exactly ``Z_L``-equivariant, and -- because the wrap uses each
        braid's own length -- what makes the result independent of how far the
        batch happens to be padded.  The two ``1x1`` convolutions are evaluated
        as the linear maps they are, on the ``[B, L, W]`` layout the encoder
        already uses; that is where M0.4's dispatch cliff lived.
        """
        update = self.norm(state)
        batch, length, width = update.shape
        base = (torch.arange(batch, device=state.device) * length).view(batch, 1)
        extended = update.reshape(batch * length, width).index_select(
            0, (halo + base).reshape(-1)
        )
        update = F.conv1d(
            extended.view(batch, -1, width).transpose(1, 2),
            self.depthwise.weight,
            self.depthwise.bias,
            groups=width,
        ).transpose(1, 2)
        update = F.linear(update, self.pointwise.weight.squeeze(-1), self.pointwise.bias)
        update = F.glu(update, dim=-1)
        update = F.linear(update, self.output.weight.squeeze(-1), self.output.bias)
        return (state + update) * mask.unsqueeze(-1)


class ScalableBraidEmbeddingEncoder(nn.Module):
    """Read an entire variable-length braid and emit three global embeddings.

    ``forward`` keeps its exact signature and its three-key return.  Callers
    that also want the per-word-position tokens -- the per-site policy head of
    note 22 §M1 indexes them -- call :meth:`encode_sites`, which returns those
    same three globals alongside ``"tokens"`` and ``"mask"``.
    """

    def __init__(
        self,
        *,
        width: int = 96,
        depth: int = 4,
        embedding_dim: int = 64,
        maximum_generator: int = 1023,
        maximum_strands: int = 1024,
        positional_geometry: str = "legacy",
        cyclic_band_generators: bool = False,
        strand_bucket_radius: int = 4,
        strand_window: int = 2,
        strand_boundary_radius: int = 8,
    ):
        super().__init__()
        if depth < 1 or width < 8 or embedding_dim < 2:
            raise ValueError("invalid encoder dimensions")
        if positional_geometry not in _GEOMETRIES:
            raise ValueError(f"unknown positional geometry {positional_geometry!r}")
        if cyclic_band_generators and positional_geometry != "necklace":
            raise ValueError(
                "cyclic_band_generators is a torus geometry and needs "
                "positional_geometry='necklace'"
            )
        self.maximum_generator = maximum_generator
        self.maximum_strands = maximum_strands
        self.positional_geometry = positional_geometry
        self.cyclic_band_generators = cyclic_band_generators
        self.strand_bucket_radius = int(strand_bucket_radius)
        self.strand_window = int(strand_window)
        self.strand_boundary_radius = int(strand_boundary_radius)
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
        # Everything below is additive and is built only under the necklace
        # geometry, so the legacy encoder keeps byte-identical initialisation
        # (no extra RNG draws) and a byte-identical state_dict key set.
        if positional_geometry == "necklace":
            self._build_necklace(width)

    def _build_necklace(self, width: int) -> None:
        if width % 2:
            raise ValueError("necklace geometry needs an even width for rotary pairs")
        if self.cyclic_band_generators and width % 4:
            raise ValueError("two-axis rotary needs a width divisible by four")
        if self.strand_bucket_radius < 1 or self.strand_boundary_radius < 1:
            raise ValueError("strand bucket and boundary radii must be positive")
        radius = self.blocks[0].radius
        if not 1 <= self.strand_window <= radius:
            raise ValueError("strand window must fit inside the convolution window")
        pairs = width // 2
        # Under the torus the group is U(1) x U(1) and the representation is
        # factorised: half the SO(2) blocks turn with the word, half with the
        # strand.  Off the torus every block turns with the word alone.
        word_pairs = pairs // 2 if self.cyclic_band_generators else pairs
        self.register_buffer(
            "word_harmonics", torch.arange(word_pairs, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "strand_harmonics",
            torch.arange(pairs - word_pairs, dtype=torch.long),
            persistent=False,
        )
        if not self.cyclic_band_generators:
            offsets = [d for d in range(-self.strand_window, self.strand_window + 1) if d]
            self.register_buffer(
                "strand_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
            )
            buckets = 2 * self.strand_bucket_radius + 1
            self.strand_relative = nn.Embedding(len(offsets) * buckets, width)
            self.strand_low = nn.Embedding(self.strand_boundary_radius + 1, width)
            self.strand_high = nn.Embedding(self.strand_boundary_radius + 1, width)

    # ------------------------------------------------------------------ token ids

    def _token_ids(self, word: Tensor) -> Tensor:
        magnitude = word.abs()
        if torch.any(magnitude > self.maximum_generator):
            raise ValueError("braid generator exceeds configured vocabulary")
        return torch.where(word > 0, 2 * magnitude - 1, 2 * magnitude)

    @staticmethod
    def _position_features(width: int, lengths: Tensor, device: torch.device) -> Tensor:
        """Positions normalized by each braid, never by unrelated batch padding."""
        position = torch.arange(width, device=device, dtype=torch.float32)[None, :]
        denominator = lengths.clamp_min(1).to(dtype=torch.float32)[:, None]
        phase = position / denominator
        angle = 2 * math.pi * phase
        return torch.stack(
            [
                torch.sin(angle),
                torch.cos(angle),
                torch.sin(2 * angle),
                torch.cos(2 * angle),
                torch.sin(4 * angle),
                torch.cos(4 * angle),
                phase,
                torch.ones_like(phase),
            ],
            dim=-1,
        )

    # -------------------------------------------------------------- necklace parts

    @staticmethod
    def _necklace_halo(padded: int, lengths: Tensor, radius: int) -> Tensor:
        """Read positions ``(j - radius) mod len_b`` of a radius-extended word.

        The modulus is each braid's own length, so a row padded to width 7 and
        the same row padded to width 254 read exactly the same letters.
        """
        column = torch.arange(
            -radius, padded + radius, device=lengths.device, dtype=torch.long
        )[None, :]
        return column % lengths.clamp_min(1).to(torch.long)[:, None]

    def _strand_features(self, word: Tensor, strands: Tensor, halo: Tensor) -> Tensor:
        """Clipped relative buckets of ``g - h`` plus absolute boundary distance.

        The bucket is exact where the move set has its threshold: ``|g - h| = 1``
        is the braid relation, ``|g - h| >= 2`` is free commutation.  Distance is
        never divided by the strand count -- two strands apart means the same
        thing at ``n = 5`` and at ``n = 21``.
        """
        generator = word.abs()
        batch, padded = generator.shape
        span = self.strand_offsets.numel()
        centre = (halo.shape[1] - padded) // 2
        columns = (
            centre
            + torch.arange(padded, device=word.device, dtype=torch.long)[:, None]
            + self.strand_offsets[None, :]
        )
        selected = halo.index_select(1, columns.reshape(-1))
        neighbour_generator = torch.gather(generator, 1, selected).reshape(
            batch, padded, span
        )
        radius = self.strand_bucket_radius
        delta = (generator[:, :, None] - neighbour_generator).clamp(-radius, radius)
        stride = 2 * radius + 1
        index = (
            delta
            + radius
            + torch.arange(span, device=word.device, dtype=torch.long)[None, None, :]
            * stride
        )
        # Accumulate offset by offset.  `embedding(index).sum(dim=2)` is the same
        # arithmetic but materialises a `[B, L, span, W]` tensor, which costs
        # 131 ms against 5 ms on MPS at (256, 21) with a batch of 256.
        features = self.strand_relative(index[..., 0])
        for offset in range(1, span):
            features = features + self.strand_relative(index[..., offset])
        boundary = self.strand_boundary_radius
        largest = (strands.to(torch.long) - 1).clamp_min(1)[:, None]
        low = (generator - 1).clamp(0, boundary)
        high = (largest - generator).clamp(0, boundary)
        return features + self.strand_low(low) + self.strand_high(high)

    def _rotary_angles(self, word: Tensor, lengths: Tensor, strands: Tensor) -> Tensor:
        """Angles for the rotary blocks, reduced in integer arithmetic.

        ``(m * j) mod L`` is taken before the division so that a position and
        its image under a cyclic shift land on the identical float angle, which
        is what makes the ``Z_L`` equivariance exact rather than approximate.
        """
        padded = word.shape[1]
        device = word.device
        dtype = self.tokens.weight.dtype
        position = torch.arange(padded, device=device, dtype=torch.long)[None, :, None]
        length = lengths.clamp_min(1).to(torch.long)[:, None, None]
        phase = (position * self.word_harmonics[None, None, :]) % length
        angle = phase.to(dtype) * (2 * math.pi) / length.to(dtype)
        if self.strand_harmonics.numel():
            generator = word.abs().to(torch.long)[:, :, None]
            count = strands.clamp_min(1).to(torch.long)[:, None, None]
            strand_phase = (generator * self.strand_harmonics[None, None, :]) % count
            strand_angle = strand_phase.to(dtype) * (2 * math.pi) / count.to(dtype)
            angle = torch.cat((angle, strand_angle), dim=-1)
        return angle

    @staticmethod
    def _apply_rotary(state: Tensor, angle: Tensor) -> Tensor:
        pairs = angle.shape[-1]
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        left, right = state[..., :pairs], state[..., pairs:]
        return torch.cat(
            (left * cosine - right * sine, left * sine + right * cosine), dim=-1
        )

    # ------------------------------------------------------------------- encoding

    def _encode(
        self, word: Tensor, lengths: Tensor, strands: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if word.ndim != 2 or lengths.ndim != 1 or strands.ndim != 1:
            raise ValueError("bad padded braid batch shape")
        if torch.any(strands > self.maximum_strands):
            raise ValueError("strand count exceeds configured embedding")
        mask = torch.arange(word.shape[1], device=word.device)[None, :] < lengths[:, None]
        if self.positional_geometry == "legacy":
            state = self.tokens(self._token_ids(word))
            state = state + self.position_projection(
                self._position_features(word.shape[1], lengths, word.device)
            )
            state = state + self.strands(strands)[:, None, :]
            state = state * mask.unsqueeze(-1)
            for block in self.blocks:
                state = block(state, mask)
            divisor = lengths.clamp_min(1).unsqueeze(-1)
            mean = state.sum(1) / divisor
            maximum = state.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
            maximum = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
            )
            first = state[:, 0]
            pooled = self.pool_projection(torch.cat((mean, maximum, first), dim=-1))
            sites = state
        else:
            halo = self._necklace_halo(word.shape[1], lengths, self.blocks[0].radius)
            state = self.tokens(self._token_ids(word))
            if not self.cyclic_band_generators:
                state = state + self._strand_features(word, strands, halo)
            state = state + self.strands(strands)[:, None, :]
            state = state * mask.unsqueeze(-1)
            for block in self.blocks:
                state = block.forward_necklace(state, mask, halo)
            divisor = lengths.clamp_min(1).unsqueeze(-1)
            mean = state.sum(1) / divisor
            maximum = state.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
            maximum = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
            )
            # ``first`` would reintroduce the storage seam the necklace geometry
            # exists to erase, so the third statistic is the masked minimum:
            # permutation invariant, hence the globals are exactly invariant
            # under conjugation-as-rotation, which is what §6.1 asks of `knot`.
            minimum = state.masked_fill(~mask.unsqueeze(-1), torch.inf).amin(1)
            minimum = torch.where(
                torch.isfinite(minimum), minimum, torch.zeros_like(minimum)
            )
            pooled = self.pool_projection(torch.cat((mean, maximum, minimum), dim=-1))
            sites = self._apply_rotary(
                state, self._rotary_angles(word, lengths, strands)
            ) * mask.unsqueeze(-1)
        globals_ = {
            name: F.normalize(head(pooled), dim=-1) for name, head in self.heads.items()
        }
        return sites, mask, globals_

    def forward(self, word: Tensor, lengths: Tensor, strands: Tensor) -> dict[str, Tensor]:
        return self._encode(word, lengths, strands)[2]

    def encode_sites(
        self, word: Tensor, lengths: Tensor, strands: Tensor
    ) -> dict[str, Tensor]:
        """``forward`` plus the per-word-position tokens the site head indexes.

        Returns ``"tokens"`` ``[B, L, W]`` zeroed on padding, ``"mask"``
        ``[B, L]``, and the ``"knot"`` / ``"isotopy"`` / ``"layout"`` globals
        ``[B, D]`` with the semantics ``forward`` already gives them.
        """
        sites, mask, globals_ = self._encode(word, lengths, strands)
        return {"tokens": sites, "mask": mask, **globals_}
