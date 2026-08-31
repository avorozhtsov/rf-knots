"""M1's per-site policy head: one score per legal site, weights shared by all sites.

`research/22-scalable-rl-million-representations-improved.md` §M1 deletes the flat
`Linear(d -> 3L + 2L(N-1) + 4 + L)` output layer -- 2.46M parameters welded to
`(max_len, max_strands)` at TIER2 -- and replaces it with

```text
logit(site s, kind t) = <e_t, W h_s> + b_t      softmax over the legal sites only
```

`h_s` is the encoder's token at the site's word position, `e_t` a learned
embedding of the action kind, and `W`, `b`, `e` are shared by every site.  Nothing
here is sized by `max_len` or `max_strands`: the output width is the number of
sites the caller passes in, and `state_dict()` from a TIER1 run loads unchanged
into a run at any other size.  `test_site_head.py` asserts both.

Three things §M1 asks for that the bare formula does not give, and how each is
paid for:

* **`INSERT` is a `(position, generator, sign)` triple, not a position.**  The
  encoder is word-only, so it emits one token per word position and knows nothing
  about the strand a candidate insertion would live on, nor which of the two
  orientations is meant.  Both enter as a **FiLM modulation of `h_s`** --
  `h_s -> (1 + a_g) * h_s + b_g` -- which makes the score a *diagonal bilinear*
  form in (generator, token) rather than a per-generator bias, so "insert
  sigma_3 *here*" is expressible.  §6.4 governs how `g` is featurised: the strand
  axis is an interval with two real boundaries, not a circle, and strand distance
  is absolute, so `g` is represented by clipped distances to strand 1 and to
  strand `n-1` (learned embeddings of a clipped bucket, exact rather than a
  frequency ladder) alongside the insertion's sign.  It is never normalised by the
  strand count and never made cyclic.

* **The global context and the remaining budgets condition every site.**  Also by
  FiLM, not by concatenation.  Concatenating `[h_s; g]` widens `W` and forces a
  per-site matvec through the widened matrix; FiLM computes one `(gamma, beta)`
  pair per *batch element* -- `O(B * width)`, paid once -- and leaves the per-site
  work a gather plus an elementwise multiply-add plus one dot product.  The
  identity `<e_t, W(gamma * h + beta)> = <(W^T e_t) * gamma, h> + <W^T e_t, beta>`
  is what keeps it a dot product: `u_t = W^T e_t` is precomputed for all nine
  kinds, so a site costs three lookups, one fused multiply-add and one dot
  product -- about `3 * width` multiplies against `width` for the bare scalar
  product, and both are the order §M1 prices as free.  Measured, scoring all
  10,598 legal successors of the target's long word takes 1.5 ms on CPU and
  1.9 ms on MPS, against ~2 ms for one batch-1 encoder forward over the same
  word; `artifacts/m1-head/report.md` has the full table and the caveat that the
  head does *not* amortise over a batch the way the encoder does.

* **The four positionless actions** (`DESTABILIZE`, `STABILIZE_POS`,
  `STABILIZE_NEG`, `PASS`) have no site.  Each gets a learned vector standing in
  for `h_s`, pushed through the same context FiLM and scored by the same `W`,
  `e_t`, `b_t` -- so they compete in the same softmax as the sited actions rather
  than in a separate head.

It is deliberately **not** an MLP over `[h_s; g; e_t]`.  §M1 prices that at ~100x
the per-site cost and says it must earn its place by the gate failing first.

Masking is exact, in both directions.  A site whose `site_mask` entry is `False`
gets `MASKED_LOGIT = -inf` through a `torch.where`, so it contributes exactly zero
to any downstream softmax and receives exactly zero gradient -- and, because the
`where` also blocks the gradient into the branch that computed it, it can perturb
no other logit and no parameter.  The same holds for the fields a kind does not
use.  `ActionSpec.decode` fills those with *placeholders that are themselves
valid values* -- a `REDUCE` site decodes to `(kind, position, 1, 1)`, and `1` is a
perfectly good generator index -- so the head keys "does this site have a
generator" off the site's **kind** and never off the column's value.  The
positionless kinds read a bank row that is not a word position at all, and the
generator-less kinds are routed to the table's "no generator" row before any
lookup, so a placeholder cannot enter the arithmetic even in principle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rf_knots.actions import (
    DESTABILIZE,
    INSERT,
    NUM_KINDS,
    PASS,
    STABILIZE_NEG,
    STABILIZE_POS,
    ActionSpec,
)

__all__ = [
    "GENERATOR_KINDS",
    "GLOBAL_NAMES",
    "MASKED_LOGIT",
    "POSITIONLESS_KINDS",
    "SITE_FIELDS",
    "SiteHead",
    "masked_log_policy",
    "set_valued_loss",
    "sites_from_actions",
]

#: What `forward` writes at a site the caller masked out.  `-inf` rather than a
#: large negative number so that the mask enforces itself in any downstream
#: `softmax`, `log_softmax` or `topk` the caller reaches for.
MASKED_LOGIT = float("-inf")

#: Column order of the `sites` tensor -- exactly `ActionSpec.encode`'s arguments
#: and exactly `ActionSpec.decode`'s return, so a row round-trips through both.
SITE_FIELDS = ("kind", "position", "generator", "sign")

#: Kinds with no site at all.  `decode` returns `(kind, 0, 1, 1)` for them: the
#: last three columns are placeholders and the head must not read any of them.
POSITIONLESS_KINDS: tuple[int, ...] = (DESTABILIZE, STABILIZE_POS, STABILIZE_NEG, PASS)

#: Kinds whose site is a `(position, generator, sign)` triple rather than a bare
#: position.  `INSERT(p, g, s)` is `2L(N-1)` of the action space -- 23,552 of the
#: 25,604 slots at `(512, 24)` -- and `s` is what separates `sigma_g sigma_g^-1`
#: from `sigma_g^-1 sigma_g` at the same place, so it is a column of its own.
#: Every other kind carries `generator = 1, sign = 1` as placeholders, and the
#: head overwrites both before anything reads them.
GENERATOR_KINDS: tuple[int, ...] = (INSERT,)

#: The encoder's three global embeddings, in the order they are concatenated.
GLOBAL_NAMES: tuple[str, ...] = ("knot", "isotopy", "layout")


def _per_state_lookup(table: Tensor, index: Tensor) -> Tensor:
    """`table[b, index[b, s]]` for a per-state table: `[B, R, ...]`, `[B, S]` -> `[B, S, ...]`.

    Written as `F.embedding` over the flattened table rather than as
    `table.gather(1, index.expand(...))`.  The forward is the same lookup; the
    backward is not.  `gather`'s backward is a `scatter_add` of `S` rows into `R`
    of them, and every site of a kind lands on the same row, so at the target's
    10,598 sites with `R = 9` kinds it is 10,598 atomic accumulations into nine
    slots.  Measured on MPS that single call costs 19.6 ms where `F.embedding`'s
    dense backward costs 1.3 ms, and it is what made the whole head's backward
    235 ms against the forward's 6 ms.  `artifacts/m1-head/report.md` §6.4 records
    the measurement.
    """
    batch, rows = table.shape[0], table.shape[1]
    offsets = torch.arange(batch, device=index.device).unsqueeze(1) * rows
    return F.embedding(index + offsets, table.reshape(batch * rows, *table.shape[2:]))


class SiteHead(nn.Module):
    """Score every legal site with one shared set of weights.

    Args:
        width: the encoder's token width `W`; `tokens` arrive as `[B, L, W]`.
        embedding_dim: the encoder's global embedding width `D`, and the width of
            the space the kind embeddings `e_t` live in.
        num_kinds: how many action kinds exist (9, from `actions.KIND_NAMES`).
        budget_dim: how many remaining-budget channels `forward` will be given.
            Two, by default: remaining moves and remaining crossing changes, the
            two fields of `aplus.Budget`.
        boundary_clip: how far from each end of the strand interval the clipped
            bucket keeps resolving generators.  §6.4: the braid relation acts at
            `|g - h| = 1` and far commutation at `|g - h| >= 2`, so the structure
            that matters is near the ends and a clip represents it exactly.
        film_init_scale: standard deviation of the FiLM output layers.  Small, so
            the head starts close to the plain scalar product `<e_t, W h_s> + b_t`
            and the modulations grow into it.

    Shapes, all of which are the contract in §M1 and none of which mention
    `max_len` or `max_strands` as a *parameter* dimension::

        tokens     [B, L, W]  float   per word-position, zeroed on padding
        mask       [B, L]     bool    which word positions are real
        globals_   {name: [B, D]}     the encoder's knot/isotopy/layout heads
        sites      [B, S, 4]  long    (kind, position, generator, sign), which is
                                       exactly `ActionSpec.decode`'s return
        site_mask  [B, S]     bool    which candidates are legal
        -> logits  [B, S]     float   `-inf` wherever `site_mask` is False
    """

    def __init__(
        self,
        width: int = 96,
        embedding_dim: int = 64,
        num_kinds: int = NUM_KINDS,
        *,
        budget_dim: int = 2,
        boundary_clip: int = 7,
        context_width: int | None = None,
        film_init_scale: float = 0.02,
        normalize_tokens: bool = True,
        global_names: Sequence[str] = GLOBAL_NAMES,
        positionless_kinds: Sequence[int] = POSITIONLESS_KINDS,
        generator_kinds: Sequence[int] = GENERATOR_KINDS,
    ) -> None:
        super().__init__()
        if width < 2 or embedding_dim < 2 or num_kinds < 1:
            raise ValueError("invalid site-head dimensions")
        if budget_dim < 0 or boundary_clip < 1:
            raise ValueError("invalid site-head conditioning dimensions")
        self.width = int(width)
        self.embedding_dim = int(embedding_dim)
        self.num_kinds = int(num_kinds)
        self.budget_dim = int(budget_dim)
        self.boundary_clip = int(boundary_clip)
        self.global_names = tuple(global_names)

        # -- the scalar product itself ---------------------------------------
        self.token_norm = nn.LayerNorm(width) if normalize_tokens else nn.Identity()
        self.projection = nn.Linear(width, embedding_dim, bias=False)  # W
        self.kind_embedding = nn.Embedding(num_kinds, embedding_dim)  # e_t
        self.kind_bias = nn.Parameter(torch.zeros(num_kinds))  # b_t
        # One learned stand-in for `h_s` per kind.  Only the positionless kinds
        # read theirs; keeping the table full-width makes the lookup branch-free.
        self.positionless_vector = nn.Embedding(num_kinds, width)

        # -- FiLM from the global context and the remaining budgets -----------
        self.context_dim = len(self.global_names) * embedding_dim + 3 * budget_dim + 1
        hidden = int(context_width) if context_width else width
        self.context_film = nn.Sequential(
            nn.Linear(self.context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * width),
        )

        # -- FiLM from the generator, per §6.4's strand-axis geometry ---------
        # Row 0 of each table is "this site has no generator"; rows 1.. are the
        # clipped distance to strand 1 and to strand n-1.  Distances are absolute
        # -- never divided by the strand count -- and the clip saturates, so the
        # table size is fixed and two strands apart means the same at n=5 and
        # n=21.  The high table's row 1 doubles as "strand count not supplied".
        rows = boundary_clip + 2
        self.generator_low = nn.Embedding(rows, 2 * width)
        self.generator_high = nn.Embedding(rows + 1, 2 * width)
        self.generator_sign = nn.Embedding(3, 2 * width)

        positionless = torch.zeros(num_kinds, dtype=torch.bool)
        for kind in positionless_kinds:
            if 0 <= int(kind) < num_kinds:
                positionless[int(kind)] = True
        uses_generator = torch.zeros(num_kinds, dtype=torch.bool)
        for kind in generator_kinds:
            if 0 <= int(kind) < num_kinds:
                uses_generator[int(kind)] = True
        if bool((positionless & uses_generator).any()):
            raise ValueError("a kind cannot be both positionless and generator-bearing")
        self.register_buffer("positionless_flag", positionless, persistent=False)
        self.register_buffer("generator_flag", uses_generator, persistent=False)

        self._reset_parameters(film_init_scale)

    def _reset_parameters(self, film_init_scale: float) -> None:
        nn.init.normal_(self.projection.weight, std=self.width**-0.5)
        nn.init.normal_(self.kind_embedding.weight, std=1.0)
        nn.init.zeros_(self.kind_bias)
        nn.init.normal_(self.positionless_vector.weight, std=1.0)
        for table in (self.generator_low, self.generator_high, self.generator_sign):
            nn.init.normal_(table.weight, std=film_init_scale)
        last = self.context_film[-1]
        nn.init.normal_(last.weight, std=film_init_scale)
        nn.init.zeros_(last.bias)

    # -- the parts of the score that do not depend on the site ---------------

    def kind_directions(self) -> Tensor:
        """`u_t = W^T e_t` for every kind: `[num_kinds, width]`.

        This is why the per-site cost is a dot product rather than a matvec.
        `<e_t, W h> = <W^T e_t, h>`, and there are only `num_kinds` distinct
        `e_t`, so the projection is applied `num_kinds` times per forward instead
        of once per site.
        """
        return self.kind_embedding.weight @ self.projection.weight

    def _generator_film_table(self) -> Tensor:
        """Every (low bucket, high bucket, sign) combination: `[R, 2 * width]`.

        The three embeddings are summed, and there are only `R = rows * (rows+1) * 3`
        distinct sums -- 270 at the default clip -- so the sum is materialised once
        per forward and each site pays a single gather instead of three.  The `1 +`
        of the FiLM scale is folded in here too, off the per-site path.
        """
        low = self.generator_low.weight[:, None, None, :]
        high = self.generator_high.weight[None, :, None, :]
        sign = self.generator_sign.weight[None, None, :, :]
        table = (low + high + sign).reshape(-1, 2 * self.width)
        scale, shift = table.chunk(2, dim=-1)
        return torch.cat([1.0 + scale, shift], dim=-1)

    def _generator_rows(
        self,
        generator: Tensor,
        sign: Tensor,
        present: Tensor,
        strands: Tensor | None,
    ) -> Tensor:
        """Row index into `_generator_film_table` for each site.

        `present` is the kind's own answer to "does this site have a generator at
        all", never a value test on the column: the placeholder a positional kind
        carries is `generator = 1`, which is a perfectly valid generator index and
        so cannot be recognised by looking at it.
        """
        clip = self.boundary_clip
        rows_high = self.generator_high.num_embeddings
        magnitude = generator.abs()
        zero = torch.zeros_like(magnitude)

        low = torch.where(present, (magnitude - 1).clamp(0, clip) + 1, zero)

        if strands is None:
            high = torch.ones_like(low)  # the "strand count not supplied" row
        else:
            if strands.ndim != 1 or strands.shape[0] != generator.shape[0]:
                raise ValueError("strands must be [B]")
            top = strands.to(generator.dtype)[:, None] - 1
            high = (top - magnitude).clamp(0, clip) + 2
        high = torch.where(present, high, zero)

        # `+` is positive-first, matching `ActionSpec.encode`'s `s = 0 if sign > 0`.
        oriented = torch.where(sign > 0, zero + 2, zero)
        row_sign = torch.where(present, oriented, zero + 1)
        return (low * rows_high + high) * 3 + row_sign

    def _context_features(
        self,
        globals_: Mapping[str, Tensor] | Tensor,
        budgets: Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if isinstance(globals_, Mapping):
            missing = [name for name in self.global_names if name not in globals_]
            if missing:
                raise KeyError(f"globals_ is missing {missing}; got {sorted(globals_)}")
            parts = [globals_[name].to(dtype) for name in self.global_names]
        else:
            parts = [globals_.to(dtype)]
        for part in parts:
            if part.ndim != 2 or part.shape[0] != batch:
                raise ValueError("each global embedding must be [B, D]")
        context = torch.cat(parts, dim=-1)
        expected = len(self.global_names) * self.embedding_dim
        if context.shape[-1] != expected:
            raise ValueError(f"globals_ width {context.shape[-1]} != {expected}")

        if budgets is None:
            zeros = context.new_zeros(batch, self.budget_dim)
            flag = context.new_zeros(batch, 1)
            budget_features = [zeros, zeros, zeros + 1.0, flag]
        else:
            if budgets.ndim != 2 or budgets.shape != (batch, self.budget_dim):
                raise ValueError(f"budgets must be [{batch}, {self.budget_dim}]")
            value = budgets.to(dtype)
            budget_features = [
                value,
                torch.log1p(value.clamp_min(0.0)),
                (value <= 0).to(dtype),
                context.new_ones(batch, 1),
            ]
        return torch.cat([context, *budget_features], dim=-1).to(device)

    # -- the forward pass -----------------------------------------------------

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        globals_: Mapping[str, Tensor] | Tensor,
        sites: Tensor,
        site_mask: Tensor,
        *,
        budgets: Tensor | None = None,
        strands: Tensor | None = None,
    ) -> Tensor:
        """Score every site.  Returns `[B, S]`, `-inf` wherever `site_mask` is False.

        `budgets` (`[B, budget_dim]`, remaining moves and crossing changes) and
        `strands` (`[B]`, the braid's strand count) are optional extensions of the
        §M1 contract: both condition the score without widening the per-site work,
        and both are `None`-safe.  `strands` is used only for the distance-to-the-
        top-strand feature of §6.4; without it that feature reads "unknown".

        Everything about a masked site is discarded through `torch.where`, so a
        masked site influences no other logit and no parameter's gradient.  So are
        the fields a kind does not use, and so are sites whose position falls on
        word padding: those are dropped even if the caller called them legal,
        because the token there is padding rather than a site.
        """
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B, L, W]")
        batch, length, width = tokens.shape
        if width != self.width:
            raise ValueError(f"token width {width} != head width {self.width}")
        if length < 1:
            raise ValueError("tokens must carry at least one word position")
        if mask.shape != (batch, length):
            raise ValueError("mask must be [B, L]")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a BoolTensor")
        if sites.ndim != 3 or sites.shape[0] != batch or sites.shape[2] != len(SITE_FIELDS):
            raise ValueError("sites must be [B, S, 4] of (kind, position, generator, sign)")
        if site_mask.shape != sites.shape[:2]:
            raise ValueError("site_mask must be [B, S]")
        if site_mask.dtype != torch.bool:
            raise TypeError("site_mask must be a BoolTensor")
        if not torch.is_floating_point(tokens):
            raise TypeError("tokens must be a floating-point tensor")

        kind, position, generator, sign = (sites[..., i] for i in range(len(SITE_FIELDS)))

        # Padding in `sites` is free to be anything -- -1, 0, garbage.  Clamp
        # before any gather so out-of-range never raises, then drop the result.
        kind_safe = kind.clamp(0, self.num_kinds - 1)
        positionless = self.positionless_flag[kind_safe]
        uses_generator = self.generator_flag[kind_safe]

        in_range = (position >= 0) & (position < length)
        position_safe = position.clamp(0, length - 1)
        position_real = mask.gather(1, position_safe) & in_range
        effective = site_mask & (positionless | position_real)

        # One bank holds the word's tokens followed by the positionless kinds'
        # learned stand-ins, so a site is a single gather and a kind that has no
        # position reads a row that is not a word position at all.  Normalising
        # the bank rather than the gathered sites is the same arithmetic over
        # `L + num_kinds` vectors instead of over `S` of them, and `S` is 41x `L`
        # on the target's long word.
        stand_ins = self.token_norm(self.positionless_vector.weight)
        bank = torch.cat(
            [self.token_norm(tokens), stand_ins.unsqueeze(0).expand(batch, -1, -1)], dim=1
        )
        index = torch.where(positionless, length + kind_safe, position_safe)
        h = _per_state_lookup(bank, index)

        rows = self._generator_rows(generator, sign, uses_generator, strands)
        scale, shift = F.embedding(rows, self._generator_film_table()).chunk(2, dim=-1)

        # `<e_t, W(gamma_c * x + beta_c)> = <(W^T e_t) * gamma_c, x> + <W^T e_t, beta_c>`,
        # so the context FiLM is folded into the nine kind directions and into a
        # per-(state, kind) scalar.  Neither is a per-site tensor, which is the
        # whole reason §M1 asks for FiLM rather than concatenation.
        context = self._context_features(globals_, budgets, batch, tokens.device, tokens.dtype)
        gamma_c, beta_c = self.context_film(context).chunk(2, dim=-1)
        direction = self.kind_directions()
        modulated = direction.unsqueeze(0) * (1.0 + gamma_c).unsqueeze(1)
        offset = beta_c @ direction.t() + self.kind_bias

        site_direction = _per_state_lookup(modulated, kind_safe)
        logits = torch.linalg.vecdot(site_direction, torch.addcmul(shift, h, scale))
        logits = logits + _per_state_lookup(offset.unsqueeze(-1), kind_safe).squeeze(-1)
        return torch.where(effective, logits, logits.new_full((), MASKED_LOGIT))

    # -- supervision ----------------------------------------------------------

    def effective_site_mask(self, mask: Tensor, sites: Tensor, site_mask: Tensor) -> Tensor:
        """The mask `forward` actually used: `site_mask` minus sites on word padding.

        Exposed because the loss must be given the *same* mask the logits were
        computed under, and a caller who marked a padded position legal would
        otherwise supply a positive the softmax has already discarded.
        """
        length = mask.shape[1]
        kind_safe = sites[..., 0].clamp(0, self.num_kinds - 1)
        position = sites[..., 1]
        in_range = (position >= 0) & (position < length)
        real = mask.gather(1, position.clamp(0, max(length - 1, 0))) & in_range
        return site_mask & (self.positionless_flag[kind_safe] | real)

    def extra_repr(self) -> str:
        return (
            f"width={self.width}, embedding_dim={self.embedding_dim}, "
            f"num_kinds={self.num_kinds}, budget_dim={self.budget_dim}, "
            f"parameters={sum(p.numel() for p in self.parameters())}"
        )


def masked_log_policy(logits: Tensor, site_mask: Tensor) -> Tensor:
    """`log pi(a|s)` over the legal sites only.  Illegal entries come back `-inf`."""
    if site_mask.dtype != torch.bool:
        raise TypeError("site_mask must be a BoolTensor")
    floor = torch.finfo(logits.dtype).min
    legal = torch.where(site_mask, logits, logits.new_full((), floor))
    return torch.log_softmax(legal, dim=-1).masked_fill(~site_mask, MASKED_LOGIT)


def set_valued_loss(
    logits: Tensor,
    site_mask: Tensor,
    positive_mask: Tensor,
    *,
    reduction: str = "mean",
    validate: bool = True,
) -> Tensor:
    """`-log sum_{a in A_plus} pi(a|s)`, the loss of §5.3.

    `positive_mask` marks the sites that are **demonstrated good** -- `A_plus`,
    built by `rf_knots.aplus`, where every membership was exhibited and replayed.
    There is deliberately no way to say "bad".  Note 21 §5.1's rule is that a site
    outside `A_plus` is *unexplored, not negative*, and this signature is the
    enforcement: the only masks are "legal" and "known good", the labels must be
    boolean (so no caller can smuggle in a negative weight), and a positive that
    is not legal is an error rather than a silently dropped label.

    What that buys, exactly.  With `P = sum_{i in A_plus} pi_i`, the loss is
    `L = -log P = log sum_{legal} e^z - log sum_{A_plus} e^z`, so

        dL/dz_i = pi_i - q_i   for i in A_plus,  where q = softmax within A_plus
        dL/dz_j = pi_j         for j legal but unexplored
        dL/dz_k = 0            for k illegal or padded

    Three consequences, each of which `test_site_head.py` checks numerically:

    * **No pressure between two positives.**  `L` depends on `A_plus` only through
      the total `sum e^z`, so along any direction that moves mass between two
      positives while holding that total fixed -- `d = e^{-z_i} u_i - e^{-z_j} u_j`
      -- the directional derivative is `(1/S_all - 1/S_pos) - (1/S_all - 1/S_pos) = 0`
      *exactly*, and swapping two positives' logits leaves the loss and every
      other gradient unchanged.  The loss cannot pick a favourite among the sites
      that work.  It does not follow that training keeps them level: a gradient
      step is Euclidean in the logits, so it amplifies whatever the model already
      believed and the mass usually concentrates on one of them.  That preference
      comes from the encoder's features, never from this loss -- which is the
      point, and is the difference from cross-entropy on the single site the
      witness happened to take, whose gradient on the *other* known-good site is
      `+pi` and pushes it down.
    * **Nothing pushes an unexplored site down on its own account.**  Its gradient
      is `pi_i`, its own share of the normaliser and nothing else -- the same term
      every positive also carries.  The loss never references the unexplored set;
      it appears only inside `log sum_{legal}`.
    * **Illegal and padded sites are inert.**  Their logits are replaced before the
      `logsumexp`, so they contribute exactly zero mass and exactly zero gradient.

    Rows with no positives are skipped rather than fabricated: they contribute `0`
    to the sum and are excluded from the mean, with gradients that cancel exactly
    rather than being multiplied by zero (which would leave NaN in place).

    Args:
        logits: `[B, S]`, as returned by `SiteHead.forward`.  `-inf` at masked
            sites is expected and handled.
        site_mask: `[B, S]` bool, the legal sites.  Pass the mask the logits were
            computed under -- `SiteHead.effective_site_mask` if in doubt.
        positive_mask: `[B, S]` bool, the demonstrated-good sites.  Must be a
            subset of `site_mask`.
        reduction: `"mean"` (over rows that carry a positive), `"sum"`, or
            `"none"` (`[B]`, zero on rows with nothing to learn from).
        validate: check the masks agree.  Costs one host synchronisation; turn it
            off only in a hot loop that has already been checked.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [B, S]")
    if site_mask.shape != logits.shape or positive_mask.shape != logits.shape:
        raise ValueError("site_mask and positive_mask must match logits [B, S]")
    if site_mask.dtype != torch.bool or positive_mask.dtype != torch.bool:
        raise TypeError(
            "site_mask and positive_mask must be BoolTensors: a site is legal or not, "
            "and known-good or not yet known -- there is no weight and no negative"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")
    if validate and bool((positive_mask & ~site_mask).any()):
        raise ValueError(
            "a positive site is outside site_mask: A_plus must be a subset of the "
            "legal sites the softmax runs over"
        )

    floor = logits.new_full((), torch.finfo(logits.dtype).min)
    legal = torch.where(site_mask, logits, floor)
    usable = positive_mask.any(-1) & site_mask.any(-1)
    # On an unusable row the numerator is given the whole legal set, so the two
    # logsumexps are the identical tensor node and their gradients cancel exactly.
    selected = positive_mask | ~usable.unsqueeze(-1)

    denominator = torch.logsumexp(legal, dim=-1)
    numerator = torch.logsumexp(torch.where(selected, legal, floor), dim=-1)
    per_row = torch.where(usable, denominator - numerator, torch.zeros_like(denominator))

    if reduction == "none":
        return per_row
    total = per_row.sum()
    if reduction == "sum":
        return total
    return total / usable.sum().clamp_min(1).to(total.dtype)


#: §5.3's loss, reachable from a caller that holds only the module.
SiteHead.set_valued_loss = staticmethod(set_valued_loss)
SiteHead.masked_log_policy = staticmethod(masked_log_policy)


def sites_from_actions(
    actions: Sequence[int] | Tensor,
    spec: ActionSpec,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Turn flat action ids from `rf_knots.actions` into the head's `[S, 4]` sites.

    A row is exactly `ActionSpec.decode(action)`, so `spec.encode(*row)` returns
    the action it came from -- for every kind, including the placeholder columns
    the positionless and position-only kinds carry.  The head never sees an action
    id, that layout being what §M1 deletes, so this is the one place the two
    representations meet and the round trip is the check that they agree.

    The ordering of `actions` is the ordering of the returned sites, which is also
    the ordering `site_mask` and `positive_mask` are indexed by.
    """
    if isinstance(actions, Tensor):
        values = [int(a) for a in actions.reshape(-1).tolist()]
    else:
        values = [int(a) for a in actions]
    rows = [spec.decode(action) for action in values]
    if not rows:
        return torch.zeros((0, len(SITE_FIELDS)), dtype=torch.long, device=device)
    return torch.tensor(rows, dtype=torch.long, device=device)
