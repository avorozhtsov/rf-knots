"""`K # m(K)`: the one family whose certified floor is loose *by construction*.

Note 22 §M1, "Where to look for both axes at once: the target's own family", is
the specification this module implements.  The argument it makes, restated so it
can be checked rather than trusted:

* a growth gap needs `c_shrink > c_full >= L`, so a **tight** certified floor
  (`c_shrink == L`) forecloses it with no search at all -- which is what settled
  1,070 of the previous run's 1,162 candidates and why every certified-hard seed
  accepted nothing;
* `K # m(K)` is **slice**, so `g_4 = 0` and `tau = s = 0` and every 4-genus
  bound vanishes *structurally*.  Only the Montesinos obstruction on
  `H_1(Sigma_2) = H_1(Sigma_2 K) (+) H_1(Sigma_2 mK)` fires, and it caps at 2;
* meanwhile `u` scales with `K`, up to `2 u(K)` by subadditivity.

So the certified interval is `[2, 2u(K)]` and its width grows without bound in
`K` while the floor stays pinned at 2.  §1.2 measured exactly one instance of
this (`7_1 # m(7_1)`, `L = 2` against `u <= 5`); `premise_rows` measures the
whole family, which is what turns an anecdote into a premise.

**The construction.**  For a braid word `w` on `n` strands whose closure is `K`,
the mirror is the word with every letter negated, and the connected sum is the
concatenation with the second factor shifted onto strands `n .. 2n-1`:

```text
K # m(K) = closure( w  ++  shift(negate(w), n-1) )   on  2n - 1  strands
```

For `K = T(2, 2m+1) = sigma_1^(2m+1)` that is `sigma_1^q sigma_2^-q` on three
strands, `q = 2m+1` -- and §1.2 records that `connected_summands` splits exactly
this shape unaided, which is both the check that the construction is right and
the §1.4 hazard that every emitted instance has to be measured against.

**The program is explicit, and verified rather than argued.**  `u <= 2 u(K)` is
subadditivity, but subadditivity is not a *witness*; §4.1b's rule is that only a
replayable program may be composed.  `kmk_torus_seed` therefore carries the
summand-wise program written out -- `m` times "flip the first letter, cancel it
against the next" on each block, then two destabilisations -- and
`SeedRepresentation.verify_program` replays it to the empty one-braid before the
seed is allowed to exist.  That program is also the §1.4 baseline the corpus has
to beat, so the same object is the certificate and the thing to be beaten.

**Screening (§4.3), and what it does not guarantee.**  Seeds of this shape sit
next to the exam: `7_1 # m(7_1)` *is* the target.  `screen` runs three tests on
every emitted word -- the stored-representation screen, the `(det, |sigma|)`
fingerprint, and a summand-name screen -- and the middle one is the only one
that actually fires on the target.  Measured, and worth its own line: the
shipped `Quarantine.screen_name` **cannot** catch `7_1 # m(7_1)`, because
`invariants` reports its summands as the ambiguous `'7_1 or 12n_749'` (§1.2's
identification trap) and never as `m(7_1)`, so the exact-set test
`{'7_1','m(7_1)'} <= {'7_1 or 12n_749'}` is false.  `summand_alias_hit` repairs
that by matching the quarantined names *inside* the ambiguous labels.  None of
the three is closure under the `k`-ball, which §4.1b says outright is not
computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from m1_target import LEX, certified_change_floor, shortest_completion

from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE, ActionSpec
from rf_knots.interleaved_generator import (
    Quarantine,
    SeedRepresentation,
    Word,
)
from rf_knots.lower_bounds import computed_claims, strongest

SLICE_REASON = (
    "K # m(K) bounds a slice disc (the standard one for a knot summed with its "
    "reverse mirror), so g_4 = 0 and tau = s = 0 and every 4-genus lower bound "
    "on u vanishes by construction; only the Montesinos obstruction on the "
    "double branched cover can fire, and it certifies at most 2"
)


# -- the construction ----------------------------------------------------------


def mirror(word: Word) -> Word:
    """`m(K)` as a braid word: every crossing switched."""
    return tuple(-int(x) for x in word)


def shift(word: Word, by: int) -> Word:
    """Move a braid word onto the strands `by` higher, preserving signs."""
    return tuple((abs(int(x)) + by) * (1 if int(x) > 0 else -1) for x in word)


def mirror_connected_sum(word: Word, strands: int) -> tuple[Word, int]:
    """`K # m(K)` on `2 strands - 1` strands, as a braid word.

    The two factors occupy disjoint generator ranges and share exactly one
    strand, which is the standard braid form of a connected sum: the closure of
    `beta_1 ++ shift(beta_2, n-1)` in `B_(n+m-1)` is `closure(beta_1) #
    closure(beta_2)`.
    """
    word = tuple(int(x) for x in word)
    return word + shift(mirror(word), strands - 1), 2 * strands - 1


def kmk_torus_seed(m: int, spec: ActionSpec | None = None) -> SeedRepresentation:
    """`T(2,2m+1) # m(T(2,2m+1))` as `sigma_1^q sigma_2^-q`, program included.

    `u(T(2,q)) = m` is the Milnor conjecture (Kronheimer-Mrowka) and it is what
    makes `u_upper = 2m` a *composed* claim rather than a catalogue one -- but
    the claim recorded here is the program, not the theorem: `verify_program`
    replays `2m` crossing changes to the empty one-braid or the seed is refused.

    The certified lower bound is `computed_claims` on the composite word itself,
    never a table row (§4.1b's second hazard, §1.2's trap).
    """
    if m < 1:
        raise ValueError("K # m(K) needs a non-trivial K")
    q = 2 * m + 1
    word, strands = mirror_connected_sum((1,) * q, 2)
    program: list[tuple[int, int, int, int]] = []
    #: Each round flips the leading letter of a block and cancels it against its
    #: neighbour, so the block shortens by two and the word never leaves the
    #: knot case.  The first block sits at position 0 throughout; once it is one
    #: letter long the second block starts at position 1.
    for _ in range(m):
        program.append((CROSSING_CHANGE, 0, 1, 1))
        program.append((REDUCE, 0, 1, 1))
    for _ in range(m):
        program.append((CROSSING_CHANGE, 1, 1, 1))
        program.append((REDUCE, 1, 1, 1))
    program.append((DESTABILIZE, 0, 1, 1))
    program.append((DESTABILIZE, 0, 1, 1))
    seed = SeedRepresentation(
        name=f"T(2,{q}) # m(T(2,{q}))",
        family="connected-sum-mirror",
        word=word,
        strands=strands,
        lower_claims=computed_claims(word, strands),
        program=tuple(program),
        u_upper=2 * m,
        u_upper_source=(
            "summand-wise: the T(2,q) program applied to each block in turn, "
            "replayed to the empty one-braid -- this is also §1.4's free "
            "baseline and the number an instance has to beat"
        ),
    )
    if spec is not None:
        if not seed.fits(spec):
            raise ValueError(f"{seed.name} does not fit {spec}")
        seed.verify_program(spec)
    return seed


# -- the premise ---------------------------------------------------------------


@dataclass(frozen=True)
class PremiseRow:
    """One `K`, and whether `K # m(K)` gives the loose floor the note promises."""

    knot: str
    u_k: int | None
    composite_word: Word
    composite_strands: int
    letters: int
    certified_floor: int
    claims: tuple[dict, ...]
    u_upper: int | None
    summands: tuple[str, ...]
    determinant: int
    signature: int | None

    @property
    def width(self) -> int | None:
        """`2u(K) - L`.  Zero means the floor is **tight** and no gap can exist."""
        if self.u_upper is None:
            return None
        return self.u_upper - self.certified_floor

    @property
    def loose(self) -> bool:
        return bool(self.width)

    def to_dict(self) -> dict:
        return {
            "knot": self.knot,
            "u_k": self.u_k,
            "composite_word": list(self.composite_word),
            "composite_strands": self.composite_strands,
            "letters": self.letters,
            "certified_floor": self.certified_floor,
            "claims": list(self.claims),
            "u_upper": self.u_upper,
            "interval": [self.certified_floor, self.u_upper],
            "width": self.width,
            "loose": self.loose,
            "summands": list(self.summands),
            "determinant": self.determinant,
            "signature": self.signature,
        }


def premise_row(
    name: str, word: Word, strands: int, u_k: int | None
) -> PremiseRow:
    """Measure `L` and the interval on `K # m(K)`, computing nothing from a table.

    `u_upper` is `2 u(K)` when `u(K)` is known by a theorem, which is the
    subadditivity bound §1.4 warns is free.  The point of the row is the
    *difference*: `L` is measured on the composite and it is what a gap has to
    clear.
    """
    from rf_knots.invariants import invariants

    composite, n = mirror_connected_sum(word, strands)
    claims = computed_claims(composite, n)
    inv = invariants(composite, n)
    return PremiseRow(
        knot=name,
        u_k=u_k,
        composite_word=composite,
        composite_strands=n,
        letters=len(composite),
        certified_floor=int(strongest(claims) or 0),
        claims=tuple(claim.to_dict() for claim in claims),
        u_upper=None if u_k is None else 2 * u_k,
        summands=tuple(inv.summands),
        determinant=int(inv.determinant),
        signature=None if inv.signature is None else int(inv.signature),
    )


# -- §4.3 screening ------------------------------------------------------------

#: The quarantined *knot* names, stripped of the `m(...)` mirror wrapper, so
#: that an ambiguous summand label can be matched against them.  `invariants`
#: reports `7_1 # m(7_1)`'s summands as `('7_1 or 12n_749',) * 2`: the mirror is
#: not marked and the identification is not unique, so the shipped exact-set
#: test cannot fire.  Matching the bare names *inside* each label is strictly
#: more conservative -- it rejects `12n_749 # 12n_749` too.
QUARANTINED_SUMMAND_NAMES: frozenset[str] = frozenset({"7_1"})


def summand_alias_hit(summands: Sequence[str]) -> bool:
    """True when every summand label admits a quarantined name as a reading."""
    labels = [str(x) for x in summands]
    if len(labels) < 2:
        return False
    return all(
        any(
            name == alternative.strip().removeprefix("m(").removesuffix(")")
            for alternative in label.split(" or ")
            for name in QUARANTINED_SUMMAND_NAMES
        )
        for label in labels
    )


@dataclass(frozen=True)
class Screen:
    """The verdict of §4.3 on one emitted word, and what produced it."""

    reason: str | None
    determinant: int
    signature: int | None
    name: str | None
    summands: tuple[str, ...]

    @property
    def rejected(self) -> bool:
        return self.reason is not None


def screen(quarantine: Quarantine, word: Word, strands: int) -> Screen:
    """Three screens, cheapest first, and the fingerprint is the one that fires.

    Ordering matters only for cost: the stored-representation screen is a set
    lookup, the other two need `invariants` (~50 ms).  Nothing here is closure
    under the `k`-ball; see the module docstring.
    """
    from rf_knots.invariants import invariants

    stored = quarantine.screen_word(word, strands)
    if stored is not None:
        return Screen(stored, 0, None, None, ())
    inv = invariants(tuple(word), int(strands))
    summands = tuple(inv.summands)
    reason = quarantine.screen_fingerprint(inv.determinant, inv.signature or 0)
    if reason is None:
        reason = quarantine.screen_name(inv.name, summands)
    if reason is None and summand_alias_hit(summands):
        reason = "quarantined-composite-alias"
    return Screen(
        reason,
        int(inv.determinant),
        None if inv.signature is None else int(inv.signature),
        inv.name,
        summands,
    )


# -- §1.4: the summand-wise shortcut -------------------------------------------


@dataclass(frozen=True)
class SummandWise:
    """What splitting the instance into prime summands buys, measured.

    §1.4: `u(A # B) <= u(A) + u(B)` is free and `connected_summands` finds it
    trivially, so an instance whose optimum *is* the two summands solved
    independently teaches nothing.  `total_upper` is the sum of per-summand
    lexicographic optima -- each one an upper bound on that summand's `u`, so
    the sum is an upper bound on the composite's -- and `beats` says whether the
    instance's own optimum is strictly below it.

    `needs_growth` is the sharper question: if every summand's growth-free
    optimum already equals its full one, then the summand-wise route needs no
    growth at all, and whatever growth the composite instance requires is an
    artefact of *not splitting* rather than a property of the knot.
    """

    pieces: tuple[Word, ...] | None
    prime: bool
    per_summand_upper: tuple[int | None, ...]
    per_summand_shrink: tuple[int | None, ...]
    total_upper: int | None
    total_shrink: int | None
    needs_growth: bool | None
    determined: bool

    def to_dict(self) -> dict:
        return {
            "pieces": None if self.pieces is None else [list(p) for p in self.pieces],
            "prime": self.prime,
            "per_summand_upper": list(self.per_summand_upper),
            "per_summand_shrink": list(self.per_summand_shrink),
            "total_upper": self.total_upper,
            "total_shrink": self.total_shrink,
            "summandwise_needs_growth": self.needs_growth,
            "determined": self.determined,
        }

    def beats(self, c_full: int | None) -> bool | None:
        if c_full is None or self.total_upper is None:
            return None
        return c_full < self.total_upper


def summandwise(
    word: Word,
    strands: int,
    change_cap: int = 8,
    growth: int = 2,
    node_cap: int = 40_000,
) -> SummandWise:
    """Split the instance and solve each piece on its own, with the shipped search.

    Each piece is searched twice, once growth-free and once with the same growth
    cap the composite gets, so the pair answers both §1.4 questions at once.  A
    piece whose search caps leaves the total `None` and the row `determined =
    False`: an undetermined shortcut is not evidence either way and is counted
    in its own column, exactly as the gap verdicts are.
    """
    from rf_knots.invariants import connected_summands

    pieces = connected_summands(tuple(word), int(strands))
    if pieces is None:
        return SummandWise(None, False, (), (), None, None, None, False)
    if len(pieces) < 2:
        return SummandWise(pieces, True, (), (), None, None, None, True)
    uppers: list[int | None] = []
    shrinks: list[int | None] = []
    for piece in pieces:
        n = max((abs(int(x)) for x in piece), default=0) + 1
        spec = ActionSpec(max_len=len(piece) + growth, max_strands=max(n + 1, 2))
        floor = certified_change_floor(piece, n)
        full = shortest_completion(
            spec, tuple(piece), n, change_cap, growth=growth,
            node_cap=node_cap, objective=LEX, change_floor=floor,
        )
        import m1_growth

        shrink = m1_growth.shrink_optimum(
            spec, tuple(piece), n, change_cap=change_cap,
            node_cap=node_cap, change_floor=floor,
        )
        uppers.append(None if full.moves is None else int(full.crossing_changes))
        shrinks.append(
            None if shrink.moves is None else int(shrink.crossing_changes)
        )
    determined = all(x is not None for x in uppers)
    total_upper = sum(uppers) if determined else None  # type: ignore[arg-type]
    shrink_determined = all(x is not None for x in shrinks)
    total_shrink = sum(shrinks) if shrink_determined else None  # type: ignore[arg-type]
    needs_growth = (
        None
        if not (determined and shrink_determined)
        else any(u < s for u, s in zip(uppers, shrinks))  # type: ignore[operator]
    )
    return SummandWise(
        pieces, False, tuple(uppers), tuple(shrinks), total_upper, total_shrink,
        needs_growth, determined,
    )


__all__ = [
    "QUARANTINED_SUMMAND_NAMES",
    "SLICE_REASON",
    "PremiseRow",
    "Screen",
    "SummandWise",
    "kmk_torus_seed",
    "mirror",
    "mirror_connected_sum",
    "premise_row",
    "screen",
    "shift",
    "summand_alias_hit",
    "summandwise",
]
