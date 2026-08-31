"""Hard positives built backwards from the answer, with the witness for free.

`research/22-scalable-rl-million-representations-improved.md` §4.1 asks for one
thing that no scramble produces: knots carrying a **certified `u <= k` witness of
exactly the interleaved shape** -- crossing changes separated by long isotopy
runs.  The recipe is three lines::

    w <- empty 1-braid
    repeat k times:
        apply a type-preserving walk of length l
        apply one crossing change at a chosen site
    emit (w, reverse(witness))

Everything interesting is in the word "reverse".  The forward path is a
construction; the *evidence* is the path read backwards, and that is only
evidence if every reversed step is itself a legal action of the shared braid
environment.  `rf_knots.actions` is not a group: it is a move set, and three of
its moves have inverses that the move set cannot always express.

* `COMMUTE`, `BRAID` and `CROSSING_CHANGE` are involutions.  Free.
* `INSERT(p, g, s)` is undone by `REDUCE(p)`.  Always available.
* `STABILIZE_{POS,NEG}` is undone by `DESTABILIZE`: the appended letter is the
  only one of absolute value `n`, so the uniqueness test that guards
  `DESTABILIZE` is satisfied by construction.
* `REDUCE(p)` is undone by `INSERT(p, ...)` **only sometimes**.  `INSERT` splices
  a cancelling pair *before* index `p` of a list, while `REDUCE` deletes a
  cancelling pair *cyclically*.  A reduction across the seam (`p = len - 1`), or
  one that eats the last two letters, lands on a word that is the right necklace
  but the wrong list, and the move set contains no rotation to fix it -- the two
  rotation moves were deliberately removed when the word became cyclic.
* `DESTABILIZE` is undone by `STABILIZE_*` only when the deleted top letter was
  the last letter of the word, because stabilisation appends.

So the walk is not free to use every legal move: it may only use moves whose
inverse it can name.  `inverse_action` names it or returns `None`, and every
selected move is checked by *applying* its candidate inverse and comparing
states, so the guarantee is verified rather than argued.  The resulting witness
is then handed to `UnknotWitness.from_actions`, which replays it independently.

Two facts make this cheaper than it looks:

* Every move in the set preserves the permutation of the closure -- `REDUCE`,
  `INSERT` and `CROSSING_CHANGE` do not change it at all, the braid relations
  hold in the symmetric group, and stabilisation merges a fixed point into the
  cycle it is attached to.  Starting from the one-strand braid, *every* state on
  the walk is a knot, so `BraidState`'s single-component check can never fire.
  It is still run on every state -- `UnknotWitness.from_actions` builds a
  `BraidState` per step and `BraidState.__post_init__` counts the components --
  because "can never" is a claim about this move set, and move sets are edited
  by people.  The walk therefore does not pay for the check twice.
* `UnknotWitness.from_states` would recover the same witness without any of the
  above, but it scans the whole action space per transition -- roughly a
  thousand legality tests where this module does one.  It is used in the tests
  as an independent oracle and never in the hot path.

The label is `u <= crossing_changes`, and `crossing_changes` is what the witness
literally contains, which is at most `k`.  It can be strictly less: see
`erase_state_loops`.  Nothing here ever claims `u = k`.

## Seeding the walk (note 22 §4.1b)

The recipe above starts at the empty one-braid, and M0 measured what that costs:
at `k = 5` the Murasugi bound on the emitted knots came out `mean 1.01, never 5`,
with 14% of instances literally the unknot.  The reason is arithmetic rather than
tuning.  `u` is the graph distance to the unknot in the crossing-change graph, so
it is 1-Lipschitz under one change, and a `k`-change walk from a seed `R` of a
knot `K` therefore carries three statements at once::

    d(R', R) <= k                    certified by construction
    u(R')    <= k + u_upper(K)       replay-complete only if K's bound is a program
    u(R')    >= L(K) - k             certified, inherited from the seed

The third is the one that was missing.  Seeding at the unknot means `L(K) = 0`,
so the inherited bound is `0`: **the generator was never weak, it was started at
the one point of the graph that certifies nothing.**

`SeedRepresentation` carries `(word, strands)` together with the certified `L`,
the program behind `u_upper` where there is one, and therefore the flag for
whether the composed witness is *replay-complete* or merely *catalogue-composed*.
Two families are built here and both are exact-interval nodes whose bounds come
from the word alone with no table lookup:

* `torus_seed(m)` -- `T(2, 2m+1)` as `sigma_1^(2m+1)`, `u = m` by the Milnor
  conjecture, with an explicit `m`-change program;
* `positive_braid_seed(...)` -- `u = g = (c - n + 1)/2` by Bennequin and Milnor,
  built backwards from the unknot so that the program is the construction read in
  reverse.

`UNKNOT_SEED` is the special case `L = 0`, `u_upper = 0`, always replay-complete,
and reproduces every byte of the §4.1 behaviour.

Two things the inheritance does **not** give.  It degrades one-for-one in `k`, so
depth has to come from the seed and only the last few changes from the walk; and
`d(R', R) <= k` is a statement about two concrete words, never about two knot
types -- promoting it would borrow an identification this module never made.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from rf_knots import reference
from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    KIND_NAMES,
    REDUCE,
    STABILIZE_NEG,
    STABILIZE_POS,
    ActionSpec,
)
from rf_knots.evidence import BraidState, LowerBoundClaim, UnknotWitness

Word = tuple[int, ...]
#: `(kind, position, generator, sign)` -- an action independent of any `ActionSpec`.
#: Seed programs are stored this way so that a seed built under one `(max_len,
#: max_strands)` can be walked under another; the flat encoding is not portable.
SemanticTuple = tuple[int, int, int, int]

EMPTY: Word = ()

ISOTOPY_KINDS: tuple[int, ...] = (
    REDUCE,
    COMMUTE,
    BRAID,
    INSERT,
    DESTABILIZE,
    STABILIZE_POS,
    STABILIZE_NEG,
)
"""The type-preserving half of the move set: everything but `CROSSING_CHANGE`.

`PASS` is excluded deliberately.  It is legal only when nothing else is, and a
walk that consumed its budget on `PASS` would report a length it did not travel.
"""

# -- inverses ------------------------------------------------------------------


def inverse_action(
    spec: ActionSpec,
    word: Word,
    strands: int,
    action: int,
    new_word: Word,
    new_strands: int,
) -> int | None:
    """The flat action taking `(new_word, new_strands)` back to `(word, strands)`.

    `None` when the move set cannot express it -- see the module docstring for
    the two cases where that happens.  The candidate is not merely constructed,
    it is applied and compared, so a `None` is a fact about this transition
    rather than about the table of cases above.
    """
    kind, position, _generator, _sign = spec.decode(action)
    if kind in (COMMUTE, BRAID, CROSSING_CHANGE):
        candidate = action
    elif kind == INSERT:
        candidate = spec.encode(REDUCE, position=position)
    elif kind == REDUCE:
        letter = word[position]
        candidate = spec.encode(
            INSERT,
            position=position,
            generator=abs(letter),
            sign=1 if letter > 0 else -1,
        )
    elif kind in (STABILIZE_POS, STABILIZE_NEG):
        candidate = spec.encode(DESTABILIZE)
    elif kind == DESTABILIZE:
        top = strands - 1
        index = next((i for i, x in enumerate(word) if abs(x) == top), None)
        if index is None:
            return None
        candidate = spec.encode(STABILIZE_POS if word[index] > 0 else STABILIZE_NEG)
    else:
        return None
    if not reference.is_legal(spec, new_word, new_strands, candidate, True):
        return None
    back_word, back_strands = reference.apply(spec, new_word, new_strands, candidate)
    if (back_word, back_strands) != (word, strands):
        return None
    return candidate


def invertible_successors(
    spec: ActionSpec, word: Word, strands: int, allow_crossing: bool = False
) -> list[tuple[int, int, Word, int]]:
    """`(action, inverse, new_word, new_strands)` for every two-way legal move.

    The reference enumerator with the one-way moves filtered out.  Slow -- it
    materialises the whole `INSERT` block -- and used by the tests to pin down
    what the sampler in `_sample_isotopy` is allowed to reach.
    """
    out: list[tuple[int, int, Word, int]] = []
    for action, new_word, new_strands in reference.successors(
        spec, word, strands, allow_crossing=allow_crossing
    ):
        back = inverse_action(spec, word, strands, action, new_word, new_strands)
        if back is not None:
            out.append((action, back, new_word, new_strands))
    return out


# -- configuration -------------------------------------------------------------


@dataclass(frozen=True)
class WalkWeights:
    """Relative sampling weight of each isotopy move kind, before steering.

    The two braid relations dominate because they are the moves that reshape a
    word without changing its length -- they are what makes a long word *hard*
    rather than merely long.  Insertion and stabilisation supply the length and
    the strands; reduction and destabilisation are what keep the walk from being
    a monotone growth, which would make the reversed witness a monotone descent
    and teach a network that shrinking always pays.
    """

    reduce: float = 1.0
    commute: float = 3.0
    braid: float = 3.0
    insert: float = 2.0
    destabilize: float = 1.0
    stabilize: float = 1.0

    #: Multiplier on the kinds that move towards the target length/strand count.
    toward_target: float = 8.0
    #: Multiplier on the kinds that move away from it.  Not zero: pinning the
    #: dimensions exactly would remove Markov moves from the walk entirely once
    #: the target was reached, and the Markov moves are half the point.
    away_from_target: float = 0.12

    def base(self, kind: int) -> float:
        return {
            REDUCE: self.reduce,
            COMMUTE: self.commute,
            BRAID: self.braid,
            INSERT: self.insert,
            DESTABILIZE: self.destabilize,
            STABILIZE_POS: self.stabilize,
            STABILIZE_NEG: self.stabilize,
        }[kind]


@dataclass(frozen=True)
class InterleavedConfig:
    """Dials for one family of interleaved instances.

    `crossing_changes` is `k` and `walk_length` is `l` from §4.1.  They are
    independent on purpose: `k` sets the crossing-change depth of the certificate
    and `l` sets how much reshaping sits between consecutive changes.  The depth
    five target wants `k = 5` and `l` large.

    `target_strands` and `target_length` are *drift* targets, not guarantees.
    The walk is steered towards them and the achieved distribution is reported by
    the shard writer; hard-pinning either one would mean deleting a whole move
    kind from the walk once the target was hit.  `max_len` and `max_strands` are
    the hard caps, and they are the `ActionSpec`'s, enforced by the move set
    itself.
    """

    crossing_changes: int = 5
    walk_length: int = 24
    target_strands: int = 4
    target_length: int = 32
    max_len: int = 64
    max_strands: int = 6
    cyclic_band_generators: bool = False

    #: How far above `target_strands` the walk may wander.  Zero pins the strand
    #: count once the target is reached; one buys Markov churn in the middle of
    #: the walk at the price of a strand count that is `target` or `target + 1`.
    strand_band: int = 0

    #: Isotopy moves applied *after* the last crossing change.  Zero reproduces
    #: §4.1 exactly, and means the first step of the reversed witness is a
    #: crossing change -- a real tell if these instances are used as puzzles.
    #: Set it positive to bury the last change; the certificate is unaffected.
    settle_walk_length: int = 0

    #: Erase revisited states from the forward path before reversing it.  This is
    #: the only mechanism by which the emitted witness can hold fewer than `k`
    #: crossing changes; see `erase_state_loops`.
    erase_loops: bool = True

    #: Reject the previous move's own inverse as the next move.  A walk that
    #: inserts a cancelling pair and immediately reduces it has travelled two of
    #: its `l` steps and gone nowhere.
    avoid_backtrack: bool = True

    weights: WalkWeights = field(default_factory=WalkWeights)

    def __post_init__(self) -> None:
        if self.crossing_changes < 0:
            raise ValueError("crossing_changes (k) cannot be negative")
        if self.walk_length < 0 or self.settle_walk_length < 0:
            raise ValueError("walk lengths cannot be negative")
        if self.max_strands < 1:
            raise ValueError("max_strands must be at least 1")
        if self.max_len < 1:
            raise ValueError("max_len must be at least 1")
        if not 1 <= self.target_strands <= self.max_strands:
            raise ValueError("target_strands must lie in [1, max_strands]")
        if self.strand_band < 0:
            raise ValueError("strand_band cannot be negative")
        if not 0 <= self.target_length <= self.max_len:
            raise ValueError("target_length must lie in [0, max_len]")

    def spec(self) -> ActionSpec:
        return ActionSpec(
            max_len=self.max_len,
            max_strands=self.max_strands,
            cyclic_band_generators=self.cyclic_band_generators,
        )

    def provenance(self) -> dict:
        """The dials, flat and JSON-safe, for a shard manifest."""
        return {
            "crossing_changes": self.crossing_changes,
            "walk_length": self.walk_length,
            "target_strands": self.target_strands,
            "target_length": self.target_length,
            "max_len": self.max_len,
            "max_strands": self.max_strands,
            "cyclic_band_generators": self.cyclic_band_generators,
            "strand_band": self.strand_band,
            "settle_walk_length": self.settle_walk_length,
            "erase_loops": self.erase_loops,
            "avoid_backtrack": self.avoid_backtrack,
            "weights": {
                "reduce": self.weights.reduce,
                "commute": self.weights.commute,
                "braid": self.weights.braid,
                "insert": self.weights.insert,
                "destabilize": self.weights.destabilize,
                "stabilize": self.weights.stabilize,
                "toward_target": self.weights.toward_target,
                "away_from_target": self.weights.away_from_target,
            },
        }


# -- seeds: sec 4.1b -----------------------------------------------------------

MILNOR = (
    "Milnor conjecture (Kronheimer-Mrowka 1993): u(T(p,q)) = (p-1)(q-1)/2"
)
BENNEQUIN_MILNOR = (
    "Bennequin's inequality is sharp on positive braids, so the Bennequin surface "
    "is genus-minimising; with the Milnor conjecture (Kronheimer-Mrowka) a "
    "positive braid on n strands with c letters whose closure is a knot has "
    "u = g_4 = g = (c - n + 1) / 2"
)
LIPSCHITZ = (
    "u is the graph distance to the unknot in the crossing-change graph, hence "
    "1-Lipschitz under one crossing change: |u(R') - u(R)| <= d(R', R)"
)


@dataclass(frozen=True)
class SeedRepresentation:
    """A representation to start the walk from, and what is certified about it.

    Section 4.1b: the walk of sec 4.1 begins at the empty one-braid only because that
    makes the reverse path self-certifying.  Nothing else requires it, and
    starting there is what made the emitted labels useless -- the inherited lower
    bound is `L(seed) - k` and the unknot has `L = 0`.

    Three things travel with a seed and each is a different kind of claim.

    * `lower_claims` is the **certified** lower bound `L`.  Only theorems belong
      here: `lower_bounds.computed_claims` on the seed word, the Milnor
      conjecture for a torus knot, Bennequin plus Milnor for a positive braid.  A
      catalogue row is not a theorem and `Invariants.unknotting_lower` is weaker
      than `computed_claims` (note 22 sec 1.2), so neither may be used.
    * `program` is a sequence of semantic actions taking the seed to the empty
      one-braid.  When it is present the composed witness replays end to end and
      the instance is **replay-complete**; when it is `None` the upper bound is
      **catalogue-composed** and no `UnknotWitness` is emitted at all.
    * `u_upper` is what the program (or the catalogue claim) says about `u(seed)`.

    A seed with `u_upper == certified_lower_bound` is an *exact-interval* node:
    both families built here are exact, which is what makes them worth walking
    from.
    """

    name: str
    family: str
    word: Word
    strands: int
    lower_claims: tuple[LowerBoundClaim, ...] = ()
    program: tuple[SemanticTuple, ...] | None = None
    u_upper: int | None = None
    u_upper_source: str = ""
    cyclic_band_generators: bool = False

    def __post_init__(self) -> None:
        if self.strands < 1:
            raise ValueError("a seed needs at least one strand")
        BraidState(self.word, self.strands, self.cyclic_band_generators)
        if self.program is not None and self.u_upper is None:
            raise ValueError("a seed carrying a program must state its u_upper")
        if self.u_upper is not None and self.u_upper < self.certified_lower_bound:
            raise ValueError(
                f"seed {self.name}: u_upper {self.u_upper} is below its own "
                f"certified lower bound {self.certified_lower_bound}"
            )

    @property
    def certified_lower_bound(self) -> int:
        """`L`.  Zero when nothing is certified -- which is the unknot's honest value."""
        return max((claim.value for claim in self.lower_claims), default=0)

    @property
    def replay_complete(self) -> bool:
        return self.program is not None

    @property
    def is_exact(self) -> bool:
        return self.u_upper is not None and self.u_upper == self.certified_lower_bound

    def program_actions(self, spec: ActionSpec) -> tuple[int, ...]:
        if self.program is None:
            raise ValueError(f"seed {self.name} carries no replayable program")
        return tuple(
            spec.encode(kind, position=position, generator=generator, sign=sign)
            for kind, position, generator, sign in self.program
        )

    def fits(self, spec: ActionSpec) -> bool:
        return (
            len(self.word) <= spec.max_len
            and self.strands <= spec.max_strands
            and self.cyclic_band_generators == spec.cyclic_band_generators
        )

    def verify_program(self, spec: ActionSpec) -> int:
        """Replay the program to the empty one-braid; return its crossing changes.

        Raises rather than returning a flag: an unverifiable program is not a
        weaker certificate, it is not a certificate.
        """
        witness = UnknotWitness.from_actions(
            self.word, self.strands, spec, self.program_actions(spec)
        )
        changes = witness.crossing_changes
        if self.u_upper is not None and changes != self.u_upper:
            raise ValueError(
                f"seed {self.name}: program holds {changes} crossing changes but "
                f"claims u_upper {self.u_upper}"
            )
        return changes

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "word": list(self.word),
            "strands": self.strands,
            "certified_lower_bound": self.certified_lower_bound,
            "u_upper": self.u_upper,
            "u_upper_source": self.u_upper_source,
            "replay_complete": self.replay_complete,
            "exact": self.is_exact,
            "program_moves": None if self.program is None else len(self.program),
            "lower_bounds": [claim.to_dict() for claim in self.lower_claims],
        }


UNKNOT_SEED = SeedRepresentation(
    name="0_1",
    family="unknot",
    word=EMPTY,
    strands=1,
    lower_claims=(),
    program=(),
    u_upper=0,
    u_upper_source="the empty one-braid is the unknot; the empty program replays",
)
"""The sec 4.1 special case: `L = 0`, `u_upper = 0`, always replay-complete."""

UNKNOT_SEED_BAND = SeedRepresentation(
    name=UNKNOT_SEED.name,
    family=UNKNOT_SEED.family,
    word=EMPTY,
    strands=1,
    program=(),
    u_upper=0,
    u_upper_source=UNKNOT_SEED.u_upper_source,
    cyclic_band_generators=True,
)
"""The same seed under `cyclic_band_generators`.  The empty word is the same
object in both alphabets, but a seed declares which one it belongs to so that a
band-generator walk can never be handed an Artin seed by accident."""


def default_seed(config: InterleavedConfig) -> SeedRepresentation:
    return UNKNOT_SEED_BAND if config.cyclic_band_generators else UNKNOT_SEED


def torus_seed(m: int, spec: ActionSpec | None = None) -> SeedRepresentation:
    """`T(2, 2m+1)` as `sigma_1^(2m+1)`, with `u = m` a theorem and a program.

    The lower bound is the Milnor conjecture, proved by Kronheimer and Mrowka;
    the upper bound is the explicit program below, which is `m` copies of
    "flip the first letter, cancel it against the second" followed by one
    destabilisation.  The word never reaches even length, so the intermediate
    states are knots throughout and `BraidState` never has to reject one.

    `m = 0` is `T(2,1)`, the unknot on two strands: a legitimate `L = 0` seed
    that is *not* the empty one-braid, which is useful for testing that the
    inheritance formula does not secretly depend on the start being empty.
    """
    if m < 0:
        raise ValueError("T(2, 2m+1) needs m >= 0")
    word: Word = (1,) * (2 * m + 1)
    program: list[SemanticTuple] = []
    for _ in range(m):
        program.append((CROSSING_CHANGE, 0, 1, 1))
        program.append((REDUCE, 0, 1, 1))
    program.append((DESTABILIZE, 0, 1, 1))
    claims = (
        LowerBoundClaim(
            m,
            "milnor-torus",
            "rf_knots.generator.torus_unknotting_number",
            "Kronheimer-Mrowka, Gauge theory for embedded surfaces I (1993)",
            {"p": 2, "q": 2 * m + 1, "theorem": MILNOR},
        ),
    )
    seed = SeedRepresentation(
        name=f"T(2,{2 * m + 1})",
        family="torus-2q",
        word=word,
        strands=2,
        lower_claims=claims,
        program=tuple(program),
        u_upper=m,
        u_upper_source="explicit m-change program, replayed to the empty one-braid",
    )
    if spec is not None:
        if not seed.fits(spec):
            raise ValueError(f"{seed.name} does not fit {spec}")
        seed.verify_program(spec)
    return seed


def _positive_isotopy(spec: ActionSpec, word: Word, strands: int) -> list[int]:
    """Legal `COMMUTE`/`BRAID` moves: the churn that preserves positivity.

    Both are involutions, both leave the length and the strand count alone, and
    both preserve the closure's permutation -- `COMMUTE` because the two
    transpositions it swaps are disjoint by its own legality test, `BRAID`
    because the braid relation holds in the symmetric group.  So a positive braid
    stays a positive braid on the same strands with the same genus, and its
    closure stays a knot.
    """
    available = _legal_kinds(spec, word, strands)
    out: list[int] = []
    for kind in (COMMUTE, BRAID):
        positions = available.get(kind)
        if isinstance(positions, list):
            out.extend(spec.encode(kind, position=p) for p in positions)
    return out


def positive_braid_seed(
    spec: ActionSpec,
    strands: int,
    genus: int,
    seed: int = 0,
    churn: int = 6,
    name: str | None = None,
) -> SeedRepresentation:
    """A random positive braid of genus `g`, built so that its program is free.

    `u = g = (c - n + 1)/2` is a theorem about *any* positive braid closing to a
    knot, so the lower bound could be read off any positive word.  What cannot be
    read off an arbitrary positive word is a **program**: turning `u <= g` into a
    replayable sequence needs the `g` crossings that matter to be findable, and
    greedy cancellation on a churned word is not guaranteed to find them.

    So the word is built *backwards from the unknot* instead, exactly as sec 4.1
    builds its instances.  From `sigma_1 sigma_2 ... sigma_{n-1}` -- the unknot on
    `n` strands, reached by `n-1` stabilisations -- each unit of genus is one
    `INSERT(p, g, -1)` followed by `CROSSING_CHANGE(p)`, which splices `(-g, +g)`
    and then flips the first, leaving two adjacent positive `g`s.  The inverse
    pair is `CROSSING_CHANGE(p)` then `REDUCE(p)`, so the reverse path is a
    `genus`-change program by construction, and `COMMUTE`/`BRAID` churn between
    insertions hides where those crossings are without disturbing any of it.

    The positivity and single-component conditions the theorem needs are then
    re-checked on the finished word rather than argued from the construction.
    """
    if strands < 2:
        raise ValueError("a positive braid seed needs at least two strands")
    if genus < 0:
        raise ValueError("genus cannot be negative")
    if strands > spec.max_strands:
        raise ValueError(f"strands {strands} exceeds max_strands {spec.max_strands}")
    if spec.cyclic_band_generators:
        raise ValueError("positive braid seeds are Artin-word only")
    rng = np.random.default_rng([int(seed), int(strands), int(genus)])
    path = _Path(spec)
    for _ in range(strands - 1):
        if not path.push(spec.encode(STABILIZE_POS)):  # pragma: no cover - defensive
            raise ValueError("stabilisation to the seed strand count failed")

    def churn_walk(budget: int) -> None:
        for _ in range(budget):
            options = _positive_isotopy(spec, path.word, path.strands)
            if not options:
                return
            path.push(options[int(rng.integers(len(options)))])

    for _ in range(genus):
        churn_walk(churn)
        position = int(rng.integers(max(len(path.word), 1)))
        generator = int(rng.integers(1, strands))
        insert = spec.encode(INSERT, position=position, generator=generator, sign=-1)
        if not path.push(insert):
            raise ValueError(
                f"max_len {spec.max_len} is too small for genus {genus} on "
                f"{strands} strands"
            )
        if not path.push(spec.encode(CROSSING_CHANGE, position=position)):
            raise AssertionError("a crossing change on a non-empty word is legal")
    churn_walk(churn)

    word, count = path.word, path.strands
    if any(letter <= 0 for letter in word):  # pragma: no cover - defensive
        raise AssertionError(f"construction left a negative letter: {word}")
    if reference.num_components(word, count) != 1:  # pragma: no cover - defensive
        raise AssertionError("construction left a link")
    from rf_knots.generator import positive_braid_unknotting_number

    value = positive_braid_unknotting_number(len(word), count)
    if value != genus:  # pragma: no cover - defensive
        raise AssertionError(f"genus {value} from the word, {genus} requested")
    claims = (
        LowerBoundClaim(
            value,
            "positive-braid-genus",
            "rf_knots.generator.positive_braid_unknotting_number",
            "Bennequin 1983; Kronheimer-Mrowka 1993",
            {
                "crossings": len(word),
                "strands": count,
                "genus": value,
                "positive": True,
                "theorem": BENNEQUIN_MILNOR,
            },
        ),
    )
    representation = SeedRepresentation(
        name=name or f"P({count},{len(word)})#{seed}",
        family="positive-braid",
        word=word,
        strands=count,
        lower_claims=claims,
        program=tuple(spec.decode(action) for action in reversed(path.inverses)),
        u_upper=value,
        u_upper_source=(
            "reverse of the backward construction: one crossing change per unit "
            "of genus, replayed to the empty one-braid"
        ),
    )
    representation.verify_program(spec)
    return representation


def computed_seed(
    word: Word,
    strands: int,
    name: str,
    program: tuple[SemanticTuple, ...] | None = None,
    u_upper: int | None = None,
    u_upper_source: str = "",
    family: str = "computed",
) -> SeedRepresentation:
    """A seed whose `L` is `lower_bounds.computed_claims` on the word itself.

    The only catalogue route this module offers, and it is not really a catalogue
    route: nothing is looked up.  `computed_claims` runs the signature, the
    Montesinos obstruction and `tau` on the given word, so every claim it returns
    is a theorem about *this* representation.  `Invariants.unknotting_lower` is
    deliberately not used -- sec 1.2 of note 22 records that it is weaker and returns
    the smaller bound silently.
    """
    from rf_knots.lower_bounds import computed_claims

    return SeedRepresentation(
        name=name,
        family=family,
        word=tuple(word),
        strands=int(strands),
        lower_claims=computed_claims(tuple(word), int(strands)),
        program=program,
        u_upper=u_upper,
        u_upper_source=u_upper_source,
    )


@dataclass(frozen=True)
class InterleavedInstance:
    """One generated knot and the reverse witness that certifies `u <= k`."""

    word: Word
    strands: int
    #: The composed witness `R' -> seed -> unknot`, or `None` when the seed
    #: carried no program and the composition is only catalogue-deep.
    witness: UnknotWitness | None
    #: `k` as requested.  The *claim* is `certified_upper_bound`, never this.
    crossing_change_budget: int
    #: Positions the forward construction changed, in forward order.  These are
    #: sites in the word as it stood at the time, not in the emitted word; the
    #: emitted-word sites are recoverable from the witness (§5.3 wants those).
    crossing_change_sites: tuple[int, ...]
    #: Isotopy moves actually travelled per segment.  Shorter than `l` only when
    #: the `ActionSpec` caps left nothing legal to do.
    walk_lengths: tuple[int, ...]
    #: Forward transitions dropped by loop erasure, and crossing changes among
    #: them.  Both zero on a walk that never revisited a state.
    erased_steps: int = 0
    erased_crossing_changes: int = 0
    #: Crossing changes the construction could not place because the word was
    #: empty and the caps forbade growing it.  Zero unless `max_len` is tiny.
    skipped_crossing_changes: int = 0
    seed: int = 0
    index: int = 0
    #: The representation the walk started from.  `UNKNOT_SEED` reproduces sec 4.1.
    seed_representation: SeedRepresentation = UNKNOT_SEED
    #: Flat actions taking the emitted word back to the seed word.  This is the
    #: Gordian half of the certificate and it is sound whether or not the seed
    #: carries a program.
    reverse_actions: tuple[int, ...] = ()
    #: Crossing changes among `reverse_actions`: `d(R', R) <= this <= k`.
    walk_crossing_changes: int = 0

    @property
    def certified_upper_bound(self) -> int | None:
        """`u(K) <= this`.  The number of crossing changes the witness contains.

        `None` only when the seed carried no program, in which case there is no
        witness to count and `u_upper` is the composed catalogue claim instead.
        """
        if self.witness is None:
            return self.u_upper
        return self.witness.crossing_changes

    @property
    def d_upper(self) -> int:
        """`d(R', R) <= this`, by construction.  At most `k`."""
        return self.walk_crossing_changes

    @property
    def u_lower_inherited(self) -> int:
        """`u(R') >= L(seed) - d`, clamped at zero.  sec 4.1b(3).

        No invariant of `R'` is computed to obtain this.  It is the seed's
        certified bound minus the walk length, and it is exactly why seeding at
        the unknot certified nothing: `L = 0` makes every instance `>= 0`.
        """
        return max(0, self.seed_representation.certified_lower_bound - self.d_upper)

    @property
    def u_upper(self) -> int | None:
        """`u(R') <= d + u_upper(seed)`, or `None` when the seed has no bound."""
        seed_upper = self.seed_representation.u_upper
        if seed_upper is None:
            return None
        return self.d_upper + seed_upper

    @property
    def replay_complete(self) -> bool:
        """True when the composed witness replays end to end to the unknot."""
        return self.witness is not None

    @property
    def witness_provenance(self) -> str:
        return "replay-complete" if self.replay_complete else "catalogue-composed"

    @property
    def is_seed(self) -> bool:
        """True when loop erasure ate the whole walk and `R'` is the seed again."""
        return (
            self.word == self.seed_representation.word
            and self.strands == self.seed_representation.strands
        )

    @property
    def state(self) -> BraidState:
        if self.witness is not None:
            return self.witness.start
        return BraidState(
            self.word,
            self.strands,
            self.seed_representation.cyclic_band_generators,
        )

    @property
    def instance_id(self) -> str:
        from rf_knots.evidence import braid_instance_id

        if self.witness is not None:
            return self.witness.instance_id
        return braid_instance_id(self.word, self.strands)

    @property
    def is_trivial(self) -> bool:
        """True when the emitted word is already the empty one-braid.

        A `u <= 0` witness for the unknot is correct and useless.  It happens
        when loop erasure eats the entire path, which needs `l` small enough that
        the changes cancel each other outright.
        """
        return not self.word and self.strands == 1

    @property
    def witness_crossing_change_steps(self) -> tuple[int, ...]:
        """Indices into `witness.steps` that are crossing changes.

        The reversed order, so step 0 is the change nearest the emitted word.
        This is the raw material for §5.3's per-site head: the *positions* those
        steps carry are known-good sites at states the witness also carries.
        """
        if self.witness is None:
            return ()
        name = KIND_NAMES[CROSSING_CHANGE]
        return tuple(
            index
            for index, step in enumerate(self.witness.steps)
            if step.action.kind == name
        )

    def metrics(self) -> dict:
        return {
            "word_length": len(self.word),
            "strands": self.strands,
            "crossing_change_budget": self.crossing_change_budget,
            "certified_upper_bound": self.certified_upper_bound,
            "witness_moves": 0 if self.witness is None else self.witness.moves,
            "seed_name": self.seed_representation.name,
            "seed_family": self.seed_representation.family,
            "seed_certified_lower_bound": (
                self.seed_representation.certified_lower_bound
            ),
            "d_upper": self.d_upper,
            "u_lower_inherited": self.u_lower_inherited,
            "u_upper": self.u_upper,
            "replay_complete": self.replay_complete,
            "witness_provenance": self.witness_provenance,
            "walk_moves": len(self.reverse_actions),
            "walk_lengths": list(self.walk_lengths),
            "crossing_change_sites": list(self.crossing_change_sites),
            "witness_crossing_change_steps": list(self.witness_crossing_change_steps),
            "erased_steps": self.erased_steps,
            "erased_crossing_changes": self.erased_crossing_changes,
            "skipped_crossing_changes": self.skipped_crossing_changes,
            "trivial": self.is_trivial,
        }


# -- loop erasure --------------------------------------------------------------


def erase_state_loops(
    actions: Sequence[int], states: Sequence[tuple[Word, int]]
) -> tuple[list[int], list[tuple[Word, int]]]:
    """Drop every closed sub-walk, keeping the endpoints.

    `states[0]` is the start and `states[i + 1]` is the result of `actions[i]`.
    If a state recurs the segment between the two occurrences is a loop: it
    returns the braid to where it already was, so deleting it leaves a shorter
    path with the same endpoints.  The final state is unchanged either way.

    **This is where `k` can shrink.**  Two crossing changes at the same site with
    an isotopy run between them that happens to return -- likeliest at small `l`
    -- close a loop, and both changes vanish with it.  The witness then certifies
    `u <= k - 2`, which is a stronger and equally true statement.  Papering over
    it by re-inserting the changes would emit a witness with a redundant pair in
    it and call the redundancy a label.
    """
    kept_actions: list[int] = []
    kept_states: list[tuple[Word, int]] = [states[0]]
    seen: dict[tuple[Word, int], int] = {states[0]: 0}
    for action, state in zip(actions, states[1:], strict=True):
        previous = seen.get(state)
        if previous is None:
            seen[state] = len(kept_states)
            kept_states.append(state)
            kept_actions.append(action)
            continue
        for stale in kept_states[previous + 1 :]:
            seen.pop(stale, None)
        del kept_states[previous + 1 :]
        del kept_actions[previous:]
    return kept_actions, kept_states


# -- the walk ------------------------------------------------------------------


def _legal_kinds(
    spec: ActionSpec, word: Word, strands: int
) -> dict[int, list[int] | bool]:
    """Legal, *invertible* isotopy moves grouped by kind, in one pass.

    `REDUCE` and `DESTABILIZE` carry their admissible positions; the other kinds
    are flags because their candidate sets are either uniform (`COMMUTE`,
    `BRAID`, listed as positions too) or sampled parametrically (`INSERT`).
    Mirrors `reference.is_legal` exactly, including the seam rules for band
    generators; `test_interleaved_generator.py` cross-checks the two.
    """
    length = len(word)
    cyclic = spec.cyclic_band_generators
    reduce_positions: list[int] = []
    commute_positions: list[int] = []
    braid_positions: list[int] = []

    # One pass, with the magnitudes hoisted.  `abs` was the single most-called
    # builtin in a profile of this module -- three passes over the word, each
    # recomputing `abs(word[p])` for the same `p` up to five times.
    magnitudes = [x if x > 0 else -x for x in word]
    reduce_stop = max(length - 2, 1)
    # `REDUCE`'s inverse is an `INSERT`, which needs two strands and -- in band
    # mode, where the seam generator is `sigma_n` itself -- at least three before
    # it will re-insert a seam letter.
    reducible = strands >= 2
    top = strands - 1
    top_positions: list[int] = []
    seam_absent = True

    if length:
        for p in range(length):
            left = word[p]
            size = magnitudes[p]
            if size == top:
                top_positions.append(p)
            if cyclic and size == strands:
                seam_absent = False
            if length < 2:
                continue
            q = p + 1
            if q == length:
                q = 0
            right = word[q]
            if left == -right and q != 0 and p < reduce_stop and reducible:
                # Invertible only when `INSERT(p, ...)` lands on the same list.
                # `q != 0` rules out the seam; `p < max(length - 2, 1)` rules out
                # eating the tail, with the empty result exempted.
                if size != strands or strands >= 3:
                    reduce_positions.append(p)
            distance = size - magnitudes[q]
            if distance < 0:
                distance = -distance
            if cyclic:
                commutes = size != magnitudes[q] and distance not in (1, strands - 1)
            else:
                commutes = distance >= 2
            if commutes:
                commute_positions.append(p)
            if length >= 3:
                r = q + 1
                if r == length:
                    r = 0
                adjacent = distance == 1 or (cyclic and distance == strands - 1)
                if adjacent and word[r] == left and (left > 0) == (right > 0):
                    braid_positions.append(p)

    # Legal when the top letter is unique; invertible only when it is also the
    # final letter, because `STABILIZE_*` appends and nothing rotates.
    destabilize = (
        strands >= 2
        and seam_absent
        and len(top_positions) == 1
        and top_positions[0] == length - 1
    )
    stabilize = (
        strands < spec.max_strands and length + 1 <= spec.max_len and seam_absent
    )
    insert = strands >= 2 and length + 2 <= spec.max_len

    out: dict[int, list[int] | bool] = {}
    if reduce_positions:
        out[REDUCE] = reduce_positions
    if commute_positions:
        out[COMMUTE] = commute_positions
    if braid_positions:
        out[BRAID] = braid_positions
    if insert:
        out[INSERT] = True
    if destabilize:
        out[DESTABILIZE] = True
    if stabilize:
        out[STABILIZE_POS] = True
        out[STABILIZE_NEG] = True
    return out


def _steer(config: InterleavedConfig, kind: int, length: int, strands: int) -> float:
    """Base weight of `kind`, multiplied by how it moves the two dimensions."""
    weights = config.weights
    value = weights.base(kind)
    if kind in (STABILIZE_POS, STABILIZE_NEG):
        # Hard zero above the band, not merely damped.  Stabilisation is always
        # legal while under the caps, but its inverse is available only when the
        # lone top letter happens to sit last, so a damped weight is a ratchet:
        # the walk climbs away from the target and cannot climb back.  Measured
        # at 4.4 strands against a target of 4 before this was a cutoff.
        if strands >= config.target_strands + config.strand_band:
            return 0.0
        if strands < config.target_strands:
            return value * weights.toward_target
        return value * weights.away_from_target
    if kind == DESTABILIZE:
        if strands > config.target_strands:
            return value * weights.toward_target
        return value * weights.away_from_target
    if kind == INSERT:
        if length < config.target_length:
            return value * weights.toward_target
        return value * weights.away_from_target
    if kind == REDUCE:
        if length > config.target_length:
            return value * weights.toward_target
        return value * weights.away_from_target
    return value


def _sample_isotopy(
    spec: ActionSpec,
    config: InterleavedConfig,
    word: Word,
    strands: int,
    rng: np.random.Generator,
    forbidden: int | None,
) -> int | None:
    """One legal, invertible isotopy action, or `None` if there are none."""
    available = _legal_kinds(spec, word, strands)
    if not available:
        return None
    length = len(word)
    kinds = sorted(available)
    weights = [_steer(config, kind, length, strands) for kind in kinds]
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0] * len(weights)
        total = float(len(weights))

    # Hand-rolled roulette rather than `Generator.choice(p=...)`: the numpy call
    # validates and normalises the probability vector on every draw and cost a
    # measured third of this module's wall clock.
    #
    # Two draws: the second only matters when the first hit the forbidden move,
    # which is one specific action out of a block, so a retry almost always
    # clears it and an unlucky repeat costs nothing but a wasted walk step.
    for _ in range(4):
        threshold = float(rng.random()) * total
        running = 0.0
        kind = kinds[-1]
        for candidate, weight in zip(kinds, weights, strict=True):
            running += weight
            if running >= threshold:
                kind = candidate
                break
        payload = available[kind]
        if kind in (REDUCE, COMMUTE, BRAID):
            positions = payload
            assert isinstance(positions, list)
            action = spec.encode(
                kind, position=positions[int(rng.integers(len(positions)))]
            )
        elif kind == INSERT:
            stop = strands + 1 if spec.cyclic_band_generators and strands >= 3 else strands
            generator = int(rng.integers(1, stop))
            sign = 1 if rng.integers(2) == 0 else -1
            position = int(rng.integers(max(length, 1)))
            action = spec.encode(INSERT, position=position, generator=generator, sign=sign)
        else:
            action = spec.encode(kind)
        if action != forbidden:
            return action
    return action


def _grow(
    spec: ActionSpec, word: Word, strands: int, rng: np.random.Generator
) -> int | None:
    """One move that lengthens the word, for when a crossing change has no site."""
    if strands >= 2 and len(word) + 2 <= spec.max_len:
        stop = strands + 1 if spec.cyclic_band_generators and strands >= 3 else strands
        return spec.encode(
            INSERT,
            position=int(rng.integers(max(len(word), 1))),
            generator=int(rng.integers(1, stop)),
            sign=1 if rng.integers(2) == 0 else -1,
        )
    seam_absent = not spec.cyclic_band_generators or all(abs(x) != strands for x in word)
    if strands < spec.max_strands and len(word) + 1 <= spec.max_len and seam_absent:
        return spec.encode(STABILIZE_POS if rng.integers(2) == 0 else STABILIZE_NEG)
    return None


class _Path:
    """Forward path accumulator that refuses any move it cannot reverse."""

    def __init__(self, spec: ActionSpec, word: Word = EMPTY, strands: int = 1):
        self.spec = spec
        self.word: Word = tuple(word)
        self.strands = int(strands)
        self.actions: list[int] = []
        self.states: list[tuple[Word, int]] = [(self.word, self.strands)]
        self.inverses: list[int] = []

    def push(self, action: int) -> bool:
        spec = self.spec
        if not reference.is_legal(spec, self.word, self.strands, action, True):
            return False
        new_word, new_strands = reference.apply(spec, self.word, self.strands, action)
        back = inverse_action(
            spec, self.word, self.strands, action, new_word, new_strands
        )
        if back is None:
            return False
        self.word, self.strands = new_word, new_strands
        self.actions.append(action)
        self.inverses.append(back)
        self.states.append((new_word, new_strands))
        return True


def generate(
    config: InterleavedConfig,
    seed: int = 0,
    index: int = 0,
    seed_representation: SeedRepresentation | None = None,
) -> InterleavedInstance:
    """One instance.  Deterministic in `(seed, index, seed_representation)`.

    `seed_representation=None` is sec 4.1: the walk starts at the empty one-braid,
    `L = 0`, `u_upper = 0`, and the emitted witness is exactly what this module
    emitted before sec 4.1b existed.
    """
    spec = config.spec()
    representation = (
        default_seed(config) if seed_representation is None else seed_representation
    )
    if not representation.fits(spec):
        raise ValueError(
            f"seed {representation.name} ({len(representation.word)} letters on "
            f"{representation.strands} strands) does not fit the action spec "
            f"(max_len {spec.max_len}, max_strands {spec.max_strands}, cyclic "
            f"{spec.cyclic_band_generators})"
        )
    rng = np.random.default_rng([int(seed), int(index)])
    path = _Path(spec, representation.word, representation.strands)

    sites: list[int] = []
    walks: list[int] = []
    skipped = 0

    def walk(budget: int) -> int:
        travelled = 0
        forbidden: int | None = None
        for _ in range(budget):
            action = _sample_isotopy(
                spec, config, path.word, path.strands, rng, forbidden
            )
            if action is None:
                break
            if not path.push(action):
                # A sample the verifier rejected costs a step of the budget and
                # nothing else; ending the walk here would silently shorten `l`.
                forbidden = None
                continue
            travelled += 1
            forbidden = path.inverses[-1] if config.avoid_backtrack else None
        return travelled

    for _ in range(config.crossing_changes):
        walks.append(walk(config.walk_length))
        if not path.word:
            # No crossings to change.  Grow if the caps allow it, and otherwise
            # record the change as unplaced rather than pretending it happened.
            for _ in range(4):
                grow = _grow(spec, path.word, path.strands, rng)
                if grow is None or not path.push(grow):
                    break
                if path.word:
                    break
        if not path.word:
            skipped += 1
            continue
        position = int(rng.integers(len(path.word)))
        if not path.push(spec.encode(CROSSING_CHANGE, position=position)):
            skipped += 1
            continue
        sites.append(position)

    if config.settle_walk_length:
        walks.append(walk(config.settle_walk_length))

    actions, states = path.actions, path.states
    erased_steps = 0
    erased_changes = 0
    if config.erase_loops:
        kept_actions, _kept_states = erase_state_loops(actions, states)
        if len(kept_actions) != len(actions):
            erased_steps = len(actions) - len(kept_actions)
            before = sum(spec.decode(a)[0] == CROSSING_CHANGE for a in actions)
            after = sum(spec.decode(a)[0] == CROSSING_CHANGE for a in kept_actions)
            erased_changes = before - after
            # The kept prefix-with-holes is still a genuine forward path, so its
            # inverses are recomputed rather than sliced out of the old list.
            path = _Path(spec, representation.word, representation.strands)
            for action in kept_actions:
                if not path.push(action):  # pragma: no cover - defensive
                    raise AssertionError("loop erasure produced an illegal path")

    reverse = tuple(reversed(path.inverses))
    walk_changes = sum(spec.decode(action)[0] == CROSSING_CHANGE for action in reverse)

    if representation.program is None:
        # No program, so no `UnknotWitness` -- one would have to end at the empty
        # one-braid and this path ends at the seed.  The walk is still replayed
        # here, because `d(R', R) <= k` is the claim the pair carries and an
        # unreplayed claim is not one.
        witness = None
        _verify_walk(spec, path.word, path.strands, reverse, representation)
    else:
        witness = UnknotWitness.from_actions(
            path.word, path.strands, spec, reverse + representation.program_actions(spec)
        )
        # The composed witness passes through the seed at the join.  Checking it
        # costs one comparison and turns "by construction" into "verified".
        joint = (
            witness.start
            if not reverse
            else witness.steps[len(reverse) - 1].after
        )
        if (joint.word, joint.strands) != (
            representation.word,
            representation.strands,
        ):  # pragma: no cover - defensive
            raise AssertionError("the composed witness does not pass through the seed")

    return InterleavedInstance(
        word=path.word,
        strands=path.strands,
        witness=witness,
        crossing_change_budget=config.crossing_changes,
        crossing_change_sites=tuple(sites),
        walk_lengths=tuple(walks),
        erased_steps=erased_steps,
        erased_crossing_changes=erased_changes,
        skipped_crossing_changes=skipped,
        seed=int(seed),
        index=int(index),
        seed_representation=representation,
        reverse_actions=reverse,
        walk_crossing_changes=walk_changes,
    )


def _verify_walk(
    spec: ActionSpec,
    word: Word,
    strands: int,
    actions: Sequence[int],
    representation: SeedRepresentation,
) -> None:
    """Replay `R' -> R` and raise unless it lands exactly on the seed word."""
    current, count = tuple(word), int(strands)
    for position, action in enumerate(actions):
        if not reference.is_legal(spec, current, count, int(action), True):
            raise ValueError(f"walk step {position} is illegal: {spec.describe(action)}")
        current, count = reference.apply(spec, current, count, int(action))
        BraidState(current, count, spec.cyclic_band_generators)
    if (current, count) != (representation.word, representation.strands):
        raise ValueError(
            f"the walk ends at {current} on {count} strands, not at seed "
            f"{representation.name}"
        )


def generate_many(
    config: InterleavedConfig,
    seed: int = 0,
    count: int = 1,
    start: int = 0,
    skip_trivial: bool = True,
    accept: Callable[[InterleavedInstance], bool] | None = None,
    max_attempts_per_instance: int = 16,
    seed_representation: SeedRepresentation | None = None,
) -> Iterator[InterleavedInstance]:
    """`count` instances, resumable at `start`.

    `skip_trivial` drops instances whose emitted word is the empty one-braid.
    They are correct -- `u(unknot) <= 0` -- and worthless as positives.

    `accept` is the hook for everything a caller can only decide by paying for
    it.  The measured reason it exists: at `k = 5` and `l = 24` the Murasugi
    signature bound on the emitted knots is `0` or `1` for three quarters of
    them, so `u <= 5` is a true statement about a trefoil.  A caller that wants
    positives whose difficulty is certified rather than requested passes a
    predicate computing a lower bound and rejecting what does not clear it; the
    price is one Seifert matrix per candidate.

    The attempt index advances past every rejection, so the stream stays
    deterministic and a resume at `start` lands on the same instance it would
    have.  `start` is a *stream* index, not a count of what was emitted -- the
    shard manifest records both.
    """
    if count < 0:
        raise ValueError("count cannot be negative")
    attempt = int(start)
    emitted = 0
    while emitted < count:
        limit = attempt + max_attempts_per_instance
        while attempt < limit:
            instance = generate(
                config,
                seed=seed,
                index=attempt,
                seed_representation=seed_representation,
            )
            attempt += 1
            if skip_trivial and instance.is_trivial:
                continue
            if accept is not None and not accept(instance):
                continue
            yield instance
            emitted += 1
            break
        else:
            raise RuntimeError(
                f"{max_attempts_per_instance} consecutive instances rejected at "
                f"stream index {attempt}: the configuration or the `accept` "
                "predicate cannot be satisfied"
            )


def inherited_claim(instance: InterleavedInstance) -> LowerBoundClaim | None:
    """`u(R') >= L(seed) - d` as a claim, with the seed's theorem attached.

    `None` when the seed certifies nothing or the walk has eaten the whole
    bound, because a claim of `u >= 0` is true of every knot and recording it
    would inflate the count of instances that carry a lower bound.
    """
    seed = instance.seed_representation
    value = instance.u_lower_inherited
    if value <= 0 or not seed.lower_claims:
        return None
    best = max(seed.lower_claims, key=lambda claim: claim.value)
    return LowerBoundClaim(
        value,
        "inherited-lipschitz",
        "rf_knots.interleaved_generator.inherited_claim",
        best.citation,
        {
            "seed": seed.name,
            "seed_lower_bound": seed.certified_lower_bound,
            "seed_method": best.method,
            "seed_source": best.source,
            "crossing_changes_walked": instance.d_upper,
            "theorem": LIPSCHITZ,
        },
    )


# -- Gordian intervals: sec 4.1b, last paragraph --------------------------------


@dataclass(frozen=True)
class LipschitzValues:
    """Values of the 1-Lipschitz concordance invariants at one representation.

    `signature_half` is `sigma/2` **signed**, not `|sigma|/2`.  The Lipschitz
    statement is about `f`, so the difference of the signed values is what bounds
    the distance, and it is never weaker: for a pair either side of the unknot
    `|sigma_1/2 - sigma_2/2|` can be twice `||sigma_1|/2 - |sigma_2|/2|`.
    `sigma` is even for a knot, so the halving is exact.
    """

    signature_half: int | None = None
    tau: int | None = None

    def to_dict(self) -> dict:
        return {"signature_half": self.signature_half, "tau": self.tau}


def lipschitz_values(word: Word, strands: int) -> LipschitzValues:
    """`sigma/2` and `tau`, or `None` where the backend cannot supply them.

    The empty word is the unknot and both vanish there without any backend.
    """
    if not word:
        return LipschitzValues(0, 0)
    from rf_knots import seifert

    try:
        half = seifert.signature(word, strands) // 2
    except (ValueError, seifert.BackendUnavailable):
        half = None
    try:
        value = seifert.tau(word, strands)
    except (ValueError, KeyError, RuntimeError, seifert.BackendUnavailable):
        value = None
    return LipschitzValues(half, value)


def gordian_lower_bound(
    left: LipschitzValues, right: LipschitzValues
) -> tuple[int, str]:
    """`max over 1-Lipschitz f of |f(left) - f(right)|`, and which `f` won.

    Both `sigma/2` and `tau` change by at most one under a single crossing
    change, so each supplies a lower bound on the number of changes between two
    representations.  `("none", 0)` when neither is available on both sides,
    which is the honest answer and not a zero bound dressed up as a computation.
    """
    best, method = 0, "none"
    if left.signature_half is not None and right.signature_half is not None:
        value = abs(left.signature_half - right.signature_half)
        if value > best:
            best, method = value, "signature"
    if left.tau is not None and right.tau is not None:
        value = abs(left.tau - right.tau)
        if value > best:
            best, method = value, "ozsvath-szabo-tau"
    return best, method


@dataclass(frozen=True)
class GordianInterval:
    """A two-sided certified interval on `d(R', R)`, in the trainer's shape.

    `rf_knots.embedding_metric.MetricEmbeddingObjective` is an interval loss with
    `lower` and `upper` fields and a `witness_semantic_length` used to prioritise
    long zero-change witnesses; `rf_knots.embedding_supervision.DistanceInterval`
    is the row it reads.  This is that row with the two words carried inline,
    because these representations have no id in anybody's database yet.
    """

    metric: str
    left_word: Word
    left_strands: int
    right_word: Word
    right_strands: int
    lower: int
    upper: int
    witness_semantic_length: int
    lower_method: str = "none"
    provenance: str = "interleaved-seeded-generator"
    split: str = "train"

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"interval [{self.lower}, {self.upper}] is empty -- one of the "
                "two sides is wrong, and both are supposed to be theorems"
            )

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "left": {"word": list(self.left_word), "strands": self.left_strands},
            "right": {"word": list(self.right_word), "strands": self.right_strands},
            "lower": self.lower,
            "upper": self.upper,
            "witness_semantic_length": self.witness_semantic_length,
            "lower_method": self.lower_method,
            "provenance": self.provenance,
            "split": self.split,
            "exact": self.exact,
        }


def gordian_interval(
    instance: InterleavedInstance,
    left: LipschitzValues | None = None,
    right: LipschitzValues | None = None,
    split: str = "train",
) -> GordianInterval:
    """The `(R', R)` pair with `upper = d` by construction and a computed `lower`.

    `left`/`right` are passed in rather than computed here so that a caller
    walking many instances from one seed pays for the seed's invariants once.
    """
    if left is None:
        left = lipschitz_values(instance.word, instance.strands)
    if right is None:
        seed = instance.seed_representation
        right = lipschitz_values(seed.word, seed.strands)
    lower, method = gordian_lower_bound(left, right)
    upper = instance.d_upper
    if lower > upper:  # pragma: no cover - would mean a broken invariant backend
        raise ValueError(
            f"computed lower bound {lower} exceeds the constructed upper bound "
            f"{upper}; one of the two is wrong"
        )
    return GordianInterval(
        metric="cc",
        left_word=instance.word,
        left_strands=instance.strands,
        right_word=instance.seed_representation.word,
        right_strands=instance.seed_representation.strands,
        lower=lower,
        upper=upper,
        witness_semantic_length=len(instance.reverse_actions),
        lower_method=method,
        split=split,
    )


# -- quarantine: sec 4.3, moved to seed selection by sec 4.1b -------------------


@dataclass(frozen=True)
class Quarantine:
    """The sec 4.3 exclusion set, as three screens of decreasing precision.

    **What this guarantees.**  No seed and no emitted word is one of the stored
    quarantined *representations*; no seed and no emitted word is identified as
    one of the quarantined *names*; no emitted word shares the target's
    `(determinant, |signature|)` **fingerprint**.  The fingerprint screen is
    deliberately over-broad -- other knots share `(49, 0)` with `7_1 # m(7_1)`
    and they are rejected too.

    **What it does not guarantee.**  sec 4.1b asks for closure under the `k`-ball
    and says outright that exact closure is expensive; it is not computed here.
    A word within `k` crossing changes of the target, that is not itself stored,
    is not named, and does not share the fingerprint, passes.  The screen is the
    practical rule sec 4.1b names, not the closure it would prefer.
    """

    words: frozenset[tuple[Word, int]] = frozenset()
    names: frozenset[str] = frozenset()
    fingerprints: frozenset[tuple[int, int]] = frozenset()
    composite_summands: frozenset[str] = frozenset()
    description: tuple[str, ...] = ()

    def screen_word(self, word: Word, strands: int) -> str | None:
        """A rejection reason, or `None` when the representation is admissible."""
        key = (tuple(word), int(strands))
        if key in self.words:
            return "quarantined-representation"
        reduced = (reference.free_reduce(tuple(word)), int(strands))
        if reduced != key and reduced in self.words:
            return "quarantined-representation-free-reduction"
        return None

    def screen_fingerprint(self, determinant: int, signature: int) -> str | None:
        if (int(determinant), abs(int(signature))) in self.fingerprints:
            return "quarantined-fingerprint"
        return None

    def screen_name(self, name: str | None, summands: Sequence[str] = ()) -> str | None:
        if name is not None and name in self.names:
            return "quarantined-name"
        if summands and self.composite_summands.issubset({str(x) for x in summands}):
            return "quarantined-composite"
        return None

    def screen_seed(self, seed: SeedRepresentation) -> str | None:
        reason = self.screen_word(seed.word, seed.strands)
        if reason is not None:
            return reason
        return self.screen_name(seed.name)

    def provenance(self) -> dict:
        return {
            "representations": len(self.words),
            "names": sorted(self.names),
            "fingerprints": sorted(list(x) for x in self.fingerprints),
            "composite_summands": sorted(self.composite_summands),
            "screens": list(self.description),
            "k_ball_closure": (
                "not computed; the screen is exact on stored representations, "
                "exact on identified names, and over-broad on fingerprints, but "
                "a word within k crossing changes of the target that matches "
                "none of the three is not detected"
            ),
        }


def _masked_words(word: Sequence[int], radius: int) -> set[Word]:
    """Every word obtained by flipping at most `radius` letters of `word`."""
    from itertools import combinations

    base = tuple(int(x) for x in word)
    out: set[Word] = {base}
    positions = range(len(base))
    for size in range(1, radius + 1):
        for choice in combinations(positions, size):
            flipped = list(base)
            for index in choice:
                flipped[index] = -flipped[index]
            out.add(tuple(flipped))
    return out


BRITTENHAM_HERMILLER_B5: Word = (
    1, -4, 2, 3, 3, 3, 2, 3, 2, 2, 4, -3, -3, -3, -3, -1, -3, -2, -3, -3,
)
"""The 20-letter B5 word of sec 4.3 item 1; its closure is `7_1 # m(7_1)`."""

TARGET_FINGERPRINT: tuple[int, int] = (49, 0)
"""`(det, |sigma|)` of `7_1 # m(7_1)`: `det(7_1)^2 = 49`, and the mirrors cancel."""

QUARANTINED_NAMES: frozenset[str] = frozenset(
    {"14a_18636", "15n_81556", "12n_412", "7_1 # m(7_1)"}
)


def load_quarantine(
    unknotdb_root: str | None = None,
    b5_radius: int = 6,
    include_cube: bool = True,
) -> Quarantine:
    """Build the sec 4.3 set, reading the cube word from UnknotDB when present.

    `b5_radius` covers sec 4.3's "`2^5` masked states ... and their one-change
    neighbourhoods" without needing to know which five positions were marked:
    every subset of at most six of the twenty letters is flipped, which is a
    superset of any five-position mask ball of radius one.
    """
    import json
    from pathlib import Path

    words: set[tuple[Word, int]] = set()
    description: list[str] = []

    b5 = _masked_words(BRITTENHAM_HERMILLER_B5, b5_radius)
    words.update((w, 5) for w in b5)
    description.append(
        f"Brittenham-Hermiller B5 word with every mask of at most {b5_radius} of "
        f"its 20 letters ({len(b5)} representations)"
    )

    if include_cube and unknotdb_root:
        path = (
            Path(unknotdb_root)
            / "outputs/unknotdb-wang-zhang-five-crossing-cube-v0.json"
        )
        if path.exists():
            cube = json.loads(path.read_text())
            base = [int(x) for x in cube["base_word"]]
            marked = [int(x) for x in cube["marked_positions"]]
            strands = int(cube["strands"])
            for mask in range(1 << len(marked)):
                candidate = list(base)
                for bit, position in enumerate(marked):
                    if mask >> bit & 1:
                        candidate[position] = -candidate[position]
                words.add((tuple(candidate), strands))
            description.append(
                f"Wang-Zhang {len(base)}-letter base word on {strands} strands "
                f"and all {1 << len(marked)} mask states of its "
                f"{len(marked)} marked positions"
            )
        else:  # pragma: no cover - depends on the sibling checkout
            description.append(f"Wang-Zhang cube NOT screened: {path} is missing")
    else:
        description.append("Wang-Zhang cube not screened (no unknotdb root given)")

    description.append(
        "names 14a_18636, 15n_81556, 12n_412 and the composite 7_1 # m(7_1)"
    )
    description.append(
        f"fingerprint (det, |sigma|) = {TARGET_FINGERPRINT} of 7_1 # m(7_1)"
    )
    return Quarantine(
        words=frozenset(words),
        names=QUARANTINED_NAMES,
        fingerprints=frozenset({TARGET_FINGERPRINT}),
        composite_summands=frozenset({"7_1", "m(7_1)"}),
        description=tuple(description),
    )
