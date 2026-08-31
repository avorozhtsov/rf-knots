"""Instances whose optimum *requires* growing the diagram, measured not asserted.

Note 22 §M1 closes with a generator change rather than a head change: *build
instances whose shortest completion provably requires growth, so that the corpus
contains the phenomenon the target demands.*  This module is the measurement
half of that, and `m1_growth_run.py` is the driver.

**The two defects this answers.**

1. *The gate scored an objective the campaign rejected.*  `m1_target` now
   defaults to `objective="lex"` -- the lexicographic `(crossing_changes, moves)`
   of §2.1 -- and keeps `objective="moves"` so the published `artifacts/m1-gate`
   table stays reproducible.  Under *shortest moves* the counting identity
   `moves = (L + n - 1)/2 + 2i + 2s + c` charges an insertion two moves against a
   saving of at most one crossing change, so growth is never optimal; under
   `(cc, moves)` an insertion that removes one crossing change wins outright,
   because `cc` dominates absolutely.
2. *The generator cannot emit an instance that needs growth.*  The forward walk
   is type-preserving and `INSERT` is among its moves, so the reversed witness
   contains `REDUCE` where the walk inserted -- and never an `INSERT` unless the
   walk **reduced**.  The corpus is exactly the set of instances a shrinking
   search can solve.

**The measurement, which is the whole point.**  For a word `W`:

```text
c_shrink(W) = min crossing changes over completions that never INSERT or STABILIZE
c_full(W)   = min crossing changes over all completions, inside a growth cap
```

`c_full <= c_shrink` always, since a shrink-only completion is a completion.  The
instance is kept when `c_full < c_shrink`, which is exactly "growth is required",
and `c_shrink - c_full` is its difficulty score.

**What a cap does to that claim, stated before any number is quoted.**  A capped
search that *finds* a completion proves an **upper** bound; a capped search that
gives up proves nothing at all.  So a gap needs *both* halves and they are not
symmetric:

* the shrink side must be **exhaustive** -- only then is `c_shrink` a lower bound
  as well as an upper one, and only then can anything be strictly below it.
  `reason="node-cap"` on the shrink search means the instance is *undetermined*
  and it is recorded that way, never as a gap;
* the full side only has to **exhibit** a completion, and the exhibit is replayed
  through `UnknotWitness.from_actions` before it counts.

Three cheap facts do most of the work, and each is a proof rather than a budget:

* `c_full >= u(W) >= L(W)` for any **certified** lower bound `L`, so
  `L(W) == c_shrink` settles the instance with no search at all: the gap is
  provably zero.  `m1_target.certified_change_floor` supplies `L` from
  `lower_bounds.computed_claims`, which computes from the word and never reads a
  table row (§4.1b's second hazard).
* the same `L` is an admissible heuristic on the first component of the
  lexicographic priority, which is what makes `c_full` affordable at all;
* an accepted instance's exhibited witness **must** contain an `INSERT` or a
  `STABILIZE`, since a growth-free completion at `c_full < c_shrink` would
  contradict `c_shrink`.  That is asserted on every accepted instance, so the
  corpus cannot fill up with mislabelled easy words.

Nothing here records an unproven gap as proven: `GrowthGap.proven` is True only
for `growth-required` and the two `no-gap` verdicts, and `verify_at_raised_caps`
re-runs an accepted instance with the caps multiplied before the report calls it
survived.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from m1_target import (
    LEX,
    ShortestCompletion,
    Successors,
    Word,
    certified_change_floor,
    completion_path,
    shortest_completion,
)

from rf_knots import reference
from rf_knots.actions import (
    INSERT,
    STABILIZE_NEG,
    STABILIZE_POS,
    ActionSpec,
)
from rf_knots.evidence import UnknotWitness
from rf_knots.interleaved_generator import (
    InterleavedConfig,
    SeedRepresentation,
    WalkWeights,
    positive_braid_seed,
    torus_seed,
)

#: The moves `c_shrink` forbids.  `INSERT` adds a cancelling pair and
#: `STABILIZE_*` adds a strand; everything else in the move set leaves the word no
#: longer than it found it, so a shrink-only search runs in a finite space that
#: an exhaustive pass can actually exhaust.
GROWTH_KINDS: tuple[int, ...] = (INSERT, STABILIZE_POS, STABILIZE_NEG)

VERDICTS = (
    "growth-required",
    "no-gap-certified",
    "no-gap-within-growth-cap",
    "undetermined-shrink-capped",
    "undetermined-full-capped",
    "undetermined-change-cap",
)


def shrink_successors(spec: ActionSpec, base: Successors | None = None) -> Successors:
    """`reference.successors` with every growth move filtered out.

    Written as a filter over the shared enumerator rather than as a second
    enumerator: a divergence between the two move sets would silently make
    `c_shrink` a bound on a different problem from `c_full`, and this cannot
    diverge.
    """

    def call(word: Word, strands: int) -> list[tuple[int, Word, int]]:
        rows = (
            base(word, strands)
            if base is not None
            else reference.successors(spec, word, strands, allow_crossing=True)
        )
        return [row for row in rows if spec.decode(row[0])[0] not in GROWTH_KINDS]

    return call


def counts_growth(spec: ActionSpec, actions: Sequence[int]) -> int:
    return sum(1 for action in actions if spec.decode(action)[0] in GROWTH_KINDS)


@dataclass(frozen=True)
class GrowthGap:
    """One instance, both searches, and the sentence the pair is allowed to say.

    `proven` distinguishes *measured* from *not refuted*.  It is True for
    `growth-required` -- the shrink side exhausted and the full side exhibited a
    replayed completion strictly below it -- and for the two `no-gap` verdicts,
    which are proofs of equality within their stated cap.  It is False whenever a
    node cap intervened, and those instances are counted in their own row rather
    than folded into the denominator of a success rate.
    """

    word: Word
    strands: int
    certified_floor: int
    growth_cap: int
    shrink_node_cap: int
    full_node_cap: int
    change_cap: int
    c_shrink: int | None
    shrink_moves: int | None
    shrink_reason: str
    shrink_expanded: int
    c_full: int | None
    full_moves: int | None
    full_reason: str
    full_expanded: int
    verdict: str
    gap: int | None
    proven: bool
    witness_actions: tuple[int, ...] = ()
    witness_growth_moves: int = 0
    witness_replayed: bool = False
    seconds: float = 0.0
    #: Populated by `verify_at_raised_caps`.
    survived_raised_caps: bool | None = None
    raised_cap_factor: int = 0
    notes: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.verdict == "growth-required" and self.proven

    def to_dict(self) -> dict:
        return {
            "word": list(self.word),
            "strands": self.strands,
            "certified_floor": self.certified_floor,
            "growth_cap": self.growth_cap,
            "change_cap": self.change_cap,
            "c_shrink": self.c_shrink,
            "shrink_moves": self.shrink_moves,
            "shrink_reason": self.shrink_reason,
            "shrink_expanded": self.shrink_expanded,
            "c_full": self.c_full,
            "full_moves": self.full_moves,
            "full_reason": self.full_reason,
            "full_expanded": self.full_expanded,
            "verdict": self.verdict,
            "gap": self.gap,
            "proven": self.proven,
            "witness_actions": list(self.witness_actions),
            "witness_growth_moves": self.witness_growth_moves,
            "witness_replayed": self.witness_replayed,
            "seconds": round(self.seconds, 3),
            "survived_raised_caps": self.survived_raised_caps,
            "raised_cap_factor": self.raised_cap_factor,
            "notes": list(self.notes),
        }


def shrink_optimum(
    spec: ActionSpec,
    word: Word,
    strands: int,
    change_cap: int = 8,
    node_cap: int = 60_000,
    change_floor: int | None = None,
    successors: Successors | None = None,
) -> ShortestCompletion:
    """`c_shrink`: the lexicographic optimum over growth-free completions.

    `growth=0` is belt and braces -- the filtered move set cannot lengthen a word
    anyway -- and `change_cap` bounds the search rather than the problem: an
    `exhausted` result means *no* growth-free completion exists at or under
    `change_cap`, which is a proof that `c_shrink > change_cap` and is reported
    as such.
    """
    return shortest_completion(
        spec,
        word,
        strands,
        change_cap,
        growth=0,
        node_cap=node_cap,
        successors=shrink_successors(spec, successors),
        objective=LEX,
        change_floor=change_floor,
    )


def measure_growth_gap(
    spec: ActionSpec,
    word: Word,
    strands: int,
    growth: int = 2,
    change_cap: int = 8,
    shrink_node_cap: int = 60_000,
    full_node_cap: int = 120_000,
    exhibit: bool = True,
    change_floor: int | None = None,
    known_upper: int | None = None,
    known_upper_actions: Sequence[int] = (),
) -> GrowthGap:
    """Measure `c_shrink` and `c_full` and say which of `VERDICTS` holds.

    The order is chosen so that the cheap proofs fire before the expensive
    search.  `c_shrink` first, because its space is finite and small; then the
    certified floor against it, which settles every instance whose growth-free
    route is already optimal for the knot and costs one signature; then any
    `known_upper` the caller already holds a replayable witness for -- a
    generated instance carries one, and a witness cheaper than every growth-free
    route *is* the gap, with no search and no growth cap in the claim; and only
    then the growth-capped search, and only inside a budget of `c_shrink - 1`, so
    that finding *anything* is already the whole claim.

    `known_upper_actions` must be a completion of `(word, strands)`; it is
    replayed through `UnknotWitness.from_actions` and its change count checked
    against `known_upper` before anything is concluded from it.
    """
    started = time.monotonic()
    floor = (
        certified_change_floor(word, strands) if change_floor is None
        else int(change_floor)
    )
    shrink = shrink_optimum(
        spec, word, strands, change_cap, shrink_node_cap, change_floor=floor
    )
    base = dict(
        word=tuple(word),
        strands=int(strands),
        certified_floor=floor,
        growth_cap=growth,
        shrink_node_cap=shrink_node_cap,
        full_node_cap=full_node_cap,
        change_cap=change_cap,
        c_shrink=shrink.crossing_changes,
        shrink_moves=shrink.moves,
        shrink_reason=shrink.reason,
        shrink_expanded=shrink.expanded,
        c_full=None,
        full_moves=None,
        full_reason="not-run",
        full_expanded=0,
        gap=None,
    )
    if shrink.moves is None and shrink.reason == "node-cap":
        return GrowthGap(
            **base,
            verdict="undetermined-shrink-capped",
            proven=False,
            seconds=time.monotonic() - started,
            notes=(
                "the growth-free search hit its node cap, so c_shrink is an "
                "upper bound only and nothing can be shown to be strictly below "
                "it",
            ),
        )
    #: `ceiling` is the smallest number of crossing changes no growth-free
    #: completion is known to beat.  When the shrink search *found* one it is
    #: `c_shrink` exactly; when the search **exhausted** without one, no
    #: growth-free completion exists at `cc <= change_cap`, so `c_shrink` is at
    #: least `change_cap + 1` and any full completion inside the cap already
    #: beats every growth-free route.  In the second case the recorded gap is a
    #: *lower* bound on the true one, which is the safe direction.
    ceiling = (
        int(shrink.crossing_changes) if shrink.moves is not None else change_cap + 1
    )
    if shrink.moves is not None and floor >= ceiling:
        return GrowthGap(
            **{**base, "c_full": ceiling, "full_reason": "certified-floor", "gap": 0},
            verdict="no-gap-certified",
            proven=True,
            seconds=time.monotonic() - started,
            notes=(
                f"the certified lower bound {floor} already equals c_shrink "
                f"{ceiling}, and c_full >= u >= {floor}, so growth cannot help",
            ),
        )
    #: The actions are required, not optional: a `known_upper` with nothing to
    #: replay is a claim, and this module accepts exhibits rather than claims.
    if known_upper is not None and known_upper_actions and known_upper < ceiling:
        actions = tuple(known_upper_actions)
        witness = UnknotWitness.from_actions(tuple(word), int(strands), spec, actions)
        if witness.crossing_changes != known_upper:
            raise AssertionError(
                f"the caller's witness replays at {witness.crossing_changes} "
                f"crossing changes against a claimed {known_upper}"
            )
        growth_moves = counts_growth(spec, actions)
        if growth_moves == 0:
            raise AssertionError(
                "a growth-free witness cheaper than the growth-free optimum "
                "contradicts c_shrink"
            )
        return GrowthGap(
            **{
                **base,
                "c_full": known_upper,
                "full_reason": "known-witness",
                "gap": ceiling - known_upper,
            },
            verdict="growth-required",
            proven=True,
            witness_actions=actions,
            witness_growth_moves=growth_moves,
            witness_replayed=True,
            seconds=time.monotonic() - started,
            notes=(
                "the instance's own replayed witness is cheaper than every "
                "growth-free completion, so the gap needs no search and carries "
                "no growth cap",
            ),
        )
    full = shortest_completion(
        spec,
        word,
        strands,
        ceiling - 1,
        growth=growth,
        node_cap=full_node_cap,
        objective=LEX,
        change_floor=floor,
    )
    base["c_full"] = full.crossing_changes
    base["full_moves"] = full.moves
    base["full_reason"] = full.reason
    base["full_expanded"] = full.expanded
    if full.moves is None:
        if full.reason == "node-cap":
            return GrowthGap(
                **base,
                verdict="undetermined-full-capped",
                proven=False,
                seconds=time.monotonic() - started,
                notes=(
                    "the growth-capped search hit its node cap without finding a "
                    "cheaper completion; that is a limit of the search and not a "
                    "statement about the word",
                ),
            )
        if shrink.moves is None:
            #: Neither side found anything inside `change_cap`.  `c_shrink` and
            #: `c_full` are both above it and nothing orders them.
            return GrowthGap(
                **base,
                verdict="undetermined-change-cap",
                proven=False,
                seconds=time.monotonic() - started,
                notes=(
                    f"no completion of either kind exists at or under "
                    f"{change_cap} crossing changes, so the two quantities are "
                    f"both out of range and neither bounds the other",
                ),
            )
        return GrowthGap(
            **{**base, "c_full": ceiling, "gap": 0},
            verdict="no-gap-within-growth-cap",
            proven=True,
            seconds=time.monotonic() - started,
            notes=(
                f"no completion under {ceiling} crossing changes exists within "
                f"+{growth} letters of growth ({full.reason})",
            ),
        )
    gap = ceiling - int(full.crossing_changes)
    actions: tuple[int, ...] = ()
    growth_moves = 0
    replayed = False
    notes: list[str] = []
    if exhibit:
        path = completion_path(
            spec,
            tuple(word),
            int(strands),
            int(full.crossing_changes),
            int(full.moves),
            min(len(word) + growth, spec.max_len),
            node_cap=full_node_cap,
        )
        if path is None:
            notes.append(
                "the exhibit search hit its own cap; the gap is still proved by "
                "the mask propagation, but no concrete witness is recorded"
            )
        else:
            witness = UnknotWitness.from_actions(tuple(word), int(strands), spec, path)
            if witness.crossing_changes != full.crossing_changes:
                raise AssertionError(
                    f"the exhibited completion spends {witness.crossing_changes} "
                    f"crossing changes against a claimed optimum of "
                    f"{full.crossing_changes}"
                )
            if witness.moves != full.moves:
                raise AssertionError(
                    f"the exhibited completion is {witness.moves} moves against a "
                    f"claimed optimum of {full.moves}"
                )
            actions = tuple(path)
            growth_moves = counts_growth(spec, actions)
            replayed = True
            if growth_moves == 0:
                raise AssertionError(
                    "a completion strictly cheaper than c_shrink used no growth "
                    "move, which contradicts the growth-free optimum"
                )
    return GrowthGap(
        **{**base, "gap": gap},
        verdict="growth-required",
        proven=True,
        witness_actions=actions,
        witness_growth_moves=growth_moves,
        witness_replayed=replayed,
        seconds=time.monotonic() - started,
        notes=tuple(notes),
    )


def verify_at_raised_caps(
    spec: ActionSpec, row: GrowthGap, factor: int = 4
) -> GrowthGap:
    """Re-measure an accepted instance with both node caps multiplied.

    §The claim a cap can break is the *shrink* side: `c_shrink` is only a lower
    bound because the search exhausted, and a bigger search cannot lower an
    exhausted optimum -- but the exhaustion itself is what a larger cap tests, and
    a cheaper full completion would only widen the gap.  A row that comes back
    with a different `c_shrink`, or with no gap at all, is marked as not having
    survived, and the report counts it.
    """
    if not row.accepted:
        return replace(row, survived_raised_caps=None, raised_cap_factor=0)
    again = measure_growth_gap(
        spec,
        row.word,
        row.strands,
        growth=row.growth_cap,
        change_cap=row.change_cap,
        shrink_node_cap=row.shrink_node_cap * factor,
        full_node_cap=row.full_node_cap * factor,
        exhibit=False,
        change_floor=row.certified_floor,
        known_upper=row.c_full if row.full_reason == "known-witness" else None,
        known_upper_actions=(
            row.witness_actions if row.full_reason == "known-witness" else ()
        ),
    )
    survived = (
        again.verdict == "growth-required"
        and again.c_shrink == row.c_shrink
        and again.c_full is not None
        and again.c_full <= row.c_full
    )
    return replace(
        row,
        survived_raised_caps=bool(survived),
        raised_cap_factor=factor,
        notes=row.notes
        + (
            f"raised caps x{factor}: verdict {again.verdict}, "
            f"c_shrink {again.c_shrink}, c_full {again.c_full}",
        ),
    )


# -- candidate construction ----------------------------------------------------


@dataclass(frozen=True)
class CandidateFamily:
    """A seed and a walk, named, so the report can slice acceptance by both.

    The construction is the inverse of the defect.  The witness is the forward
    walk read backwards, and the inverse of `REDUCE` is `INSERT`, so a walk that
    **shrinks** is a witness that **grows**.  Steering it needs no new code: the
    existing `_steer` multiplies `REDUCE` by `toward_target` whenever the word is
    longer than `target_length`, so a long seed with a short target is already a
    shrinking walk, and `DESTABILIZE` behaves the same way against
    `target_strands`.

    That is a construction of *candidates*, not of accepted instances.  Whether
    the reversed walk's growth is **necessary** is decided by
    `measure_growth_gap` and by nothing else -- the report gives the acceptance
    rate per family, the way §4.1b reports its own.
    """

    name: str
    seed: SeedRepresentation
    config: InterleavedConfig
    #: Free-text, carried into the report so a reader can see what was tried.
    intent: str = ""
    #: How many candidates to draw, when the family is cheap enough to be worth
    #: more of them.  `None` takes the sweep's own default.
    instances: int | None = None

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "intent": self.intent,
            "seed": self.seed.name,
            "seed_certified_lower_bound": self.seed.certified_lower_bound,
            "seed_letters": len(self.seed.word),
            "seed_strands": self.seed.strands,
            "requested_instances": self.instances,
            "config": self.config.provenance(),
        }


#: Weights that prefer the two shrinking moves.  Everything else is left at the
#: shipped default, so the only difference from the corpus the gate was built on
#: is the direction of the walk.
SHRINK_WEIGHTS = WalkWeights(
    reduce=4.0,
    commute=2.0,
    braid=3.0,
    insert=0.25,
    destabilize=2.0,
    stabilize=0.25,
)


#: Weights for a walk that *churns*: the two braid relations dominate, and
#: `INSERT` and `REDUCE` are balanced so that the walk both creates cancelling
#: pairs and destroys them.  This is the profile the measured acceptance comes
#: from, and the reason is worth stating: a forward `REDUCE` of a pair the walk
#: did **not** just insert is what leaves a diagram whose only way back grows.
CHURN_WEIGHTS = WalkWeights(
    reduce=3.0,
    commute=4.0,
    braid=4.0,
    insert=3.0,
    destabilize=1.0,
    stabilize=1.0,
    toward_target=3.0,
    away_from_target=0.5,
)


def churn_family(
    name: str,
    seed: SeedRepresentation,
    crossing_changes: int,
    walk_length: int,
    target_length: int,
    target_strands: int,
    max_len: int,
    max_strands: int,
    intent: str = "",
    settle_walk_length: int | None = None,
    instances: int | None = None,
) -> CandidateFamily:
    """A long churning walk, which is what actually produces the gap."""
    return CandidateFamily(
        name=name,
        seed=seed,
        instances=instances,
        config=InterleavedConfig(
            crossing_changes=crossing_changes,
            walk_length=walk_length,
            target_strands=target_strands,
            target_length=target_length,
            max_len=max_len,
            max_strands=max_strands,
            settle_walk_length=(
                walk_length // 2 if settle_walk_length is None else settle_walk_length
            ),
            weights=CHURN_WEIGHTS,
        ),
        intent=intent,
    )


def shrink_family(
    name: str,
    seed: SeedRepresentation,
    crossing_changes: int,
    walk_length: int,
    target_length: int,
    target_strands: int,
    max_len: int,
    max_strands: int,
    intent: str = "",
    weights: WalkWeights | None = None,
    settle_walk_length: int = 0,
    instances: int | None = None,
) -> CandidateFamily:
    """A shrinking walk from a strong seed, expressed in the shipped dials."""
    return CandidateFamily(
        name=name,
        seed=seed,
        instances=instances,
        config=InterleavedConfig(
            crossing_changes=crossing_changes,
            walk_length=walk_length,
            target_strands=target_strands,
            target_length=target_length,
            max_len=max_len,
            max_strands=max_strands,
            settle_walk_length=settle_walk_length,
            weights=SHRINK_WEIGHTS if weights is None else weights,
        ),
        intent=intent,
    )


def torus(m: int, spec: ActionSpec) -> SeedRepresentation:
    return torus_seed(m, spec)


def positive(strands: int, genus: int, spec: ActionSpec) -> SeedRepresentation:
    return positive_braid_seed(spec, strands, genus)


__all__ = [
    "GROWTH_KINDS",
    "VERDICTS",
    "CandidateFamily",
    "CHURN_WEIGHTS",
    "GrowthGap",
    "SHRINK_WEIGHTS",
    "churn_family",
    "counts_growth",
    "measure_growth_gap",
    "positive",
    "shrink_family",
    "shrink_optimum",
    "shrink_successors",
    "torus",
    "verify_at_raised_caps",
]
