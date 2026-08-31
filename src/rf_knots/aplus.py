"""`A_plus(s)`: the sites at a state that are *demonstrated* to finish in budget.

M1's gate in `research/22-scalable-rl-million-representations-improved.md` is
"recall@10 of a witness-realising site", and until this module existed its
denominator was one site: the site the generated witness happened to take.
Note 21 §5.1's rule -- **absent is unexplored, never negative** -- then makes the
figure a lower bound of unknown slack, and a gate whose error bar is unknown
cannot be passed or failed honestly.

This module builds the denominator.  A site enters `A_plus(s)` only when a
completion from it has been *exhibited and replayed*; nothing here ever writes
down that a site fails.  The three sources are the three §M1 names, cheapest
first, and each carries its own kind of proof:

* `witness` -- the site the instance's own witness took.  The base case.
* `pool` -- another verified witness passing through the *same* state took a
  different action.  The two words are compared letter for letter before the
  alternative is accepted, which is the "confirm the walks actually reconverge
  on that word rather than drifting" assertion; the drift is what makes this
  method nearly useless at scale, and `report.md` records the measurement.
* `rewalk` -- a randomised walk out of `s` under a different RNG stream that
  **reconverges onto a later state of the same witness** within the budget the
  witness had left.  The detour is replayed and its endpoint compared to the
  witness state it claims to have rejoined; the suffix is then inherited from a
  witness that is already verified, so the composition is verified end to end
  without replaying it end to end.
* `commutation` -- a later action of the witness bubbled to the front through
  adjacent swaps, every swap checked **by application**: apply `a` then `b`,
  apply `b` then `a`, compare the two `BraidState`s.  Exact, no search, and the
  completion has exactly the same cost because it is the same multiset of moves.
* `bruteforce` -- every legal site tried, each with a bounded best-first search
  for a completion inside the remaining budget.  Expensive, run on a sample, and
  its only job is to say how much the cheap methods miss.

`bruteforce` is bounded, so it too under-reports: a site it did not complete is
`undetermined`, not bad.  Every count this module emits is therefore a **lower
bound on `|A_plus|`**, and the names say so -- `sites_demonstrated_good` and
`sites_undetermined`, never `negative`.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
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
from rf_knots.evidence import BraidState, UnknotWitness
from rf_knots.interleaved_generator import InterleavedInstance

Word = tuple[int, ...]
StateKey = tuple[Word, int, bool]

#: Every kind a site can carry, in the order the reports print them.
SITE_KINDS: tuple[int, ...] = (
    REDUCE,
    COMMUTE,
    BRAID,
    INSERT,
    DESTABILIZE,
    STABILIZE_POS,
    STABILIZE_NEG,
    CROSSING_CHANGE,
)

METHODS: tuple[str, ...] = ("witness", "pool", "rewalk", "commutation", "bruteforce")


def state_key(state: BraidState) -> StateKey:
    return (state.word, state.strands, state.cyclic_band_generators)


# -- budgets -------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """What is left at a state: moves and crossing changes.

    The crossing-change half is the one the `u <= k` claim is about.  The move
    half is here because a completion has to terminate, and because a "site that
    works" which needs four times the moves the witness needed is a different
    claim from the one M1's gate makes.  Both are recorded per site so that a
    reader can re-filter without re-running anything.
    """

    moves: int
    crossing_changes: int

    def admits(self, moves: int, crossing_changes: int) -> bool:
        return moves <= self.moves and crossing_changes <= self.crossing_changes

    def after(self, is_crossing_change: bool) -> Budget:
        return Budget(self.moves - 1, self.crossing_changes - int(is_crossing_change))

    def to_dict(self) -> dict:
        return {"moves": self.moves, "crossing_changes": self.crossing_changes}


# -- the spine of one witness --------------------------------------------------


@dataclass(frozen=True)
class WitnessSite:
    """One state on a verified witness together with what the witness did there."""

    depth: int
    state: BraidState
    action: int
    remaining_moves: int
    remaining_crossing_changes: int
    #: True once the walk has been undone and the states belong to the seed's own
    #: program, which every instance from that seed shares.  Pooling is trivially
    #: successful there and the reports keep the two regions apart because of it.
    in_seed_program: bool

    @property
    def budget(self) -> Budget:
        return Budget(self.remaining_moves, self.remaining_crossing_changes)


@dataclass(frozen=True)
class WitnessSpine:
    """A replay-complete witness re-indexed as `(state, action, budget)` triples."""

    spec: ActionSpec
    states: tuple[BraidState, ...]
    actions: tuple[int, ...]
    sites: tuple[WitnessSite, ...]
    walk_moves: int
    instance_id: str
    seed_name: str
    seed_certified_lower_bound: int
    u_lower_inherited: int
    d_upper: int
    certified_upper_bound: int

    def __len__(self) -> int:
        return len(self.sites)


def spine_from_witness(
    witness: UnknotWitness,
    walk_moves: int | None = None,
    instance_id: str | None = None,
    seed_name: str = "",
    seed_certified_lower_bound: int = 0,
    u_lower_inherited: int = 0,
    d_upper: int = 0,
) -> WitnessSpine:
    """Re-index any verified witness, whoever produced it.

    `witness_spine` is the generator's route into this; `unknotdb`'s replayed
    paths and any future search's output take the same one, which is the point of
    keeping the spine a function of the witness rather than of the instance.
    """
    spec = ActionSpec(
        witness.max_len, witness.max_strands, witness.cyclic_band_generators
    )
    states = (witness.start,) + tuple(step.after for step in witness.steps)
    actions = tuple(step.action.to_flat(spec) for step in witness.steps)
    crossing = tuple(
        step.action.kind == KIND_NAMES[CROSSING_CHANGE] for step in witness.steps
    )
    total = len(actions)
    suffix = [0] * (total + 1)
    for index in range(total - 1, -1, -1):
        suffix[index] = suffix[index + 1] + int(crossing[index])
    span = total if walk_moves is None else walk_moves
    sites = tuple(
        WitnessSite(
            depth=index,
            state=states[index],
            action=actions[index],
            remaining_moves=total - index,
            remaining_crossing_changes=suffix[index],
            in_seed_program=index >= span,
        )
        for index in range(total)
    )
    return WitnessSpine(
        spec=spec,
        states=states,
        actions=actions,
        sites=sites,
        walk_moves=span,
        instance_id=instance_id or witness.instance_id,
        seed_name=seed_name,
        seed_certified_lower_bound=seed_certified_lower_bound,
        u_lower_inherited=u_lower_inherited,
        d_upper=d_upper,
        certified_upper_bound=int(witness.crossing_changes),
    )


def witness_spine(instance: InterleavedInstance) -> WitnessSpine:
    """Re-index a generated instance's witness for per-site supervision.

    Raises on a `catalogue-composed` instance: without a program there is no
    witness to walk, and inventing budgets for one would be exactly the kind of
    uncertified claim §4.1b forbids.
    """
    witness = instance.witness
    if witness is None:
        raise ValueError(
            f"instance {instance.instance_id} is catalogue-composed: it carries "
            "no replayable witness, so it has no spine"
        )
    return spine_from_witness(
        witness,
        walk_moves=len(instance.reverse_actions),
        instance_id=instance.instance_id,
        seed_name=instance.seed_representation.name,
        seed_certified_lower_bound=(
            instance.seed_representation.certified_lower_bound
        ),
        u_lower_inherited=instance.u_lower_inherited,
        d_upper=instance.d_upper,
    )


# -- memoised successors -------------------------------------------------------


class _Successors:
    """Successor lists memoised for the length of one instance's analysis."""

    def __init__(self, spec: ActionSpec, cap: int = 40_000) -> None:
        self.spec = spec
        self.cap = cap
        self._cache: dict[tuple[Word, int], list[tuple[int, Word, int]]] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, word: Word, strands: int) -> list[tuple[int, Word, int]]:
        key = (word, strands)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        value = reference.successors(self.spec, word, strands, allow_crossing=True)
        if len(self._cache) < self.cap:
            self._cache[key] = value
        return value


# -- the set itself ------------------------------------------------------------


@dataclass(frozen=True)
class KnownGoodSite:
    """One site with an exhibited completion.  There is no negative counterpart."""

    action: int
    kind: int
    method: str
    completion_moves: int
    completion_crossing_changes: int
    replay_verified: bool

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "kind": KIND_NAMES[self.kind],
            "method": self.method,
            "completion_moves": self.completion_moves,
            "completion_crossing_changes": self.completion_crossing_changes,
            "replay_verified": self.replay_verified,
        }


@dataclass
class KnownGoodSites:
    """`A_plus(s)` for one state, plus the bookkeeping a gate needs to be honest.

    `sites` holds only what was demonstrated.  Everything legal and absent is
    `undetermined`: no search was run for it, or the search that was run did not
    finish inside its cap.  Neither is evidence that it fails.
    """

    state: BraidState
    depth: int
    budget: Budget
    legal_actions: tuple[int, ...]
    sites: dict[int, KnownGoodSite] = field(default_factory=dict)
    #: Methods that were actually run at this state.  A method that never ran
    #: cannot be blamed for a site it did not find.
    methods_run: tuple[str, ...] = ()
    in_seed_program: bool = False
    #: Everything each method demonstrated, independent of which method got there
    #: first.  `actions_by_method` attributes each site to one method and is
    #: therefore order-dependent; this is not, and it is what the report slices.
    demonstrated_by_method: dict[str, set[int]] = field(default_factory=dict)

    def offer(self, site: KnownGoodSite) -> bool:
        """Record a demonstration, keeping the cheapest completion per site."""
        current = self.sites.get(site.action)
        if current is None:
            self.sites[site.action] = site
            return True
        better = (site.completion_crossing_changes, site.completion_moves) < (
            current.completion_crossing_changes,
            current.completion_moves,
        )
        if better:
            self.sites[site.action] = site
        return False

    @property
    def size(self) -> int:
        return len(self.sites)

    @property
    def sites_undetermined(self) -> int:
        """Legal sites for which no completion has been exhibited *yet*."""
        return len(self.legal_actions) - len(self.sites)

    def actions_by_method(self) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {name: set() for name in METHODS}
        for action, site in self.sites.items():
            out[site.method].add(action)
        return out

    def kind_histogram(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for site in self.sites.values():
            name = KIND_NAMES[site.kind]
            out[name] = out.get(name, 0) + 1
        return out


# -- method 0: the witness's own site ------------------------------------------


def witness_known_good(
    spine: WitnessSpine, index: int, successors: _Successors | None = None
) -> KnownGoodSites:
    """The base case: one site, the one the witness took.  The naive denominator."""
    site = spine.sites[index]
    succ = successors if successors is not None else _Successors(spine.spec)
    legal = tuple(
        action for action, _word, _strands in succ(site.state.word, site.state.strands)
    )
    out = KnownGoodSites(
        state=site.state,
        depth=index,
        budget=site.budget,
        legal_actions=legal,
        methods_run=("witness",),
        in_seed_program=site.in_seed_program,
    )
    kind, _p, _g, _s = spine.spec.decode(site.action)
    out.offer(
        KnownGoodSite(
            action=site.action,
            kind=kind,
            method="witness",
            completion_moves=site.remaining_moves,
            completion_crossing_changes=site.remaining_crossing_changes,
            replay_verified=True,
        )
    )
    return out


# -- method 2: commutation closure, checked by application ---------------------


def commutes(
    spec: ActionSpec, word: Word, strands: int, first: int, second: int
) -> bool:
    """`first then second` and `second then first` land on the same state.

    Nothing here is derived from a theorem about disjoint supports.  Both orders
    are applied and the two results compared, so a `True` is a fact about this
    pair at this word and a `False` is only ever "not shown", never "does not".
    """
    if not reference.is_legal(spec, word, strands, first, True):
        return False
    if not reference.is_legal(spec, word, strands, second, True):
        return False
    word_a, strands_a = reference.apply(spec, word, strands, first)
    if not reference.is_legal(spec, word_a, strands_a, second, True):
        return False
    word_ab, strands_ab = reference.apply(spec, word_a, strands_a, second)
    word_b, strands_b = reference.apply(spec, word, strands, second)
    if not reference.is_legal(spec, word_b, strands_b, first, True):
        return False
    word_ba, strands_ba = reference.apply(spec, word_b, strands_b, first)
    return (word_ab, strands_ab) == (word_ba, strands_ba)


def _bubble_to_front(
    spine: WitnessSpine, index: int, span: int
) -> tuple[int, ...] | None:
    """Move `actions[index + span]` to position `index` by adjacent swaps.

    Every swap is checked with `commutes`, and the state each swap is checked at
    is the *original* witness state, because the swap at offset `t` only ever
    disturbs positions `t` and above -- positions below it are still the witness's
    own prefix.  That is what keeps this linear in `span` rather than quadratic.
    """
    block = list(spine.actions[index : index + span + 1])
    for offset in range(span, 0, -1):
        state = spine.states[index + offset - 1]
        if not commutes(
            spine.spec, state.word, state.strands, block[offset - 1], block[offset]
        ):
            return None
        block[offset - 1], block[offset] = block[offset], block[offset - 1]
    return tuple(block)


def commutation_known_good(
    spine: WitnessSpine, index: int, window: int = 8
) -> list[KnownGoodSite]:
    """Alternative first actions obtained by reordering the witness itself.

    The completion is the same multiset of moves in a different order, so it
    costs exactly what the witness cost and is inside the budget by construction.
    The reordered block is replayed and its endpoint compared to the witness
    state it must land on; the suffix past that point is untouched and already
    verified, so the composition is verified without being replayed again.
    """
    spec = spine.spec
    limit = min(window, len(spine.actions) - index - 1)
    seen: set[int] = {spine.actions[index]}
    out: list[KnownGoodSite] = []
    site = spine.sites[index]
    for span in range(1, limit + 1):
        block = _bubble_to_front(spine, index, span)
        if block is None:
            continue
        candidate = block[0]
        if candidate in seen:
            continue
        start = spine.states[index]
        end = _replay(spec, start, block)
        if end is None or end != spine.states[index + span + 1]:  # pragma: no cover
            raise AssertionError(
                "a commutation-closed block did not rejoin the witness; the "
                "by-application check and the replay disagree"
            )
        seen.add(candidate)
        kind, _p, _g, _s = spec.decode(candidate)
        out.append(
            KnownGoodSite(
                action=candidate,
                kind=kind,
                method="commutation",
                completion_moves=site.remaining_moves,
                completion_crossing_changes=site.remaining_crossing_changes,
                replay_verified=True,
            )
        )
    return out


def _replay(
    spec: ActionSpec, start: BraidState, actions: Sequence[int]
) -> BraidState | None:
    """Apply `actions` from `start`, building a `BraidState` at every step.

    `None` when any step is illegal or any intermediate closure stops being a
    knot.  Used wherever a demonstration has to be a replay rather than an
    argument.
    """
    word, strands = start.word, start.strands
    cyclic = start.cyclic_band_generators
    for action in actions:
        if not reference.is_legal(spec, word, strands, int(action), True):
            return None
        word, strands = reference.apply(spec, word, strands, int(action))
        try:
            BraidState(word, strands, cyclic)
        except ValueError:  # pragma: no cover - the move set preserves knots
            return None
    return BraidState(word, strands, cyclic)


# -- method 1a: pooling several witnesses over the same state ------------------


@dataclass
class WitnessPool:
    """Every `(state, action)` any added witness took, with its completion cost.

    The literal reading of §M1's first bullet: emit several witnesses and let
    them vote.  It only pays where two witnesses actually meet, so
    `collision_report` is part of the output rather than an afterthought.
    """

    spines: list[WitnessSpine] = field(default_factory=list)
    _by_state: dict[StateKey, dict[int, tuple[int, int, int, int]]] = field(
        default_factory=dict
    )
    _words: dict[StateKey, int] = field(default_factory=dict)

    def add(self, spine: WitnessSpine) -> None:
        reference_index = len(self.spines)
        self.spines.append(spine)
        for site in spine.sites:
            key = state_key(site.state)
            bucket = self._by_state.setdefault(key, {})
            cost = (
                site.remaining_moves,
                site.remaining_crossing_changes,
                reference_index,
                site.depth,
            )
            current = bucket.get(site.action)
            if current is None or cost[:2] < current[:2]:
                bucket[site.action] = cost
        root = state_key(spine.states[0])
        self._words[root] = self._words.get(root, 0) + 1

    def emitted_word_collisions(self) -> dict:
        """How often two seeds really did land on the same emitted word."""
        counts = list(self._words.values())
        return {
            "emitted_words": sum(counts),
            "distinct_emitted_words": len(counts),
            "words_seen_more_than_once": sum(1 for c in counts if c > 1),
            "surplus_witnesses_on_shared_words": sum(c - 1 for c in counts if c > 1),
        }

    def lookup(self, spine: WitnessSpine, index: int) -> list[KnownGoodSite]:
        """Alternatives another witness demonstrated at *this exact state*."""
        site = spine.sites[index]
        key = state_key(site.state)
        bucket = self._by_state.get(key)
        if not bucket:
            return []
        out: list[KnownGoodSite] = []
        for action, (moves, changes, spine_index, depth) in bucket.items():
            if action == site.action:
                continue
            if not site.budget.admits(moves, changes):
                continue
            other = self.spines[spine_index]
            # The reconvergence assertion: the other witness must be standing on
            # the same word, letter for letter, not merely on the same knot.
            if other.states[depth] != site.state:  # pragma: no cover - key is the word
                raise AssertionError(
                    "pool key collision without state equality: the walks drifted"
                )
            end = _replay(spine.spec, site.state, other.actions[depth:])
            if end is None or end.word or end.strands != 1:  # pragma: no cover
                raise AssertionError(
                    "a pooled continuation did not replay to the unknot"
                )
            kind, _p, _g, _s = spine.spec.decode(action)
            out.append(
                KnownGoodSite(
                    action=action,
                    kind=kind,
                    method="pool",
                    completion_moves=moves,
                    completion_crossing_changes=changes,
                    replay_verified=True,
                )
            )
        return out


# -- method 1b: re-walking under a different RNG, with reconvergence -----------

#: Kind weights for the detour walk.  The same philosophy as the generator's own
#: `WalkWeights`: the two braid relations churn without changing length and are
#: what a reconvergent detour needs, while `INSERT` is one kind holding most of
#: the actions and would otherwise be the only thing ever sampled.
DETOUR_WEIGHTS: dict[int, float] = {
    REDUCE: 3.0,
    COMMUTE: 3.0,
    BRAID: 3.0,
    INSERT: 1.0,
    DESTABILIZE: 2.0,
    STABILIZE_POS: 0.5,
    STABILIZE_NEG: 0.5,
    CROSSING_CHANGE: 1.0,
}


def _grouped(
    spec: ActionSpec, moves: Sequence[tuple[int, Word, int]]
) -> dict[int, list[tuple[int, Word, int]]]:
    out: dict[int, list[tuple[int, Word, int]]] = {}
    for entry in moves:
        out.setdefault(spec.decode(entry[0])[0], []).append(entry)
    return out


def _pick(
    rng: np.random.Generator,
    grouped: dict[int, list[tuple[int, Word, int]]],
    weights: dict[int, float] | None,
) -> tuple[int, Word, int] | None:
    kinds = sorted(grouped)
    if not kinds:
        return None
    if weights is None:
        scores = [1.0] * len(kinds)
    else:
        scores = [weights.get(kind, 1.0) for kind in kinds]
    total = float(sum(scores))
    if total <= 0.0:  # pragma: no cover - the table has no zeros
        scores = [1.0] * len(kinds)
        total = float(len(kinds))
    threshold = float(rng.random()) * total
    running = 0.0
    chosen = kinds[-1]
    for kind, score in zip(kinds, scores, strict=True):
        running += score
        if running >= threshold:
            chosen = kind
            break
    block = grouped[chosen]
    return block[int(rng.integers(len(block)))]


def rewalk_known_good(
    spine: WitnessSpine,
    index: int,
    rng: np.random.Generator,
    trials: int = 64,
    max_detour: int = 4,
    successors: _Successors | None = None,
    growth: int = 4,
) -> list[KnownGoodSite]:
    """Randomised detours out of `states[index]` that rejoin the same witness.

    A detour is accepted only when it lands **exactly** on a later state of this
    witness -- word, strand count and alphabet -- within the moves and crossing
    changes the witness had left at `index`.  That endpoint comparison is the
    reconvergence assertion: a walk that drifted onto a different word of the
    same knot is rejected, because the suffix it would inherit belongs to the
    other word.

    Cheap in the way §M1 means: no completion is ever searched for, only a
    rejoining, and the rest of the proof is a witness that already replays.
    """
    spec = spine.spec
    site = spine.sites[index]
    succ = successors if successors is not None else _Successors(spec)
    rejoin: dict[StateKey, int] = {}
    for later in range(index + 1, len(spine.states)):
        rejoin.setdefault(state_key(spine.states[later]), later)
    cap = len(site.state.word) + growth
    found: dict[int, KnownGoodSite] = {}
    for _ in range(trials):
        word, strands = site.state.word, site.state.strands
        used_moves = 0
        used_changes = 0
        first: int | None = None
        path: list[int] = []
        for step in range(max_detour + 1):
            grouped = _grouped(spec, succ(word, strands))
            if step == 0:
                # Uniform over *kinds* for the first move, so the 70%-of-the-slots
                # `INSERT` block does not swallow the whole sample.
                choice = _pick(rng, grouped, None)
            else:
                choice = _pick(rng, grouped, DETOUR_WEIGHTS)
            if choice is None:  # pragma: no cover - PASS-only states are terminal
                break
            action, word, strands = choice
            kind = spec.decode(action)[0]
            used_moves += 1
            used_changes += int(kind == CROSSING_CHANGE)
            path.append(action)
            if first is None:
                first = action
            if used_changes > site.remaining_crossing_changes:
                break
            if len(word) > cap:
                break
            landed = rejoin.get((word, strands, site.state.cyclic_band_generators))
            if landed is None:
                continue
            tail_moves = len(spine.actions) - landed
            tail_changes = spine.sites[landed].remaining_crossing_changes if (
                landed < len(spine.sites)
            ) else 0
            total_moves = used_moves + tail_moves
            total_changes = used_changes + tail_changes
            if not site.budget.admits(total_moves, total_changes):
                break
            if first == site.action and len(path) == 1:
                break
            end = _replay(spec, site.state, path)
            if end is None or end != spine.states[landed]:  # pragma: no cover
                raise AssertionError(
                    "a detour claimed to rejoin the witness but did not replay "
                    "onto the state it named"
                )
            kind_first = spec.decode(first)[0]
            candidate = KnownGoodSite(
                action=first,
                kind=kind_first,
                method="rewalk",
                completion_moves=total_moves,
                completion_crossing_changes=total_changes,
                replay_verified=True,
            )
            previous = found.get(first)
            if previous is None or (
                candidate.completion_crossing_changes,
                candidate.completion_moves,
            ) < (previous.completion_crossing_changes, previous.completion_moves):
                found[first] = candidate
            break
    found.pop(site.action, None)
    return list(found.values())


# -- method 3: brute force, for calibration only -------------------------------


def move_lower_bound(word: Word, strands: int) -> int:
    """Moves any completion must spend, from the word and the strand count alone.

    Every letter has to leave, and no move removes more than two of them, so at
    least `ceil(len/2)` moves are removals; and `strands - 1` destabilisations are
    unavoidable because nothing else lowers the strand count.  Both are counts of
    moves the move set forces, so the maximum of the two is admissible and it
    costs one division.
    """
    return max(-(-len(word) // 2), strands - 1)


def bounded_completion(
    spec: ActionSpec,
    word: Word,
    strands: int,
    crossing_change_budget: int,
    move_budget: int,
    node_cap: int = 400,
    growth: int = 3,
    order: str = "length",
) -> tuple[int, ...] | None:
    """Bounded search for a replayable path to the empty one-braid.

    `None` means **not found under this cap**.  It is never a statement that no
    completion exists, and no caller may record it as one.

    `order="length"` follows `research/experiments/beam_rung.py`: the frontier is
    ordered by word length, because on these words length is the signal and the
    certified bound is far too expensive to sit inside the loop.  It finds *a*
    completion quickly and says nothing about how short it is.

    `order="moves"` is A* on the move count with `move_lower_bound` as the
    heuristic, and the goal is tested on pop rather than on generation, so the
    length it returns is the shortest **this search reached** -- an upper bound on
    the true shortest completion, never a claim to be it.
    """
    if crossing_change_budget < 0 or move_budget < 0:
        return None
    if not word and strands == 1:
        return ()
    if order not in ("length", "moves"):
        raise ValueError(f"unknown search order {order!r}")
    cap = len(word) + growth
    counter = 0
    start = (word, strands)

    def priority(candidate: Word, active: int, moves: int) -> int:
        if order == "moves":
            return moves + move_lower_bound(candidate, active)
        return len(candidate)

    heap: list[tuple[int, int, int, int, Word, int, tuple[int, ...]]] = [
        (priority(word, strands, 0), 0, 0, counter, word, strands, ())
    ]
    best: dict[tuple[Word, int], tuple[int, int]] = {start: (0, 0)}
    expanded = 0
    while heap and expanded < node_cap:
        _key, used_moves, used_changes, _tie, current, active, path = heapq.heappop(
            heap
        )
        if not current and active == 1:
            return path
        recorded = best.get((current, active))
        if recorded is not None and recorded < (used_moves, used_changes):
            continue
        expanded += 1
        if used_moves >= move_budget:
            continue
        allow = used_changes < crossing_change_budget
        for action, next_word, next_strands in reference.successors(
            spec, current, active, allow_crossing=allow
        ):
            if len(next_word) > cap:
                continue
            changes = used_changes + int(spec.decode(action)[0] == CROSSING_CHANGE)
            if changes > crossing_change_budget:
                continue
            moves = used_moves + 1
            key = (next_word, next_strands)
            recorded = best.get(key)
            if recorded is not None and recorded <= (moves, changes):
                continue
            best[key] = (moves, changes)
            if not next_word and next_strands == 1 and order == "length":
                return (*path, action)
            if moves >= move_budget:
                continue
            counter += 1
            heapq.heappush(
                heap,
                (
                    priority(next_word, next_strands, moves),
                    moves,
                    changes,
                    counter,
                    next_word,
                    next_strands,
                    (*path, action),
                ),
            )
    return None


@dataclass
class BruteForceReport:
    """What one brute-force calibration of a single state cost and found."""

    legal_actions: int
    demonstrated: int
    undetermined: int
    node_cap: int
    seconds: float


def bruteforce_known_good(
    spine: WitnessSpine,
    index: int,
    node_cap: int = 400,
    growth: int = 3,
    verify: bool = True,
    order: str = "length",
    successors: _Successors | None = None,
) -> tuple[list[KnownGoodSite], BruteForceReport]:
    """Try every legal site and search for a completion inside the budget.

    The expensive method, and the only one that says anything about coverage.
    Sites it does not complete stay `undetermined` -- the cap is the reason, and
    a cap is not a proof.
    """
    import time

    spec = spine.spec
    site = spine.sites[index]
    started = time.monotonic()
    succ = successors if successors is not None else _Successors(spec)
    moves = succ(site.state.word, site.state.strands)
    out: list[KnownGoodSite] = []
    for action, next_word, next_strands in moves:
        kind = spec.decode(action)[0]
        is_change = kind == CROSSING_CHANGE
        remaining = site.budget.after(is_change)
        if remaining.moves < 0 or remaining.crossing_changes < 0:
            continue
        tail = bounded_completion(
            spec,
            next_word,
            next_strands,
            remaining.crossing_changes,
            remaining.moves,
            node_cap=node_cap,
            growth=growth,
            order=order,
        )
        if tail is None:
            continue
        path = (action, *tail)
        if verify:
            witness = UnknotWitness.from_actions(
                site.state.word, site.state.strands, spec, path
            )
            changes = witness.crossing_changes
            length = witness.moves
        else:  # pragma: no cover - the driver always verifies
            changes = sum(spec.decode(a)[0] == CROSSING_CHANGE for a in path)
            length = len(path)
        if not site.budget.admits(length, changes):  # pragma: no cover - defensive
            continue
        out.append(
            KnownGoodSite(
                action=action,
                kind=kind,
                method="bruteforce",
                completion_moves=length,
                completion_crossing_changes=changes,
                replay_verified=verify,
            )
        )
    elapsed = time.monotonic() - started
    return out, BruteForceReport(
        legal_actions=len(moves),
        demonstrated=len(out),
        undetermined=len(moves) - len(out),
        node_cap=node_cap,
        seconds=elapsed,
    )


# -- assembling A_plus ---------------------------------------------------------


def build_known_good(
    spine: WitnessSpine,
    index: int,
    pool: WitnessPool | None = None,
    rng: np.random.Generator | None = None,
    commutation_window: int = 8,
    rewalk_trials: int = 64,
    rewalk_detour: int = 4,
    successors: _Successors | None = None,
    bruteforce_node_cap: int | None = None,
) -> KnownGoodSites:
    """`A_plus(states[index])` from every method the caller switched on."""
    out = witness_known_good(spine, index, successors=successors)
    methods = ["witness"]
    out.demonstrated_by_method["witness"] = {spine.actions[index]}
    batches: list[tuple[str, list[KnownGoodSite]]] = []
    if pool is not None:
        methods.append("pool")
        batches.append(("pool", pool.lookup(spine, index)))
    if commutation_window > 0:
        methods.append("commutation")
        batches.append(
            ("commutation", commutation_known_good(spine, index, commutation_window))
        )
    if rng is not None and rewalk_trials > 0:
        methods.append("rewalk")
        batches.append(
            (
                "rewalk",
                rewalk_known_good(
                    spine,
                    index,
                    rng,
                    trials=rewalk_trials,
                    max_detour=rewalk_detour,
                    successors=successors,
                ),
            )
        )
    if bruteforce_node_cap is not None:
        methods.append("bruteforce")
        sites, _report = bruteforce_known_good(
            spine, index, node_cap=bruteforce_node_cap, successors=successors
        )
        batches.append(("bruteforce", sites))
    for name, sites in batches:
        out.demonstrated_by_method[name] = {site.action for site in sites}
        for site in sites:
            out.offer(site)
    out.methods_run = tuple(methods)
    return out


# -- deliverable 2: the hand-written baselines ---------------------------------

#: Hand-written tier order, fixed *before* any recall was measured.  Length that
#: comes off the word for free ranks above reshaping, reshaping above growth.
KIND_TIER: dict[int, int] = {
    REDUCE: 6,
    DESTABILIZE: 6,
    BRAID: 4,
    COMMUTE: 3,
    INSERT: 1,
    STABILIZE_POS: 0,
    STABILIZE_NEG: 0,
}
CROSSING_CHANGE_TIER_PRODUCTIVE = 5
CROSSING_CHANGE_TIER_FLAT = 2


def free_reduced_drop(word: Word, next_word: Word) -> int:
    """Letters the flip (or any move) removes from the free reduction.

    `reference.free_reduce` is the repository's linear free reduction, not a
    cyclic one; the word is a necklace, so this under-counts a cancellation that
    straddles the seam.  Under-counting is the right way round for a baseline.
    """
    return len(reference.free_reduce(word)) - len(reference.free_reduce(next_word))


def insert_context_score(word: Word, position: int, letter: int) -> int:
    """How much local structure the spliced pair `(letter, -letter)` touches.

    `INSERT` is 70-80% of the legal successors and free-reduced length cannot
    separate them at all: splicing a cancelling pair leaves the free reduction
    of the word exactly as it was, so every `INSERT` scores a drop of zero.
    Something else has to order them, and the cheapest defensible thing is what
    the new letters are *next to*, because a spliced pair that touches nothing
    can only ever be undone:

    * `+2` if a new letter cancels against its outer neighbour -- a `REDUCE` is
      immediately available, which is the slide that moves a crossing along the
      word;
    * `+1` if a new letter is braid-adjacent (`||a| - |b|| == 1`) to its outer
      neighbour -- a `BRAID` may become available once the pair separates;
    * `+1` if the generator already occurs in the word, since a generator the
      word does not use has to be introduced and removed again within the same
      budget.

    O(1) per site, no successor enumeration, and every clause is a statement
    about the move set rather than about knots.
    """
    length = len(word)
    if length == 0:
        return 0
    left = word[position - 1] if position > 0 else word[-1]
    right = word[position % length]
    score = 0
    if left == -letter:
        score += 2
    elif abs(abs(left) - abs(letter)) == 1:
        score += 1
    if right == letter:
        score += 2
    elif abs(abs(right) - abs(letter)) == 1:
        score += 1
    if any(abs(x) == abs(letter) for x in word):
        score += 1
    return score


def _shuffled(actions: Sequence[int], rng: np.random.Generator) -> list[int]:
    order = list(actions)
    rng.shuffle(order)
    return order


def rank_uniform(
    spec: ActionSpec,
    word: Word,
    strands: int,
    moves: Sequence[tuple[int, Word, int]],
    rng: np.random.Generator,
) -> list[int]:
    """The control.  Any ranker that cannot beat this has learned nothing."""
    return _shuffled([action for action, _w, _s in moves], rng)


def rank_kind_prior(
    spec: ActionSpec,
    word: Word,
    strands: int,
    moves: Sequence[tuple[int, Word, int]],
    rng: np.random.Generator,
) -> list[int]:
    """Kind order only, no look at the word.  Isolates how much the kind is worth."""
    scored = []
    for action, _next_word, _next_strands in moves:
        kind = spec.decode(action)[0]
        tier = KIND_TIER.get(kind, CROSSING_CHANGE_TIER_FLAT)
        scored.append((tier, action))
    order = _shuffled([action for _t, action in scored], rng)
    tiers = {action: tier for tier, action in scored}
    order.sort(key=lambda action: -tiers[action])
    return order


def rank_free_reduced_drop(
    spec: ActionSpec,
    word: Word,
    strands: int,
    moves: Sequence[tuple[int, Word, int]],
    rng: np.random.Generator,
) -> list[int]:
    """M1's kill criterion, literally: rank by the drop in free-reduced length."""
    scores: dict[int, int] = {}
    for action, next_word, _next_strands in moves:
        scores[action] = free_reduced_drop(word, next_word)
    order = _shuffled(list(scores), rng)
    order.sort(key=lambda action: -scores[action])
    return order


def rank_growth_aware(
    spec: ActionSpec,
    word: Word,
    strands: int,
    moves: Sequence[tuple[int, Word, int]],
    rng: np.random.Generator,
) -> list[int]:
    """The full hand-written heuristic: tiers, then drop, then `INSERT` context."""
    scores: dict[int, tuple[int, int]] = {}
    for action, next_word, _next_strands in moves:
        kind, position, generator, sign = spec.decode(action)
        drop = free_reduced_drop(word, next_word)
        if kind == CROSSING_CHANGE:
            tier = (
                CROSSING_CHANGE_TIER_PRODUCTIVE
                if drop > 0
                else CROSSING_CHANGE_TIER_FLAT
            )
            secondary = drop
        elif kind == INSERT:
            tier = KIND_TIER[INSERT]
            secondary = insert_context_score(word, position, sign * generator)
        else:
            tier = KIND_TIER.get(kind, 0)
            secondary = drop
        scores[action] = (tier, secondary)
    order = _shuffled(list(scores), rng)
    order.sort(key=lambda action: (-scores[action][0], -scores[action][1]))
    return order


def rank_by_kind_order(kind_order: Sequence[int]):
    """A ranker for a *given* kind order, for the fitted upper bound in the report."""
    rank = {kind: -position for position, kind in enumerate(kind_order)}

    def ranker(
        spec: ActionSpec,
        word: Word,
        strands: int,
        moves: Sequence[tuple[int, Word, int]],
        rng: np.random.Generator,
    ) -> list[int]:
        order = _shuffled([action for action, _w, _s in moves], rng)
        order.sort(key=lambda action: -rank.get(spec.decode(action)[0], -99))
        return order

    return ranker


RANKERS = {
    "uniform_random": rank_uniform,
    "kind_prior": rank_kind_prior,
    "free_reduced_drop": rank_free_reduced_drop,
    "growth_aware": rank_growth_aware,
}


def hit_at_k(order: Sequence[int], known_good: Iterable[int], k: int) -> bool:
    """Did the ranker put *any* demonstrated site in the top `k`?"""
    good = set(known_good)
    if not good:
        return False
    return any(action in good for action in order[:k])


def random_hit_probability(legal: int, known_good: int, k: int) -> float:
    """Chance a uniform ranker hits, in closed form.  The honest chance line."""
    if known_good <= 0 or legal <= 0:
        return 0.0
    if k >= legal:
        return 1.0
    miss = legal - known_good
    if miss < k:
        return 1.0
    return 1.0 - math.comb(miss, k) / math.comb(legal, k)


__all__ = [
    "Budget",
    "BruteForceReport",
    "KIND_TIER",
    "KnownGoodSite",
    "KnownGoodSites",
    "METHODS",
    "RANKERS",
    "SITE_KINDS",
    "WitnessPool",
    "WitnessSite",
    "WitnessSpine",
    "bounded_completion",
    "move_lower_bound",
    "bruteforce_known_good",
    "build_known_good",
    "commutation_known_good",
    "commutes",
    "free_reduced_drop",
    "hit_at_k",
    "insert_context_score",
    "random_hit_probability",
    "rank_by_kind_order",
    "rank_free_reduced_drop",
    "rank_growth_aware",
    "rank_kind_prior",
    "rank_uniform",
    "rewalk_known_good",
    "state_key",
    "witness_known_good",
    "spine_from_witness",
    "witness_spine",
]
