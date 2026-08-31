"""The revised M1 gate's target, ranker interface and split, as reusable pieces.

`research/experiments/m1_gate.py` is the driver; everything it needs that can be
unit-tested without generating a corpus lives here.

Three things the original gate got wrong, and what this module does instead
(note 22 §M1, "The revised gate"):

1. **Feasibility is vacuous.**  Brute force measured that 80.5% of all legal
   sites admit *a* completion inside the witness's own remaining budget, because
   there is a cliff at `m* + 2` moves of slack -- exactly the cost of undoing one
   move in a move set closed under inverses.  So the target here is the
   **shortest** completion: a site is positive when it is the first move of a
   completion no longer than the shortest one that exists, under the crossing
   change budget the witness had left.  `shortest_completion` computes the whole
   set in one search rather than one search per site.
2. **recall@10 has no headroom.**  Everything below reports recall@1 and leaves
   the k to the caller.
3. **A ranker with no inputs passed.**  The ranker interface is
   `rank(state, sites) -> scores`, ties are broken by the *harness* with a seeded
   shuffle rather than by the ranker, and the same code path scores the
   hand-written baselines and a learned head.  A baseline cannot earn credit
   from correlation with the flat action index and neither can the head.

And one thing §M1 asks for that this module measured to be **not well posed**:
gating on the `INSERT` block.  `BlockTarget`'s docstring carries the argument and
the numbers; the short form is that `moves = (L + n - 1)/2 + 2i + 2s + c` exactly,
so an insertion costs two moves nothing pays back, and `REDUCE(p)` undoes
`INSERT(p, ...)`, so every `INSERT` site reaches the unknot in exactly `m* + 2`.
The block is flat under a target derived from *move* distance to the unknot -- so
`score_state` reports it, and reports the residual `rest@k` beside it, and the
driver gates on neither alone.

## The retarget: `(crossing_changes, moves)`, lexicographically

**The flatness above is a property of the objective, not of the move set.**  §2.1
rejects `A*cc + B*moves` as a scalar and asks for the lexicographic comparison
`(cc, moves)` with `A = 100_000 > M = 10_000` wherever a scalar is forced -- and
the previous target here scored *shortest moves*, which is the objective the
campaign threw out.  Under shortest-moves the `INSERT` block is provably
uninteresting from the counting identity; under `(cc, moves)` an insertion that
removes even one crossing change wins outright, because `cc` dominates
absolutely.  `objective="lex"` is now the default and `objective="moves"` is kept
so the published `artifacts/m1-gate` table stays reproducible; the two are never
mixed in one column.

`shortest_completion` therefore returns **both** components of the optimum:
`crossing_changes` is `c*(s)` and `moves` is the fewest moves among completions
achieving it.  Two things make the lexicographic search affordable:

* the priority is the pair `(max(used, floor), moves + move_bound)`, which is
  consistent for exactly the same reason the scalar one was -- the first
  component never decreases along an edge and the second is the old consistent
  `f` -- so A* may still test the goal on pop and the first-action mask is still
  exact;
* `floor` is a **certified** lower bound on `u` of the starting state
  (`certified_change_floor`, from `rf_knots.lower_bounds.computed_claims`, so a
  theorem rather than a table row).  Without it the search has to exhaust the
  whole `cc < c*` region to prove `c*`; with a tight floor it stops at the first
  goal it pops, and costs what the move-ordered search cost.  An unsound floor
  would be caught rather than believed: reaching the unknot with fewer changes
  than the floor raises.

**Nothing here ever records a site as bad.**  A search that hits its cap leaves
the state `undetermined` and the state is dropped from the gate and counted;
`aplus.py`'s rule -- absent is unexplored, never negative -- is why the target is
only ever called *optimal known* -- shortest known moves under
`objective="moves"`, and the lexicographically least known `(cc, moves)` under
`objective="lex"`.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, INSERT, KIND_NAMES, REDUCE, ActionSpec
from rf_knots.aplus import (
    CROSSING_CHANGE_TIER_FLAT,
    CROSSING_CHANGE_TIER_PRODUCTIVE,
    KIND_TIER,
    free_reduced_drop,
    insert_context_score,
)
from rf_knots.evidence import UnknotWitness

Word = tuple[int, ...]
Successors = Callable[[Word, int], list[tuple[int, Word, int]]]


def move_bound(length: int, strands: int) -> int:
    """Moves any completion must spend.  Strictly at least `aplus.move_lower_bound`.

    Count the moves of a completion by kind -- `r` reductions, `i` insertions,
    `s` stabilisations, `d` destabilisations and `c` length-preserving moves
    (`COMMUTE`, `BRAID`, `CROSSING_CHANGE`).  The move set forces two balances:

        strands:  n + s - d = 1          =>  d = n - 1 + s
        letters:  L + 2i + s - 2r - d = 0  =>  r = (L + 2i - n + 1) / 2

    Substituting into `r + i + s + d + c` gives

        moves = (L + n - 1) / 2 + 2i + 2s + c   >=  ceil((L + n - 1) / 2)

    which dominates `aplus.move_lower_bound`'s `ceil(L/2)` by exactly the
    destabilisations the strand count forces, and is what makes an optimal search
    on a twenty-letter word finish at all.  `n - 1` is kept as a separate term
    because it is the binding one on a short word over many strands.

    It is consistent, so A* may test the goal on pop: every move changes
    `L + n - 1` by at most two and `n - 1` by at most one, and a maximum of two
    consistent functions is consistent.
    """
    return max(strands - 1, -(-(length + strands - 1) // 2))


# -- the optimal-completion target ---------------------------------------------


@dataclass(frozen=True)
class ShortestCompletion:
    """The optimum `(c*(s), m*(s))` and every site that starts a completion of it.

    `moves is None` means **no completion was found under the caps**, which is a
    statement about the search and never about the state.  `complete` is True
    only when the search that produced `first_actions` exhausted its `f <= f*`
    layer, which is what makes the set exact for the searched subgraph.

    `crossing_changes` is the first component of the optimum under
    `objective="lex"`, and is simply the change count of the shortest completion
    under `objective="moves"` -- under that objective it is *not* minimised and
    must not be read as `c*`.
    """

    moves: int | None
    first_actions: tuple[int, ...]
    legal_actions: tuple[int, ...]
    expanded: int
    rounds: int
    complete: bool
    reason: str
    growth: int
    node_cap: int
    #: The first moves the search was allowed to take.  Equal to `legal_actions`
    #: for a whole-state target; one kind block for a block-conditional one.
    candidates: tuple[int, ...] = ()
    #: True when no path was allowed to re-enter the starting state.
    commit: bool = False
    #: "lex" -- minimise `(crossing_changes, moves)`; "moves" -- minimise moves
    #: subject to the crossing-change budget, which is what the published
    #: `artifacts/m1-gate` run scored.
    objective: str = "lex"
    #: The optimum's crossing-change count; `None` alongside `moves is None`.
    crossing_changes: int | None = None
    #: The certified lower bound on `u` that the lexicographic search used as its
    #: admissible first-component heuristic.  Zero is always sound.
    crossing_change_floor: int = 0

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "shortest_completion_moves": self.moves,
            "optimal_crossing_changes": self.crossing_changes,
            "crossing_change_floor": self.crossing_change_floor,
            "candidates": len(self.candidates),
            "commit": self.commit,
            "first_actions": list(self.first_actions),
            "legal_actions": len(self.legal_actions),
            "expanded": self.expanded,
            "rounds": self.rounds,
            "complete": self.complete,
            "reason": self.reason,
        }


LEX = "lex"
MOVES = "moves"
OBJECTIVES = (LEX, MOVES)


def certified_change_floor(word: Word, strands: int) -> int:
    """A **certified** lower bound on `u` of the closure, or 0 if none is free.

    `rf_knots.lower_bounds.computed_claims` computes from the word alone -- the
    Murasugi signature bound, the Montesinos cyclic obstruction and the
    Ozsvath-Szabo `tau` bound -- so every value is a theorem and never a table
    row, which is the rule §4.1b's second hazard states.  Measured at ~2.5 ms per
    state, and it is what makes the lexicographic search affordable: it is the
    admissible heuristic on the first component.

    Returns 0 rather than raising when the backend is unavailable.  Zero is sound
    -- it is the trivial bound -- and it costs search time, not correctness.
    """
    from rf_knots import lower_bounds

    try:
        claims = lower_bounds.computed_claims(tuple(word), int(strands))
    except Exception:  # pragma: no cover - optional backend
        return 0
    value = lower_bounds.strongest(claims)
    return int(value) if value else 0


def _successors_for(spec: ActionSpec, successors: Successors | None) -> Successors:
    if successors is not None:
        return successors

    def call(word: Word, strands: int) -> list[tuple[int, Word, int]]:
        return reference.successors(spec, word, strands, allow_crossing=True)

    return call


def _goal(cost: tuple[int, int] | None, changes: int) -> tuple[int, int] | None:
    """`(crossing changes, moves)` of the optimum, whichever objective ordered it."""
    return None if cost is None else (changes, cost[1])


def _f_bounded_pass(
    spec: ActionSpec,
    word: Word,
    strands: int,
    crossing_change_budget: int,
    f_limit: tuple[int, int],
    length_cap: int,
    node_cap: int,
    successors: Successors,
    index_of: dict[int, int],
    forbidden: frozenset[tuple[Word, int]],
    lex: bool,
    change_floor: int,
) -> tuple[tuple[int, int] | None, int, tuple[int, int] | None, int, bool]:
    """One A* pass restricted to `f <= f_limit`, propagating first-action masks.

    Returns `(goal_cost, goal_mask, next_f, expanded, hit_cap)`.  `goal_cost` is
    the pair `(crossing_changes, moves)`; `next_f` is the smallest `f` the pass
    refused, so the caller can raise the limit to it and re-run, and `None` means
    the whole `f <= f_limit` component was exhausted.

    **The priority.**  `f = (max(used, change_floor), moves + move_bound)` under
    `lex`, and `(0, moves + move_bound)` otherwise -- so the scalar case is the
    lexicographic one with its first component held constant, and one code path
    serves both.  Both components are consistent: the first never decreases along
    an edge because `used` never decreases, and the second is the old `f`, since
    `move_bound` moves by at most one per move.  A maximum of consistent
    functions is consistent, so lexicographic order on the pair is too.

    Correctness of the mask, which is the only subtle part.  Consistency gives
    `f(u) <= f(v)` for any edge `u -> v` on an optimal path, and the heap breaks
    ties on `f` by `g = (used, moves)` ascending -- which every edge strictly
    increases, since it adds `(0, 1)` or `(1, 1)`.  So every optimal predecessor
    of `v` pops strictly before `v`, and the mask read at `v`'s pop is complete.
    """
    start = (word, strands, 0)
    best: dict[tuple[Word, int, int], int] = {start: 0}
    masks: dict[tuple[Word, int, int], int] = {start: 0}
    settled: set[tuple[Word, int, int]] = set()
    counter = 0
    floor = change_floor if lex else 0
    heap: list[tuple[int, int, int, int, int, tuple[Word, int, int]]] = [
        (floor, move_bound(len(word), strands), 0, 0, counter, start)
    ]
    expanded = 0
    goal_cost: tuple[int, int] | None = None
    goal_mask = 0
    goal_changes = 0
    next_f: tuple[int, int] | None = None
    while heap:
        f_changes, f_moves, _used, g, _tie, key = heapq.heappop(heap)
        if goal_cost is not None and (f_changes, f_moves) > goal_cost:
            break
        if key in settled or best.get(key, -1) < g:
            continue
        settled.add(key)
        current, active, used = key
        if not current and active == 1:
            if lex and used < floor:  # pragma: no cover - a false certificate
                raise AssertionError(
                    f"reached the unknot in {used} crossing changes from a state "
                    f"whose certified lower bound is {floor}"
                )
            #: The ordering key, which under `moves` must not see `used` at all
            #: -- two completions of equal length tie however many changes they
            #: spend, and their first actions are pooled.  `goal_changes` is the
            #: fewest changes among the completions that attain the optimum, and
            #: under `lex` it *is* the optimum's first component.
            here = (used, g) if lex else (0, g)
            if goal_cost is None or here < goal_cost:
                goal_cost, goal_mask, goal_changes = here, masks[key], used
            elif here == goal_cost:
                goal_mask |= masks[key]
                goal_changes = min(goal_changes, used)
            continue
        if expanded >= node_cap:
            return _goal(goal_cost, goal_changes), goal_mask, next_f, expanded, True
        expanded += 1
        mask_here = masks[key]
        at_start = key == start
        for action, next_word, next_strands in successors(current, active):
            if at_start and action not in index_of:
                continue
            if len(next_word) > length_cap:
                continue
            if (next_word, next_strands) in forbidden:
                continue
            changes = used + int(spec.decode(action)[0] == CROSSING_CHANGE)
            if changes > crossing_change_budget:
                continue
            moves = g + 1
            child = (next_word, next_strands, changes)
            if child in settled:
                continue
            bound = (
                (changes if changes > floor else floor) if lex else 0,
                moves + move_bound(len(next_word), next_strands),
            )
            if bound > f_limit:
                next_f = bound if next_f is None else min(next_f, bound)
                continue
            mask = 1 << index_of[action] if at_start else mask_here
            recorded = best.get(child)
            if recorded is None or moves < recorded:
                best[child] = moves
                masks[child] = mask
                counter += 1
                heapq.heappush(heap, (*bound, changes, moves, counter, child))
            elif moves == recorded:
                masks[child] = masks[child] | mask
    return _goal(goal_cost, goal_changes), goal_mask, next_f, expanded, False


def shortest_completion(
    spec: ActionSpec,
    word: Word,
    strands: int,
    crossing_change_budget: int,
    growth: int = 3,
    node_cap: int = 20_000,
    successors: Successors | None = None,
    block: Sequence[int] | None = None,
    commit: bool = False,
    objective: str = LEX,
    change_floor: int | None = None,
) -> ShortestCompletion:
    """Every site that begins an optimal completion, in one search.

    Under `objective="lex"` -- the default, and §2.1's objective -- the optimum is
    the lexicographic minimum of `(crossing_changes, moves)`: fewest crossing
    changes first, then fewest moves among those.  Under `objective="moves"` it is
    the fewest moves subject to the crossing-change budget, which is what the
    published `artifacts/m1-gate` table scored and what the counting identity in
    `BlockTarget` makes flat on the `INSERT` block.

    The crossing-change budget is the witness's remaining one -- that is the
    `u <= k` constraint and it is what makes the problem the problem.  A witness
    achieving it exists by construction, so under `lex` the budget never binds the
    optimum; it only bounds the search.  The move budget is deliberately **not**
    the witness's: the shortest completion is whatever it is, and the old gate's
    slack is exactly what made it vacuous.

    `change_floor` is the admissible first-component heuristic and must be a
    *certified* lower bound on `u` of `(word, strands)`; `None` computes one with
    `certified_change_floor`, and `0` disables it at a cost in search, never in
    correctness.  It is ignored under `objective="moves"`.

    `block` restricts the first move to one kind block, so the result is the
    optimum *through that block* and the set is its argmin.  `commit`
    additionally forbids every path from re-entering the starting state, which is
    what makes the `INSERT` block a real question rather than a degenerate one --
    see `BlockTarget`.

    `growth` caps the word length at `len(word) + growth`, following
    `aplus.bounded_completion`; a completion that has to grow the diagram further
    is outside the searched subgraph and this is a limit of the target, recorded
    as such.  The returned set is therefore *optimal known*: exact for the
    searched subgraph, an upper bound on the true optimum.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, not {objective!r}")
    lex = objective == LEX
    succ = _successors_for(spec, successors)
    legal = tuple(action for action, _w, _s in succ(word, strands))
    candidates = tuple(block) if block is not None else legal
    index_of = {action: bit for bit, action in enumerate(candidates)}
    forbidden = frozenset({(word, strands)}) if commit else frozenset()
    length_cap = min(len(word) + growth, spec.max_len)
    floor = 0
    if lex:
        floor = (
            certified_change_floor(word, strands) if change_floor is None
            else int(change_floor)
        )
        if floor > crossing_change_budget:
            #: A **proof**, not a cap.  `floor` is a certified lower bound on
            #: `u`, and any completion spends at least `u` crossing changes, so
            #: no completion fits inside a budget below it.  `complete` is True
            #: and `moves` is None: infeasible-and-proved, which is a different
            #: fact from infeasible-as-far-as-the-search-got and is why the two
            #: carry different `reason` strings.
            return ShortestCompletion(
                None, (), tuple(action for action, _w, _s in succ(word, strands)),
                0, 0, True, "certified-floor-exceeds-budget", growth, node_cap,
                candidates, commit, objective, None, floor,
            )
    limit = (floor, move_bound(len(word), strands))
    total = 0
    rounds = 0
    while True:
        rounds += 1
        goal, mask, next_f, expanded, hit_cap = _f_bounded_pass(
            spec,
            word,
            strands,
            crossing_change_budget,
            limit,
            length_cap,
            node_cap - total,
            succ,
            index_of,
            forbidden,
            lex,
            floor,
        )
        total += expanded
        if hit_cap:
            return ShortestCompletion(
                None, (), legal, total, rounds, False, "node-cap", growth, node_cap,
                candidates, commit, objective, None, floor,
            )
        if goal is not None:
            firsts = tuple(
                sorted(candidates[bit] for bit in range(len(candidates)) if mask >> bit & 1)
            )
            return ShortestCompletion(
                goal[1], firsts, legal, total, rounds, True, "found", growth, node_cap,
                candidates, commit, objective, goal[0], floor,
            )
        if next_f is None:
            return ShortestCompletion(
                None, (), legal, total, rounds, True, "exhausted", growth, node_cap,
                candidates, commit, objective, None, floor,
            )
        limit = next_f


def completion_path(
    spec: ActionSpec,
    word: Word,
    strands: int,
    crossing_change_budget: int,
    f_limit: int,
    growth_from: int,
    node_cap: int = 20_000,
    forbidden: frozenset[tuple[Word, int]] = frozenset(),
) -> tuple[int, ...] | None:
    """A concrete completion of at most `f_limit` moves, or `None` under the cap.

    Used only to *exhibit* what `shortest_completion` claims, so the length cap is
    measured from the state the claim was made at, not from here.
    """
    length_cap = min(growth_from, spec.max_len)
    start = (word, strands, 0)
    if not word and strands == 1:
        return ()
    best: dict[tuple[Word, int, int], int] = {start: 0}
    parent: dict[tuple[Word, int, int], tuple[tuple[Word, int, int], int]] = {}
    counter = 0
    heap = [(move_bound(len(word), strands), 0, counter, start)]
    expanded = 0
    while heap:
        f, g, _tie, key = heapq.heappop(heap)
        if f > f_limit or best.get(key, -1) < g:
            continue
        current, active, used = key
        if not current and active == 1:
            path: list[int] = []
            node = key
            while node in parent:
                node, action = parent[node]
                path.append(action)
            return tuple(reversed(path))
        if expanded >= node_cap:
            return None
        expanded += 1
        for action, next_word, next_strands in reference.successors(
            spec, current, active, allow_crossing=True
        ):
            if len(next_word) > length_cap:
                continue
            if (next_word, next_strands) in forbidden:
                continue
            changes = used + int(spec.decode(action)[0] == CROSSING_CHANGE)
            if changes > crossing_change_budget:
                continue
            moves = g + 1
            if moves + move_bound(len(next_word), next_strands) > f_limit:
                continue
            child = (next_word, next_strands, changes)
            recorded = best.get(child)
            if recorded is not None and recorded <= moves:
                continue
            best[child] = moves
            parent[child] = (key, action)
            counter += 1
            heapq.heappush(
                heap,
                (moves + move_bound(len(next_word), next_strands), moves, counter, child),
            )
    return None


def exhibit_target(
    spec: ActionSpec,
    word: Word,
    strands: int,
    crossing_change_budget: int,
    result: ShortestCompletion,
    node_cap: int = 20_000,
) -> dict[int, int]:
    """Replay a shortest completion out of every site the target claims.

    `aplus.py`'s standard is that membership is *exhibited*, never inferred, and
    the target set above comes out of a mask propagation rather than out of a
    path.  This reconstructs one concrete path per site and hands it to
    `UnknotWitness.from_actions`, which replays it independently.  A site that
    does not replay at exactly the claimed length raises: that is a bug in the
    propagation, not a datum.

    Returns `{action: replayed length}`.  A site whose extraction hits its own cap
    is omitted, so the caller can report the exhibited fraction.
    """
    if result.moves is None:
        return {}
    out: dict[int, int] = {}
    length_cap = min(len(word) + result.growth, spec.max_len)
    #: Under `lex` the tail is searched inside the optimum's *own* change count,
    #: not the state's budget.  Fewer moves at that count would contradict `m*`
    #: and fewer changes would contradict `c*`, so the min-moves path under this
    #: budget is exactly the optimum's tail -- and the replay below checks it
    #: rather than trusting the argument.
    budget = (
        result.crossing_changes
        if result.objective == LEX and result.crossing_changes is not None
        else crossing_change_budget
    )
    for action in result.first_actions:
        next_word, next_strands = reference.apply(spec, word, strands, action)
        is_change = spec.decode(action)[0] == CROSSING_CHANGE
        tail = completion_path(
            spec,
            next_word,
            next_strands,
            budget - int(is_change),
            result.moves - 1,
            length_cap,
            node_cap=node_cap,
        )
        if tail is None:
            continue
        witness = UnknotWitness.from_actions(word, strands, spec, (action, *tail))
        if witness.moves != result.moves:
            raise AssertionError(
                f"target site {spec.describe(action)} replayed at {witness.moves} "
                f"moves, but the target claims {result.moves}"
            )
        if witness.crossing_changes > budget:
            raise AssertionError(
                f"target site {spec.describe(action)} spends "
                f"{witness.crossing_changes} crossing changes over a budget of "
                f"{budget}"
            )
        if (
            result.objective == LEX
            and witness.crossing_changes != result.crossing_changes
        ):
            raise AssertionError(
                f"target site {spec.describe(action)} replayed at "
                f"{witness.crossing_changes} crossing changes against a claimed "
                f"optimum of {result.crossing_changes}"
            )
        out[action] = witness.moves
    return out


# -- the block-conditional target, and why the gate needs one ------------------


@dataclass(frozen=True)
class BlockTarget:
    """Inside one kind block: which sites lead to the unknot fastest.

    **Why this exists.  It is a third defect in the gate, not a convenience.**
    §M1's revised gate asks for two things at once -- score against the shortest
    completion, and gate on the `INSERT` block -- and the two are jointly empty.

    Count any completion's moves by kind: `r` reductions, `i` insertions, `s`
    stabilisations, `d` destabilisations and `c` length-preserving moves
    (`COMMUTE`, `BRAID`, `CROSSING_CHANGE`).  The move set forces
    `d = n - 1 + s` and `2r = L + 2i - n + 1`, so

        moves = (L + n - 1) / 2 + 2i + 2s + c

    **exactly** -- verified by counting the kinds of real witness suffixes, not
    argued.  Every insertion therefore costs two moves that nothing in the
    identity pays back, and an `INSERT` reaches a shortest completion only by
    buying back two length-preserving moves.  Measured over 212 states of three
    configurations, at word lengths 3 to 21 and 2 to 6 strands, that happened at
    **no state**: the shortest-*moves* target contains zero `INSERT` sites.  That
    is a statement about the move objective and about nothing else, and it is
    what `objective="lex"` exists to stop being the only thing measured.

    Worse than empty, the block is *degenerate*.  `INSERT(p, g, s)` splices a
    cancelling pair before position `p` and `REDUCE(p)` deletes it again --
    checked by application -- so **every** `INSERT` site admits a completion of
    exactly `m* + 2`: insert, undo, take a shortest completion.  Under any
    shortest-completion target the whole block ties, and every ordering of it is
    equally right.  That is also the mechanical reason the earlier report found
    that no hand-written score beat random ordering inside it.

    **The repair, which keeps both of §M1's requirements.**  A search choosing an
    insertion has already decided to grow; what it needs ranked is which growth
    is worth it.  So the target forbids the undo: it is the shortest completion
    through a site of the block **that never re-enters the state it came from**.
    The trivial `m* + 2` route is gone by construction, the block splits, and the
    question is exactly the one §M1 says nothing currently ranks -- *where may the
    diagram grow*.
    """

    kind: int
    baseline_moves: int
    block: tuple[int, ...]
    best_moves: int | None
    target: tuple[int, ...]
    exhibited: dict[int, int]
    undo_cost_exhibited: dict[int, int]
    complete: bool
    expanded: int

    @property
    def discriminating(self) -> bool:
        """True when the block splits -- some sites reach the unknot sooner."""
        return bool(self.target) and len(self.target) < len(self.block)

    def to_dict(self) -> dict:
        return {
            "kind": KIND_NAMES[self.kind] if self.kind >= 0 else None,
            "baseline_moves": self.baseline_moves,
            "block_sites": len(self.block),
            "committed_best_moves": self.best_moves,
            "target": list(self.target),
            "discriminating": self.discriminating,
            "complete": self.complete,
        }


def undo_completion(
    spec: ActionSpec,
    word: Word,
    strands: int,
    action: int,
    shortest_path: Sequence[int],
) -> tuple[int, ...]:
    """The `m* + 2` completion every `INSERT` site has: insert, undo, then go.

    Constructive -- no search -- because `INSERT(p, ...)` is undone by
    `REDUCE(p)` by construction of the move set.  The caller replays it, which is
    what turns "the block is degenerate" from an argument into an exhibit.
    """
    kind, position, _generator, _sign = spec.decode(action)
    if kind != INSERT:
        raise ValueError("only an INSERT has a guaranteed undo at its own position")
    return (action, spec.encode(REDUCE, position=position), *shortest_path)


def block_target(
    spec: ActionSpec,
    word: Word,
    strands: int,
    crossing_change_budget: int,
    block: Sequence[int],
    shortest_moves: int,
    growth: int = 3,
    node_cap: int = 20_000,
    successors: Successors | None = None,
    exhibit: bool = True,
    objective: str = LEX,
    change_floor: int | None = None,
) -> BlockTarget:
    """The committed shortest completion through one kind block, in one search.

    One search, not one per site: `shortest_completion` restricts the first move
    to the block and propagates a first-action mask, so the argmin over the whole
    block comes out of a single `f`-bounded A*.
    """
    if not block:
        return BlockTarget(-1, shortest_moves, (), None, (), {}, {}, True, 0)
    found = shortest_completion(
        spec,
        word,
        strands,
        crossing_change_budget,
        growth=growth,
        node_cap=node_cap,
        successors=successors,
        block=block,
        commit=True,
        objective=objective,
        change_floor=change_floor,
    )
    exhibited: dict[int, int] = {}
    if exhibit and found.moves is not None:
        forbidden = frozenset({(word, strands)})
        length_cap = min(len(word) + growth, spec.max_len)
        budget = (
            found.crossing_changes
            if objective == LEX and found.crossing_changes is not None
            else crossing_change_budget
        )
        for action in found.first_actions:
            is_change = spec.decode(action)[0] == CROSSING_CHANGE
            next_word, next_strands = reference.apply(spec, word, strands, action)
            tail = completion_path(
                spec,
                next_word,
                next_strands,
                budget - int(is_change),
                found.moves - 1,
                length_cap,
                node_cap=node_cap,
                forbidden=forbidden,
            )
            if tail is None:
                continue
            witness = UnknotWitness.from_actions(
                word, strands, spec, (action, *tail)
            )
            if witness.moves != found.moves:
                raise AssertionError(
                    f"block site {spec.describe(action)} replayed at "
                    f"{witness.moves} moves against a claim of {found.moves}"
                )
            exhibited[action] = witness.moves
    return BlockTarget(
        kind=spec.decode(block[0])[0],
        baseline_moves=shortest_moves,
        block=tuple(block),
        best_moves=found.moves,
        target=found.first_actions,
        exhibited=exhibited,
        undo_cost_exhibited={},
        complete=found.complete and found.moves is not None,
        expanded=found.expanded,
    )


# -- the ranker interface ------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """One legal action at a state, decoded once so no ranker has to decode it."""

    action: int
    kind: int
    position: int
    generator: int
    sign: int
    word: Word
    strands: int

    @property
    def letter(self) -> int:
        return self.sign * self.generator


@dataclass(frozen=True)
class GateState:
    """A state the gate scores, with everything a ranker or a slice may read."""

    spec: ActionSpec
    word: Word
    strands: int
    remaining_crossing_changes: int
    remaining_moves: int
    config: str
    seed_name: str
    instance_id: str
    identity: str
    depth: int
    region: str


#: `rank(state, sites) -> scores`, higher is better.  Ties are the harness's job.
Ranker = Callable[[GateState, Sequence[Site]], Sequence[float]]


def sites_of(
    spec: ActionSpec,
    word: Word,
    strands: int,
    successors: Successors | None = None,
) -> tuple[Site, ...]:
    """Decode every legal action once, and check the decode round-trips.

    `(kind, position, generator, sign)` is the whole of a site, and the fourth
    field is the one it is tempting to drop: `INSERT` is indexed as
    `3L + ((g - 1) * 2 + s) * L + p`, so `INSERT(p, g, +1)` and `INSERT(p, g, -1)`
    are **distinct actions**.  Collapsing the sign would merge pairs of sites
    inside the one block this gate is scored on and inflate every ranker's
    recall there.  The round-trip through `ActionSpec.encode` is asserted on
    every state rather than in a test alone: a site that cannot reproduce its own
    flat index means the harness and `aplus.py` -- which stores flat indices --
    disagree about what a site *is*, and every number would then be about a
    different action space.
    """
    succ = _successors_for(spec, successors)
    out = []
    for action, next_word, next_strands in succ(word, strands):
        kind, position, generator, sign = spec.decode(action)
        flat = spec.encode(kind, position=position, generator=generator, sign=sign)
        if flat != action:
            raise AssertionError(
                f"site {spec.describe(action)} decodes to "
                f"(kind={kind}, position={position}, generator={generator}, "
                f"sign={sign}) which re-encodes to {flat}, not {action}"
            )
        out.append(
            Site(action, kind, position, generator, sign, next_word, next_strands)
        )
    return tuple(out)


def _squash(value: float) -> float:
    """A bounded, strictly monotone secondary key, so tiers stay lexicographic."""
    return 0.5 * value / (1.0 + abs(value))


def uniform_random(state: GateState, sites: Sequence[Site]) -> list[float]:
    """The control.  Every score equal, so the harness's tie-break decides."""
    return [0.0] * len(sites)


def kind_prior(state: GateState, sites: Sequence[Site]) -> list[float]:
    """`aplus.KIND_TIER` and nothing else.  It never looks at the word.

    This is the gate's baseline.  Inside the `INSERT` block it is constant, which
    is the whole point: there it *is* random ordering, and a head that cannot beat
    it there has learned nothing about where the diagram may grow.
    """
    return [
        float(KIND_TIER.get(site.kind, CROSSING_CHANGE_TIER_FLAT)) for site in sites
    ]


def free_reduced_drop_ranker(state: GateState, sites: Sequence[Site]) -> list[float]:
    """§M1's named kill criterion, literally: the drop in free-reduced length."""
    return [float(free_reduced_drop(state.word, site.word)) for site in sites]


def growth_aware(state: GateState, sites: Sequence[Site]) -> list[float]:
    """The full hand-written heuristic of `aplus.rank_growth_aware`, as scores."""
    out = []
    for site in sites:
        drop = free_reduced_drop(state.word, site.word)
        if site.kind == CROSSING_CHANGE:
            tier = (
                CROSSING_CHANGE_TIER_PRODUCTIVE if drop > 0 else CROSSING_CHANGE_TIER_FLAT
            )
            secondary = float(drop)
        elif site.kind == INSERT:
            tier = KIND_TIER[INSERT]
            secondary = float(insert_context_score(state.word, site.position, site.letter))
        else:
            tier = KIND_TIER.get(site.kind, 0)
            secondary = float(drop)
        out.append(float(tier) + _squash(secondary))
    return out


BASELINES: dict[str, Ranker] = {
    "uniform_random": uniform_random,
    "kind_prior": kind_prior,
    "free_reduced_drop": free_reduced_drop_ranker,
    "growth_aware": growth_aware,
}


def order_by_scores(scores: Sequence[float], rng: np.random.Generator) -> list[int]:
    """Indices best-first, ties broken uniformly at random by the harness.

    Every ranker gets the same treatment, so no score function can earn credit
    from correlation with the flat action index -- which is the order
    `reference.successors` happens to emit.
    """
    noise = rng.random(len(scores))
    return sorted(range(len(scores)), key=lambda i: (-scores[i], noise[i]))


def recall_at(order: Sequence[int], positives: set[int], k: int) -> int:
    return int(any(index in positives for index in order[:k]))


def chance_at(total: int, positives: int, k: int) -> float:
    """`1 - C(n-m, k)/C(n, k)`: what a uniform ranker scores, in closed form."""
    import math

    if positives <= 0 or total <= 0:
        return 0.0
    if k >= total or total - positives < k:
        return 1.0
    return 1.0 - math.comb(total - positives, k) / math.comb(total, k)


# -- scoring one state ---------------------------------------------------------


@dataclass
class ScoredState:
    """One state, its target, and where each ranker put a target site."""

    state: GateState
    legal: int
    insert_legal: int
    target: tuple[int, ...]
    target_insert: tuple[int, ...]
    rest_legal: int
    target_rest: tuple[int, ...]
    shortest_moves: int
    exhibited: int
    hits: dict[str, dict[str, int]] = field(default_factory=dict)
    chance: dict[str, float] = field(default_factory=dict)
    tier2_sites: tuple[int, ...] = ()
    tier2_in_target: int = 0
    tier3_sites: tuple[int, ...] = ()


def score_state(
    state: GateState,
    sites: Sequence[Site],
    target: Sequence[int],
    shortest_moves: int,
    rankers: dict[str, Ranker],
    rng: np.random.Generator,
    ks: Sequence[int] = (1, 5, 10),
) -> ScoredState:
    """Score every ranker on the whole state and on the `INSERT` block alone.

    Three scopes, all through the same interface, because the interface takes
    the site list as an argument: the whole state (`all@k`), the `INSERT` block
    alone (`insert@k`), and everything that is not an `INSERT` (`rest@k`).  A
    block is scored by handing the ranker *only* that block, which is the honest
    form of "rank where the diagram may grow" -- the ranker is asked the question
    the search would ask it, not asked a wider question and graded on part of the
    answer.

    `rest@k` is where a head can still earn something.  `INSERT` is most of the
    legal set and none of the target, so `all@k` is dominated by a decision
    `kind_prior` already makes for free; the residual ordering lives among the
    reductions, commutations, braid moves and crossing changes.
    """
    positives = set(target)
    insert_indices = [i for i, site in enumerate(sites) if site.kind == INSERT]
    insert_sites = [sites[i] for i in insert_indices]
    insert_positives = {
        i for i, site in enumerate(insert_sites) if site.action in positives
    }
    rest_sites = [site for site in sites if site.kind != INSERT]
    rest_positives = {
        i for i, site in enumerate(rest_sites) if site.action in positives
    }
    scored = ScoredState(
        state=state,
        legal=len(sites),
        insert_legal=len(insert_sites),
        target=tuple(sorted(positives)),
        target_insert=tuple(
            sorted(site.action for site in insert_sites if site.action in positives)
        ),
        rest_legal=len(rest_sites),
        target_rest=tuple(
            sorted(site.action for site in rest_sites if site.action in positives)
        ),
        shortest_moves=shortest_moves,
        exhibited=0,
    )
    full_positives = {i for i, site in enumerate(sites) if site.action in positives}
    for name, ranker in rankers.items():
        full_order = order_by_scores(ranker(state, sites), rng)
        cells = {f"all@{k}": recall_at(full_order, full_positives, k) for k in ks}
        if insert_sites:
            insert_order = order_by_scores(ranker(state, insert_sites), rng)
            for k in ks:
                cells[f"insert@{k}"] = recall_at(insert_order, insert_positives, k)
        if rest_sites:
            rest_order = order_by_scores(ranker(state, rest_sites), rng)
            for k in ks:
                cells[f"rest@{k}"] = recall_at(rest_order, rest_positives, k)
        scored.hits[name] = cells
    for k in ks:
        scored.chance[f"all@{k}"] = chance_at(len(sites), len(full_positives), k)
        scored.chance[f"insert@{k}"] = chance_at(
            len(insert_sites), len(insert_positives), k
        )
        scored.chance[f"rest@{k}"] = chance_at(
            len(rest_sites), len(rest_positives), k
        )
    return scored


# -- aggregation ---------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return round(float(sum(values) / len(values)), 4) if values else None


def summarise(
    rows: Sequence[ScoredState],
    metric: str,
    ranker_names: Sequence[str],
) -> dict:
    """Mean of one metric over `rows`, per ranker, with the closed-form chance.

    A state with an empty block has no score for that block's metric -- there is
    nothing to order -- so it is dropped from both the recall and the chance.
    Averaging chance over rows a ranker was not scored on would put the two
    columns on different denominators, which is how a chance line ends up
    flattering a ranker for free.
    """
    subset = [row for row in rows if metric in row.chance and _scored(row, metric)]
    out: dict = {"states": len(subset)}
    if not subset:
        return out
    for name in ranker_names:
        out[name] = _mean(
            [row.hits[name][metric] for row in subset if metric in row.hits[name]]
        )
    out["uniform_chance"] = _mean([row.chance[metric] for row in subset])
    return out


def _scored(row: ScoredState, metric: str) -> bool:
    return any(metric in cells for cells in row.hits.values())


def cluster_bootstrap(
    rows: Sequence[ScoredState],
    metric: str,
    left: str,
    right: str,
    draws: int = 2000,
    seed: int = 20260831,
) -> dict:
    """Paired bootstrap of `left - right`, resampling **instances**, not states.

    States on one witness are not independent -- they share a word, a seed and a
    budget -- so a state-level interval would be far too narrow.  The unit of
    resampling is the instance.
    """
    if not rows:
        return {"draws": 0}
    groups: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        if metric not in row.hits[left] or metric not in row.hits[right]:
            continue
        groups.setdefault(row.state.instance_id, []).append(
            (row.hits[left][metric], row.hits[right][metric])
        )
    keys = sorted(groups)
    if not keys:
        return {"draws": 0}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        picked = rng.integers(0, len(keys), size=len(keys))
        lefts = rights = 0
        count = 0
        for index in picked:
            for a, b in groups[keys[index]]:
                lefts += a
                rights += b
                count += 1
        if count:
            deltas.append((lefts - rights) / count)
    if not deltas:
        return {"draws": 0}
    array = np.asarray(deltas)
    observed = sum(a for values in groups.values() for a, _b in values) - sum(
        b for values in groups.values() for _a, b in values
    )
    total = sum(len(v) for v in groups.values())
    return {
        "draws": draws,
        "instances": len(keys),
        "states": total,
        "observed_delta": round(observed / total, 4),
        "ci95_low": round(float(np.quantile(array, 0.025)), 4),
        "ci95_high": round(float(np.quantile(array, 0.975)), 4),
        "fraction_of_draws_at_or_below_zero": round(float((array <= 0).mean()), 4),
    }


# -- the identity-level split --------------------------------------------------


@dataclass(frozen=True)
class SplitReport:
    """What the split screened, and the sentence it is allowed to claim."""

    train_instances: int
    eval_instances: int
    identities: int
    train_identities: int
    eval_identities: int
    merged_by_state_collision: int
    eval_states_screened_out: int
    eval_states_kept: int
    train_strand_counts: tuple[int, ...]
    transfer_strand_counts: tuple[int, ...]
    largest_group: int = 0
    singleton_groups: int = 0

    def to_dict(self) -> dict:
        return {
            "train_instances": self.train_instances,
            "eval_instances": self.eval_instances,
            "identity_groups": self.identities,
            "train_identity_groups": self.train_identities,
            "eval_identity_groups": self.eval_identities,
            "group_pairs_sharing_a_representation": self.merged_by_state_collision,
            "largest_identity_group": self.largest_group,
            "identity_groups_holding_one_instance": self.singleton_groups,
            "eval_states_screened_out_for_appearing_in_training": (
                self.eval_states_screened_out
            ),
            "eval_states_kept": self.eval_states_kept,
            "train_strand_counts": list(self.train_strand_counts),
            "transfer_strand_counts": list(self.transfer_strand_counts),
            "guarantees": [
                "the grouping key is an invariant of the closed braid, so two "
                "representations of the same knot always share a key and can "
                "never land on opposite sides of the split",
                "whole groups move together, so every representation of a "
                "held-out identity that this corpus contains is held out",
                "the split is over whole instances, so no state of a held-out "
                "instance is used for training under any other label",
                "every evaluated state is checked against the exact set of "
                "training state keys and dropped on a collision",
            ],
            "not_guaranteed": [
                "the key is coarser than identity: distinct knots sharing an "
                "Alexander polynomial and a signature are grouped together, "
                "which makes the split safer, not weaker, but means the group "
                "count is not a count of knots",
                "CC=0 closure is NOT claimed across instances. Merging groups "
                "that share a representation was tried and collapses the corpus "
                "into one group, because every witness ends at the empty "
                "one-braid; the count of group pairs that share a representation "
                "is reported instead, and state-level leakage is handled by the "
                "exact screen rather than by the grouping",
                "every witness ends at the unknot and every instance from a seed "
                "ends in that seed's own program, so the deepest states are "
                "shared by construction -- the gate reports the walk region "
                "separately for exactly this reason",
                "the transfer row is a strand count absent from training, not a "
                "knot family absent from training",
            ],
        }


class _Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def identity_key(word: Word, strands: int) -> str:
    """A key coarser than the knot type, computed from the word alone.

    The Alexander polynomial and the signature are invariants of the closed
    braid, so two representations of the same knot always get the same key.  The
    converse fails -- distinct knots share these -- and that is the right way
    round: a coarser key merges more instances into one group and therefore holds
    out *more* than an identity-exact key would.
    """
    from rf_knots import invariants

    polynomial = invariants.alexander_polynomial(tuple(word), int(strands))
    signature = invariants.signature(tuple(word), int(strands))
    pairs = ";".join(f"{power}:{value}" for power, value in invariants.to_pairs(polynomial))
    return f"a[{pairs}]|s[{signature}]"


def assign_split(
    instances: Sequence[tuple[str, Word, int, Sequence[tuple[Word, int]]]],
    eval_fraction: float = 0.35,
    seed: int = 20260831,
) -> tuple[dict[str, str], SplitReport]:
    """Group instances by the identity of the representation they emit, then split.

    `instances` is `(instance_id, emitted word, strands, [(word, strands), ...])`
    where the last element is every state on that instance's witness.  Returns
    `{instance_id: "train" | "eval"}`.

    **Why the grouping is on the emitted representation and not on shared
    states, which was tried first and does not work.**  Merging two groups
    whenever their witnesses share a single `(word, strands)` sounds like the
    right closure, and it collapses the whole corpus into one group: every
    witness ends at the empty one-braid, so every pair of instances shares a
    state by construction.  The sharing is a property of the generator, not
    evidence that two instances are the same knot.

    So the closure the split can honestly claim is at the level of the *instance
    identity*: whole groups of instances whose emitted representations share an
    invariant move together.  Leakage through the shared tails is handled where
    it actually lives, by screening every evaluated state against the exact set
    of training state keys -- which the driver does and reports.  The number of
    groups that *would* have merged on a shared representation is measured and
    reported rather than acted on, because it is the size of that shared tail.
    """
    keys: dict[str, str] = {}
    for instance_id, word, strands, _states in instances:
        keys[instance_id] = identity_key(word, strands)
    seen: dict[tuple[Word, int], str] = {}
    would_merge: set[frozenset[str]] = set()
    for instance_id, _word, _strands, states in instances:
        key = keys[instance_id]
        for state in states:
            other = seen.setdefault(tuple(state), key)
            if other != key:
                would_merge.add(frozenset({other, key}))
    sizes = Counter(keys.values())
    groups = sorted(sizes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    #: Whole groups move together, but the *target* is a fraction of instances,
    #: not of groups.  At these sizes the generator emits a few identities over
    #: and over -- the largest group is reported -- so splitting groups evenly
    #: would put almost every instance on one side.  Groups are taken in random
    #: order until the held-out side reaches the requested share of instances.
    wanted = eval_fraction * len(keys)
    held: set[str] = set()
    accumulated = 0
    for index in order:
        if accumulated >= wanted and held:
            break
        held.add(groups[index])
        accumulated += sizes[groups[index]]
    if len(held) == len(groups) and len(groups) > 1:
        held.discard(groups[order[-1]])
    assignment = {
        instance_id: ("eval" if key in held else "train")
        for instance_id, key in keys.items()
    }
    return assignment, SplitReport(
        train_instances=sum(1 for v in assignment.values() if v == "train"),
        eval_instances=sum(1 for v in assignment.values() if v == "eval"),
        identities=len(groups),
        train_identities=len(groups) - len(held),
        eval_identities=len(held),
        merged_by_state_collision=len(would_merge),
        eval_states_screened_out=0,
        eval_states_kept=0,
        train_strand_counts=(),
        transfer_strand_counts=(),
        largest_group=max(sizes.values(), default=0),
        singleton_groups=sum(1 for value in sizes.values() if value == 1),
    )


__all__ = [
    "BASELINES",
    "LEX",
    "MOVES",
    "OBJECTIVES",
    "certified_change_floor",
    "BlockTarget",
    "block_target",
    "undo_completion",
    "move_bound",
    "GateState",
    "Ranker",
    "ScoredState",
    "ShortestCompletion",
    "Site",
    "SplitReport",
    "assign_split",
    "chance_at",
    "cluster_bootstrap",
    "completion_path",
    "exhibit_target",
    "growth_aware",
    "identity_key",
    "kind_prior",
    "order_by_scores",
    "recall_at",
    "score_state",
    "shortest_completion",
    "sites_of",
    "summarise",
    "uniform_random",
]
