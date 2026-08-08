r"""Branch-and-bound over crossing changes, pruned by the certified lower bounds.

[research/10 §1C](../../research/10-invariants-and-representations.md) calls this
"the single highest-leverage change available", and it is the piece that turns a
search result into mathematics. The argument is short:

* `u(K)` is the graph distance from `K` to the unknot in the crossing-change
  graph, so a search for a short unknotting sequence is a shortest-path problem.
* A certified lower bound `L(K)` is an **admissible heuristic** for that
  distance — it never overestimates, because it is a theorem.
* So a branch whose `changes_used + L(current knot)` exceeds the best sequence
  already found can be cut without losing an optimum.
* And when a found sequence has length exactly `L(start)`, the node is **solved
  exactly**: `u(K) = L(start)` is then a theorem rather than an estimate.

## The shape of the search, and why it is two-level

Markov and braid-group moves change the diagram but not the knot, so the
certified bound is *constant* along them — and computing it costs a Seifert
matrix and a knot Floer homology, which is far too slow to do per node. The
search is therefore levelled by crossing-change count:

```
level k:  explore the diagram graph with type-preserving moves only
          (the bound is fixed here, computed once)
          |
          | one crossing change
          v
level k+1: a different knot, so recompute the bound and prune
```

Within a level the exploration is an ordinary breadth-first search over the
type-preserving moves, bounded by node count and by how far the word is allowed
to grow. Across levels the frontier is deduplicated **by knot fingerprint** rather
than by diagram, since two diagrams of the same knot are the same vertex of the
graph the search is really in — the transposition-table idea from
[10 §1A](../../research/10-invariants-and-representations.md).

## What it returns

A replayable `UnknotWitness` (from `rf_knots.evidence`), which is a sequence of
legal actions that `verify()` replays through the reference implementation and
checks reaches the empty 1-braid. A bound without one of these is an assertion;
with one it is checkable by anybody.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness
from rf_knots.reference import successors

Word = tuple[int, ...]
State = tuple[Word, int]


@dataclass
class SearchReport:
    """Everything needed to quote the result honestly, including the failures."""

    witness: UnknotWitness | None
    crossing_changes: int | None
    lower_bound: int | None
    exact: bool
    diagrams_explored: int
    knots_expanded: int
    pruned_by_bound: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def solved_exactly(self) -> bool:
        """`u` is determined: the sequence found meets the certified bound."""
        return self.exact


def _fingerprint(word: Word, strands: int) -> tuple:
    """A cheap knot invariant tuple, for deduplicating across diagrams.

    The Alexander polynomial would be a sharper key, but it is ~10 ms and this
    runs on every candidate flip; at a few hundred diagrams times twenty positions
    per level that is the difference between a minute and an afternoon. The
    determinant alone is one Burau evaluation and separates enough.

    Collisions merely merge two branches that were probably the same knot anyway.
    A wrong merge can lose a route but can never invent one, because every witness
    returned is replay-verified against the reference implementation.
    """
    from rf_knots.invariants import determinant

    if not word:
        return (0, strands)
    return (1, determinant(word, strands))


def certified_lower_bound(word: Word, strands: int) -> int:
    """`max` over the certified bounds. Zero when none is available."""
    if not word:
        return 0
    from rf_knots.lower_bounds import computed_claims, strongest

    return strongest(computed_claims(word, strands)) or 0


def _explore_level(
    spec: ActionSpec,
    roots: list[tuple[State, list[int]]],
    node_budget: int,
    growth: int,
) -> tuple[list[tuple[State, list[int]]], State | None, list[int] | None]:
    """Best-first over type-preserving moves from every root at this level.

    Ordered by **word length, then strand count**, not by breadth. That matters:
    the interesting rungs arrive as long words -- `R(3,18)#0` is eighteen letters
    for a seven-crossing knot -- and a plain breadth-first search spends its whole
    node budget on the eighteen-letter neighbourhood without ever simplifying.
    Length-first drives straight at short representatives, which are both the ones
    worth flipping a crossing on and the ones a solution has to pass through.

    Returns the reachable diagrams with the path that reached each, plus the
    solved state and its path if the empty 1-braid was found without spending
    another crossing change.
    """
    import heapq

    seen: set[State] = set()
    heap: list[tuple[int, int, int, State, list[int]]] = []
    counter = 0
    for state, path in roots:
        if state not in seen:
            seen.add(state)
            heapq.heappush(heap, (len(state[0]), state[1], counter, state, path))
            counter += 1
    reached: list[tuple[State, list[int]]] = []
    cap = max((len(state[0]) for state, _ in roots), default=0) + growth

    while heap and len(seen) < node_budget:
        _, _, _, (word, strands), path = heapq.heappop(heap)
        reached.append(((word, strands), path))
        if not word and strands == 1:
            return reached, (word, strands), path
        for action, next_word, next_strands in successors(spec, word, strands):
            if len(next_word) > cap:
                continue
            nxt = (next_word, next_strands)
            if nxt in seen:
                continue
            seen.add(nxt)
            counter += 1
            heapq.heappush(
                heap, (len(next_word), next_strands, counter, nxt, [*path, action])
            )
    return reached, None, None


def search(
    word: Word,
    strands: int,
    *,
    max_crossing_changes: int = 3,
    node_budget: int = 20_000,
    growth: int = 2,
    frontier_width: int = 24,
    flip_from: int = 150,
    max_len: int | None = None,
    log=lambda *_: None,
) -> SearchReport:
    """Find a short unknotting sequence, pruning with the certified bounds.

    `flip_from` caps how many diagrams per level are used as crossing-change
    sites, taking the shortest; `frontier_width` caps how many distinct knots are
    carried to the next level,
    keeping the *strongest-bounded* ones — a beam, so the search is complete only
    up to that width. Any witness returned is exact evidence for an upper bound
    whatever the width; only the claim "no shorter sequence exists" would need
    completeness, and this function never makes that claim.
    """
    letters = tuple(int(x) for x in word if int(x))
    spec = ActionSpec(max_len or len(letters) + growth + 4, max(strands, 2))

    start_bound = certified_lower_bound(letters, strands)
    log(f"start: {len(letters)} letters on {strands} strands, "
        f"certified lower bound {start_bound}")
    if start_bound > max_crossing_changes:
        return SearchReport(None, None, start_bound, False, 0, 0, 0,
                            ("the certified bound already exceeds the budget",))

    diagrams = knots = pruned = 0
    level: list[tuple[State, list[int]]] = [((letters, strands), [])]
    seen_knots: set[tuple] = set()

    for used in range(max_crossing_changes + 1):
        reached, solved, path = _explore_level(spec, level, node_budget, growth)
        diagrams += len(reached)
        log(f"level {used}: {len(reached)} diagrams from {len(level)} roots")
        if solved is not None:
            witness = UnknotWitness.from_actions(letters, strands, spec, path)
            exact = witness.crossing_changes == start_bound
            return SearchReport(witness, witness.crossing_changes, start_bound,
                                exact, diagrams, knots, pruned)
        if used == max_crossing_changes:
            break

        # One crossing change from the *shortest* diagrams reached at this level.
        # Flipping from all of them is what makes this intractable: 30,000
        # diagrams times twenty positions is 600,000 invariant computations per
        # level. Short diagrams are also the useful ones -- a crossing change on a
        # word that has been inflated by insertions is almost always undone by
        # free reduction -- so the cap costs far less than it saves.
        ordered_by_length = sorted(reached, key=lambda item: len(item[0][0]))[:flip_from]
        candidates: dict[tuple, tuple[State, list[int], int]] = {}
        flipped_seen: set[Word] = set()
        for (current, current_strands), path in ordered_by_length:
            for position in range(len(current)):
                flipped = (
                    current[:position] + (-current[position],) + current[position + 1:]
                )
                if flipped in flipped_seen:
                    continue
                flipped_seen.add(flipped)
                key = _fingerprint(flipped, current_strands)
                if key in seen_knots or key in candidates:
                    continue
                bound = certified_lower_bound(flipped, current_strands)
                knots += 1
                if used + 1 + bound > max_crossing_changes:
                    pruned += 1
                    continue
                action = spec.encode_crossing_change(position) if hasattr(
                    spec, "encode_crossing_change"
                ) else _crossing_change_action(spec, position)
                candidates[key] = (
                    (flipped, current_strands), [*path, action], bound
                )
        if not candidates:
            log(f"level {used}: every continuation pruned by the certified bound")
            break
        # Keep the most promising: lowest remaining bound, then shortest word.
        ordered = sorted(candidates.items(), key=lambda kv: (kv[1][2], len(kv[1][0][0])))
        seen_knots.update(key for key, _ in ordered[:frontier_width])
        level = [(state, path) for _, (state, path, _) in ordered[:frontier_width]]
        log(f"level {used} -> {used + 1}: {len(candidates)} distinct knots, "
            f"{pruned} pruned, carrying {len(level)}")

    return SearchReport(None, None, start_bound, False, diagrams, knots, pruned,
                        ("no sequence found within the budget",))


def _crossing_change_action(spec: ActionSpec, position: int) -> int:
    from rf_knots.actions import CROSSING_CHANGE

    return spec.encode(CROSSING_CHANGE, position)
