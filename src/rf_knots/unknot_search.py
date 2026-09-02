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
    # The two growth dials, recorded so a run can be quoted without its caller.
    growth: int = 0
    strand_growth: int = 0
    strand_cap: int | None = None
    length_growth: int = 0
    width_penalty: int = 0
    # Cost and drift, filled from `_explore_level`'s counters.
    expansions: int = 0
    successors_generated: int = 0
    successors_admitted: int = 0
    stabilizations_generated: int = 0
    destabilizations_generated: int = 0
    max_strands_expanded: int = 0
    expansions_above_source_width: int = 0
    flip_sites: int = 0
    flip_sites_above_source_width: int = 0
    witness_stabilizations: int = 0
    # The flip-site quota (note 22 §3.6's "one cheap experiment first").
    flip_growth_quota: float = 0.0
    flip_quota_slots: int = 0
    flip_quota_filled: int = 0
    flip_quota_levels: int = 0
    flip_quota_levels_filled: int = 0
    # Whether the `flip_from` cut is even binding: reserving slots can only
    # matter at a level where there were more diagrams than slots.
    flip_reached: int = 0
    flip_levels: int = 0
    flip_levels_cut: int = 0

    @property
    def solved_exactly(self) -> bool:
        """`u` is determined: the sequence found meets the certified bound."""
        return self.exact

    @property
    def branching_factor(self) -> float:
        """Legal successors generated per node expanded, over the whole search."""
        return self.successors_generated / self.expansions if self.expansions else 0.0

    @property
    def admitted_branching_factor(self) -> float:
        """Successors that survived the length and strand caps, per expansion."""
        return self.successors_admitted / self.expansions if self.expansions else 0.0

    @property
    def flip_quota_fill_rate(self) -> float:
        """Fraction of the reserved flip slots that a widened diagram existed for.

        Note 22 §3.6 names the third possible outcome of the quota experiment:
        *the quota cannot be filled*, because too few widened diagrams reach the
        frontier at all.  That is a frontier problem rather than a selection
        problem, and this is the number that tells the two apart.
        """
        return (
            self.flip_quota_filled / self.flip_quota_slots
            if self.flip_quota_slots
            else 0.0
        )

    @property
    def width_drift(self) -> float:
        """Fraction of expansions spent on braids wider than the source.

        This is the number the growth policy has to keep small.  An unconstrained
        walk ratchets -- `STABILIZE` is nearly always legal and `DESTABILIZE`
        nearly never is -- so if enabling growth is not to make the search
        strictly worse, the frontier must keep coming back down.
        """
        return (
            self.expansions_above_source_width / self.expansions
            if self.expansions
            else 0.0
        )


def _fingerprint(word: Word, strands: int) -> tuple:
    """A cheap knot invariant tuple, for deduplicating across diagrams.

    The determinant separates enough, and it runs on every candidate flip, so
    *which route computes it* is the hot path of this whole search.

    An earlier version of this docstring argued that "the determinant alone is
    one Burau evaluation" and cost about 10 ms. Both halves were wrong, and the
    measurement is the reason this now prefers the Seifert route.
    `invariants.determinant` builds the whole Alexander polynomial -- it costs
    the same as `alexander_polynomial` to within 3% -- while
    `seifert.determinant` reads one Smith-normal-form invariant off the Seifert
    matrix. Warmed up, on random words:

    | letters/strands | `invariants` | `seifert` | ratio |
    |---|---:|---:|---:|
    | 20/5  |   21.2 ms | 1.2 ms |  18x |
    | 60/5  |  905.2 ms | 9.7 ms |  93x |
    | 80/12 | 3926.4 ms | 15.9 ms | 247x |

    and on the 254-letter target, 647 s against 0.42 s -- 1,539x for the same
    answer, which is one search level in 45.7 hours rather than 107 seconds.
    There is no crossover: the Seifert route wins at every shape measured.
    (Time it warm. A single cold call charges spherogram's import to the Seifert
    side and reverses the verdict.)

    Both routes return the same value, so the keys are identical and this is a
    pure speedup. The fallback is therefore free of consequence: `seifert` needs
    the optional `spherogram` backend and raises on some split-link words, and in
    either case the Burau route computes the same key more slowly.

    Collisions merely merge two branches that were probably the same knot anyway.
    A wrong merge can lose a route but can never invent one, because every witness
    returned is replay-verified against the reference implementation.
    """
    if not word:
        return (0, strands)
    from rf_knots import seifert

    try:
        return (1, seifert.determinant(word, strands))
    except (ValueError, KeyError, RuntimeError, seifert.BackendUnavailable):
        from rf_knots.invariants import determinant

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
    *,
    strand_cap: int | None = None,
    source_strands: int | None = None,
    width_penalty: int = 0,
    stats: dict | None = None,
) -> tuple[list[tuple[State, list[int]]], State | None, list[int] | None]:
    """Best-first over type-preserving moves from every root at this level.

    Ordered by **word length, then strand count**, not by breadth. That matters:
    the interesting rungs arrive as long words -- `R(3,18)#0` is eighteen letters
    for a seven-crossing knot -- and a plain breadth-first search spends its whole
    node budget on the eighteen-letter neighbourhood without ever simplifying.
    Length-first drives straight at short representatives, which are both the ones
    worth flipping a crossing on and the ones a solution has to pass through.

    ## Two caps, not one

    `growth` bounds the **word length** relative to the roots. `strand_cap` bounds
    the **strand count** absolutely, and is the second half of the same idea: the
    length cap alone never stopped `STABILIZE`, because until this parameter
    existed the caller built the `ActionSpec` at the source's own strand count and
    `STABILIZE` was illegal from the first node. Passing `strand_cap=None` keeps
    whatever the spec permits.

    ## The ratchet, and what actually holds it

    `STABILIZE` is legal at almost every node and `DESTABILIZE` almost nowhere --
    it needs the top generator to occur exactly once -- so a walk that treats the
    two symmetrically widens monotonically and never comes back.

    Length-first ordering is *not* enough on its own. `STABILIZE` raises both keys
    by one, so a widened diagram is deferred by one letter's worth of priority and
    no more; once the narrow diagrams within the length cap run out, every
    remaining expansion is wide. Measured on `5_1` at `strand_growth=1`, that is
    97.8% of expansions above the source width.

    `width_penalty` is the control: each strand above `source_strands` costs that
    many letters of priority. The charge is on the *state*, not the path, so
    destabilising back to the source width refunds it in full -- a
    stabilise/manoeuvre/destabilise round trip that shortens the word is rewarded
    the moment it closes, while sitting wide is paid for at every expansion. It is
    inert at `strand_growth=0`, where every node has the same width.
    `stats["expansions_above_source_width"]` reports what the policy achieved.

    Returns the reachable diagrams with the path that reached each, plus the
    solved state and its path if the empty 1-braid was found without spending
    another crossing change.
    """
    import heapq

    from rf_knots.actions import DESTABILIZE, STABILIZE_NEG, STABILIZE_POS

    seen: set[State] = set()
    heap: list[tuple[int, int, int, State, list[int]]] = []
    counter = 0
    if source_strands is None:
        source_strands = min((state[1] for state, _ in roots), default=0)

    def key(word: Word, strands: int) -> int:
        return len(word) + width_penalty * max(0, strands - source_strands)

    for state, path in roots:
        if state not in seen:
            seen.add(state)
            heapq.heappush(
                heap, (key(*state), state[1], counter, state, path)
            )
            counter += 1
    reached: list[tuple[State, list[int]]] = []
    cap = max((len(state[0]) for state, _ in roots), default=0) + growth
    stabilize_actions = {spec.encode(STABILIZE_POS), spec.encode(STABILIZE_NEG)}
    destabilize_action = spec.encode(DESTABILIZE)
    expansions = generated = admitted = 0
    stabilizations = destabilizations = 0
    widest = 0
    wide_expansions = 0

    while heap and len(seen) < node_budget:
        _, _, _, (word, strands), path = heapq.heappop(heap)
        reached.append(((word, strands), path))
        expansions += 1
        widest = max(widest, strands)
        if strands > source_strands:
            wide_expansions += 1
        if not word and strands == 1:
            break
        for action, next_word, next_strands in successors(spec, word, strands):
            generated += 1
            if action in stabilize_actions:
                stabilizations += 1
            elif action == destabilize_action:
                destabilizations += 1
            if len(next_word) > cap:
                continue
            if strand_cap is not None and next_strands > strand_cap:
                continue
            admitted += 1
            nxt = (next_word, next_strands)
            if nxt in seen:
                continue
            seen.add(nxt)
            counter += 1
            heapq.heappush(
                heap,
                (key(next_word, next_strands), next_strands, counter, nxt,
                 [*path, action]),
            )

    if stats is not None:
        stats["expansions"] = stats.get("expansions", 0) + expansions
        stats["successors_generated"] = (
            stats.get("successors_generated", 0) + generated
        )
        stats["successors_admitted"] = stats.get("successors_admitted", 0) + admitted
        stats["stabilizations_generated"] = (
            stats.get("stabilizations_generated", 0) + stabilizations
        )
        stats["destabilizations_generated"] = (
            stats.get("destabilizations_generated", 0) + destabilizations
        )
        stats["max_strands_expanded"] = max(
            stats.get("max_strands_expanded", 0), widest
        )
        stats["expansions_above_source_width"] = (
            stats.get("expansions_above_source_width", 0) + wide_expansions
        )

    solved = reached[-1][0] if reached and reached[-1][0] == ((), 1) else None
    if solved is not None:
        return reached, solved, reached[-1][1]
    return reached, None, None


def search(
    word: Word,
    strands: int,
    *,
    max_crossing_changes: int = 3,
    node_budget: int = 20_000,
    growth: int = 2,
    strand_growth: int = 0,
    width_penalty: int = 2,
    frontier_width: int = 24,
    flip_from: int = 150,
    flip_growth_quota: float = 0.0,
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

    ## The two growth dials

    `growth` is the **length** headroom, relative to the roots of each level.
    `strand_growth` is the **width** headroom, absolute and relative to the
    source: the braid may reach `strands + strand_growth` strands and no more.

    `strand_growth=0` is the default and reproduces the search exactly as it
    behaved before this parameter existed — the `ActionSpec` is built at the
    source's own strand count, so `STABILIZE_POS`/`STABILIZE_NEG` (which take
    `n -> n+1` and are legal only while `n < spec.max_strands`) are illegal from
    the first node and the braid can never widen. That was not a policy; it was
    an oversight, and note 22 §2.3 / §M1b measured what it costs: the published
    witness for `7_1 # m(7_1)` creates crossings sixteen times, and the 20-letter
    entry word admits only two of its five crossing changes without inflating.

    A stabilisation also costs one letter, so with `strand_growth > 0` the length
    cap is raised by the same amount: widening is charged against its own budget
    rather than silently eating the insertion budget, and the two dials stay
    independent. Both are recorded on the returned `SearchReport`.

    ## The growth policy, in one sentence

    **Cap the width absolutely, and charge every strand above the source
    `width_penalty` letters of frontier priority, refunded on the way back down.**
    The cap stops the braid running away; the charge stops the frontier *sitting*
    wide, because `DESTABILIZE` is nearly never legal while `STABILIZE` nearly
    always is and length-first ordering alone defers a widened diagram by exactly
    one letter (measured: 97.8% of expansions above source width on `5_1` at
    `width_penalty=0`). All three of `growth`, `strand_growth` and
    `width_penalty` are recorded on the report, and `width_penalty` is inert
    whenever `strand_growth=0`, so no existing caller changes.

    ## The flip-site quota

    `flip_growth_quota` reserves `int(flip_from * quota)` of the crossing-change
    slots at each level for diagrams **above the source width**, filling the rest
    by length exactly as before.  It exists because growth turned out to be
    unusable for a reason that is neither the branching factor nor the frontier
    policy: note 22 §3.6 measured `flip_sites_above_source_width` at **0 of
    7,500** on the target and **0 of 6,066** on the hard sweep subset, because a
    stabilised diagram is at least one letter longer than the one it came from
    and there are always more than `flip_from` shorter ones.  The one move the
    growth work made legal could never reach the stage it was made legal for.

    That is the same length-monotone bias §Status consequence 3 disqualified for
    *search*, surviving one stage later as a length-monotone *selection*.  The
    quota is the cheapest possible test of whether flipping a widened diagram
    helps at all: no training, one selection rule.

    The default `0.0` reserves nothing, so the selection is the same list of
    indices in the same order and every existing caller is unchanged.  Three
    counters say what happened: `flip_sites_above_source_width` (did the quota do
    what it says), `flip_quota_slots`/`flip_quota_filled` (were there even enough
    widened diagrams to fill it -- a quota that cannot be filled is a statement
    about the frontier, not about the selection).
    """
    letters = tuple(int(x) for x in word if int(x))
    strand_growth = max(int(strand_growth), 0)
    flip_growth_quota = min(max(float(flip_growth_quota), 0.0), 1.0)
    strand_cap = max(strands, 2) + strand_growth
    # One stabilisation costs one letter as well as one strand, so the length cap
    # is opened by exactly the strand budget.  At `strand_growth=0` this is `growth`
    # and every existing caller is bit-for-bit unchanged.
    length_growth = growth + strand_growth
    spec = ActionSpec(max_len or len(letters) + length_growth + 4, strand_cap)

    def _report(*args, **kwargs) -> SearchReport:
        report = SearchReport(*args, **kwargs)
        report.growth = growth
        report.strand_growth = strand_growth
        report.strand_cap = strand_cap
        report.length_growth = length_growth
        report.width_penalty = width_penalty
        report.expansions = stats.get("expansions", 0)
        report.successors_generated = stats.get("successors_generated", 0)
        report.successors_admitted = stats.get("successors_admitted", 0)
        report.stabilizations_generated = stats.get("stabilizations_generated", 0)
        report.destabilizations_generated = stats.get(
            "destabilizations_generated", 0
        )
        report.max_strands_expanded = stats.get("max_strands_expanded", strands)
        report.expansions_above_source_width = stats.get(
            "expansions_above_source_width", 0
        )
        report.flip_sites = stats.get("flip_sites", 0)
        report.flip_sites_above_source_width = stats.get(
            "flip_sites_above_source_width", 0
        )
        report.flip_growth_quota = flip_growth_quota
        report.flip_quota_slots = stats.get("flip_quota_slots", 0)
        report.flip_quota_filled = stats.get("flip_quota_filled", 0)
        report.flip_quota_levels = stats.get("flip_quota_levels", 0)
        report.flip_quota_levels_filled = stats.get("flip_quota_levels_filled", 0)
        report.flip_reached = stats.get("flip_reached", 0)
        report.flip_levels = stats.get("flip_levels", 0)
        report.flip_levels_cut = stats.get("flip_levels_cut", 0)
        if report.witness is not None:
            report.witness_stabilizations = sum(
                step.action.kind in ("STABILIZE_POS", "STABILIZE_NEG")
                for step in report.witness.steps
            )
        return report

    stats: dict = {}
    start_bound = certified_lower_bound(letters, strands)
    log(f"start: {len(letters)} letters on {strands} strands, "
        f"certified lower bound {start_bound}")
    if start_bound > max_crossing_changes:
        return _report(None, None, start_bound, False, 0, 0, 0,
                       ("the certified bound already exceeds the budget",))

    diagrams = knots = pruned = 0
    level: list[tuple[State, list[int]]] = [((letters, strands), [])]
    seen_knots: set[tuple] = set()

    for used in range(max_crossing_changes + 1):
        reached, solved, path = _explore_level(
            spec, level, node_budget, length_growth,
            strand_cap=strand_cap, source_strands=strands,
            width_penalty=width_penalty, stats=stats,
        )
        diagrams += len(reached)
        log(f"level {used}: {len(reached)} diagrams from {len(level)} roots")
        if solved is not None:
            witness = UnknotWitness.from_actions(letters, strands, spec, path)
            exact = witness.crossing_changes == start_bound
            return _report(witness, witness.crossing_changes, start_bound,
                           exact, diagrams, knots, pruned)
        if used == max_crossing_changes:
            break

        # One crossing change from the *shortest* diagrams reached at this level.
        # Flipping from all of them is what makes this intractable: 30,000
        # diagrams times twenty positions is 600,000 invariant computations per
        # level. Short diagrams are also the useful ones -- a crossing change on a
        # word that has been inflated by insertions is almost always undone by
        # free reduction -- so the cap costs far less than it saves.
        # Note 22 §3.6: the count of flip sites above the source width is what
        # says whether growth ever reached the stage it was enabled for, and it
        # was measured at **0 of 7,500** on the target because this selection
        # takes the shortest diagrams and a stabilised diagram is always longer.
        # `flip_growth_quota` reserves a fraction of the slots for widened
        # diagrams; at the default `0.0` not a single index changes and
        # `strand_growth=0` remains bit-for-bit what it was.
        by_length = sorted(range(len(reached)),
                           key=lambda i: len(reached[i][0][0]))
        stats["flip_reached"] = stats.get("flip_reached", 0) + len(reached)
        stats["flip_levels"] = stats.get("flip_levels", 0) + 1
        if len(reached) > flip_from:
            stats["flip_levels_cut"] = stats.get("flip_levels_cut", 0) + 1
        slots = int(flip_from * flip_growth_quota)
        reserved: list[int] = []
        if slots:
            reserved = [i for i in by_length if reached[i][0][1] > strands][:slots]
            stats["flip_quota_slots"] = stats.get("flip_quota_slots", 0) + slots
            stats["flip_quota_filled"] = (
                stats.get("flip_quota_filled", 0) + len(reserved)
            )
            stats["flip_quota_levels"] = stats.get("flip_quota_levels", 0) + 1
            if len(reserved) == slots:
                stats["flip_quota_levels_filled"] = (
                    stats.get("flip_quota_levels_filled", 0) + 1
                )
        held = set(reserved)
        rest = [i for i in by_length if i not in held][:flip_from - len(reserved)]
        # Keep the union in length order, so the narrow sites keep the dedup
        # priority they have today and the reserved wide ones can only *add*
        # crossing changes that were not already available.
        chosen = set(rest) | held
        ordered_by_length = [reached[i] for i in by_length if i in chosen]
        stats["flip_sites"] = stats.get("flip_sites", 0) + len(ordered_by_length)
        stats["flip_sites_above_source_width"] = (
            stats.get("flip_sites_above_source_width", 0)
            + sum(1 for (_, n), _ in ordered_by_length if n > strands)
        )
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

    return _report(None, None, start_bound, False, diagrams, knots, pruned,
                   ("no sequence found within the budget",))


def _crossing_change_action(spec: ActionSpec, position: int) -> int:
    from rf_knots.actions import CROSSING_CHANGE

    return spec.encode(CROSSING_CHANGE, position)
