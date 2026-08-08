r"""Beam over crossing-change choices, for when branch-and-bound is too slow.

`rf_knots.unknot_search` prunes with the certified bounds, which is what makes a
result a theorem -- but the bound costs a Seifert matrix and a knot Floer
homology, and it is recomputed for every distinct knot on the frontier. On the
ladder's 20-letter positive-braid rung that never finished: nine levels deep with
nothing pruned, because every candidate's bound was inside the budget and the
search was paying full price for each one.

For a **positive braid** the bound does not need to be in the loop at all.
`u = g = (L - n + 1)/2` (Kronheimer-Mrowka), and every useful crossing change
turns an adjacent `sigma_i sigma_i` into `sigma_i sigma_i^-1`, which free-reduces
and drops the length by exactly two. So *length is the signal*, and the certified
bound is needed exactly twice: once at the start to know what optimal would be,
and once at the end to say whether the sequence met it.

That is the whole difference between this and a greedy descent, which also gets
the length down but is myopic: on this rung greedy reached two letters after nine
changes and still needed a tenth, because it had landed on the wrong two-letter
word. Keeping sixty candidates per level instead of one finds the sequence that
ends at a two-letter word which simply destabilises away -- **9 changes against a
certified bound of 9, so `u = 9` is determined**, where the ratchet stood at 11.

The result is still a theorem: the witness replays, and 9 meets the bound. Only
the *search* gave up completeness, and it never claimed any.
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

from rf_knots.actions import CROSSING_CHANGE, ActionSpec
from rf_knots.evidence import UnknotWitness
from rf_knots.reference import successors
from rf_knots.unknot_search import certified_lower_bound


def reachable(spec, word, strands, budget, growth=2):
    """Type-preserving diagrams, length-first, plus a solution if one is reached."""
    seen = {(word, strands)}
    heap = [(len(word), 0, word, strands, [])]
    out, counter, cap = [], 0, len(word) + growth
    while heap and len(seen) < budget:
        length, _, current, active, path = heapq.heappop(heap)
        out.append((current, active, path))
        if not current and active == 1:
            return out, (current, active, path)
        for action, nxt_word, nxt_strands in successors(spec, current, active):
            if len(nxt_word) > cap or (nxt_word, nxt_strands) in seen:
                continue
            seen.add((nxt_word, nxt_strands))
            counter += 1
            heapq.heappush(
                heap, (len(nxt_word), counter, nxt_word, nxt_strands, [*path, action])
            )
    return out, None


def solve(word, strands, *, beam=60, per_level=120, node_budget=6000,
          max_changes=None, log=lambda *_: None):
    spec = ActionSpec(len(word) + 6, max(strands + 2, 5))
    bound = certified_lower_bound(word, strands)
    limit = max_changes if max_changes is not None else bound + 2
    log(f"certified lower bound {bound}, searching to {limit} changes")
    level = [(tuple(word), strands, [])]
    started = time.perf_counter()
    for step in range(limit + 1):
        candidates, seen = [], set()
        for current, active, actions in level:
            diagrams, solved = reachable(spec, current, active, node_budget)
            if solved is not None:
                witness = UnknotWitness.from_actions(
                    word, strands, spec, actions + solved[2]
                )
                witness.verify()
                return witness, bound
            for cur, st, path in sorted(diagrams, key=lambda d: len(d[0]))[:per_level]:
                for position in range(len(cur)):
                    flipped = cur[:position] + (-cur[position],) + cur[position + 1:]
                    if (flipped, st) in seen:
                        continue
                    seen.add((flipped, st))
                    candidates.append(
                        (flipped, st,
                         actions + path + [spec.encode(CROSSING_CHANGE, position)])
                    )
        candidates.sort(key=lambda item: (len(item[0]), item[1]))
        level = candidates[:beam]
        log(f"  after {step + 1} changes: {len(candidates)} candidates, shortest "
            f"{len(level[0][0]) if level else '-'} letters "
            f"({time.perf_counter() - started:.0f}s)")
        if not level:
            break
    return None, bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", required=True, help="comma-separated braid word")
    parser.add_argument("--strands", type=int, required=True)
    parser.add_argument("--beam", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    word = tuple(int(x) for x in args.word.split(",") if x.strip())
    witness, bound = solve(word, args.strands, beam=args.beam, log=print)
    if witness is None:
        print(f"no sequence found; certified lower bound is {bound}")
        return
    exact = witness.crossing_changes == bound
    print(f"{witness.crossing_changes} crossing changes, {witness.moves} moves, "
          f"bound {bound}, {'u DETERMINED' if exact else 'upper bound only'}")
    if args.output:
        args.output.write_text(json.dumps(
            {"crossing_changes": witness.crossing_changes, "moves": witness.moves,
             "lower_bound": bound, "exact": exact, "witness": witness.to_dict()},
            indent=2))


if __name__ == "__main__":
    main()
