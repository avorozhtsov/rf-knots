# Single-knot mastery: reducing open unknotting upper bounds

## Outcome and scope

This program searches for replayable witnesses that lower the upper endpoint of
an open KnotInfo interval. It does not infer a knot-type bound from a classifier
score. A new upper bound is recorded only when a complete semantic action path
from a declared braid representation to the empty one-braid passes
`UnknotWitness.verify()`.

The pinned 2026-08-14 KnotInfo snapshot contains 482 requested open intervals:

| interval | knots | locally stored braid | immediate target |
|---|---:|---:|---:|
| `[1,3]` | 409 | 37 | `u <= 2` |
| `[1,4]` | 6 | 0 | `u <= 3` |
| `[2,4]` | 67 | 14 | `u <= 3` |
| total | 482 | 51 | — |

Another five knots are present in the local source catalogue but were omitted
by its historical strand cap. The remaining 426 are 13-crossing knots outside
the bundled at-most-12-crossing braid table. The complete provenance-bearing
list is
[`benchmarks/knotinfo-unknotting-gap-candidates-20260814.json`](../benchmarks/knotinfo-unknotting-gap-candidates-20260814.json).

## First targets

`12a_815` is the capacity smoke test: interval `[2,4]`, target 3, three strands,
and a 12-letter stored word. The first scientifically stronger pilot should be
`12n_140` or `12n_859`: both have interval `[1,3]`, target 2, five strands, and
14-letter stored words. A two-crossing-change witness would lower the public
upper bound from 3 to 2.

The preferred inherited scientist is `strand-graph` with its strongest verified
factorized-head checkpoint. It led the last inspected interim of the first
12-strand Q gate on retained capped objective quality, and its representation is
naturally compatible with changing strand count. The runner accepts an explicit
scientist and checkpoint rather
than silently selecting whichever live file happens to be newest. A five-strand
checkpoint can start the low-capacity subset; the migrated 12-strand checkpoint
covers all 51 stored local candidates after its own retention/capacity gate.

## Search protocol

For a current node at crossing-change distance `d` and knot target `T`, define

```text
remaining_cc = T - d
cap = 1000 * remaining_cc + move_allowance
P_T(node) = P(this frozen solver succeeds | representation, encoded cap)
heap key = (-P_T(node), predicted L1000, age, stable node id)
```

This is an operational probability about the current solver, not the
probability that a mathematical witness exists. A low score may mean the
solver is weak, not that the representation is impossible.

The negative sign makes the smallest heap key the largest probability. Thus the
frequently refreshed smallest keys requested by the protocol are precisely the
current leaders. Every refresh also takes nodes from a deterministic aging
cursor, so every live key is eventually recomputed even if it never becomes a
leader.

One coordinator iteration is:

1. Refresh the top and fairness-selected heap keys under the current network.
2. Pop `K` representations and run `K` batched MCTS lanes. The lanes share one
   network, while each has an independent stochastic seed and objective-budget
   state.
3. Replay-verify every claimed solution. Train the solve head from an
   approximately 50:50 success/failure replay mixture. Failed MCTS policies are
   not imitation targets, but their conditional solve loss still
   backpropagates through the shared encoder. A failure is admitted only after
   at least three independent seeds fail at the same representation, encoded
   budget, and simulation dose.
4. From each selected node, explore a bounded neighbourhood using only
   type-preserving braid/Markov moves. Prefer its shortest diagrams, apply one
   crossing change at every eligible site, deduplicate, score, and insert the
   resulting variants into the heap.
5. If a witness with `c` crossing changes is verified, retain the best L1000
   witness at that `c`, set `T = c - 1`, prune nodes whose spent distance exceeds
   the new target, and rescore the surviving heap.
6. Stop once `T` is below a certified lower bound. Without a certified lower
   bound, failure to find a smaller witness is not a proof of exactness.

### Outcome and budget controller

The scientific target and the training target are distinct. At least half of
MCTS lanes always use the strict scientific target `T`. The remaining lanes may
use `T + slack`, which changes the objective-budget channel and the conditional
label seen by the solve head. A rolling controller increases `slack` when the
success fraction is below 0.50 and decreases it when the fraction is above
0.50. Relaxed successes are valid training data, but they ratchet the knot bound
only when their replayed total crossing-change count also satisfies the strict
target.

### Rehearsal and retention

Verified inherited solutions occupy 25% of each replay batch initially. The
other positive quarter is current-knot data and the remaining half is composed
of independently confirmed negatives. A frozen rehearsal panel is evaluated at
a fixed simulation dose and fixed seeds every 20 coordinator steps. If its
solve rate falls below 0.80, the absolute rehearsal fraction rises in 0.05
increments, capped at 0.50. This protects inherited skill without allowing
rehearsal to remove negative supervision.

### Simulation-dose controller

Simulation count is treated as a paired compute dose rather than an arbitrary
constant. At a fixed cadence, the same representations and random seeds are
evaluated at adjacent doses, by default `S`, `2S`, and `4S`. The controller uses
paired success differences and paired L1000 differences with a 95% lower
confidence bound. It promotes to the next dose only when the success advantage
exceeds 0.05 or the L1000 improvement exceeds 5. The selected dose, probe panel,
paired seeds, confidence bounds, and network evaluations are logged. Easy
rehearsals remain on the fixed baseline dose, so retention is not confounded by
changing search compute.

## Evidence index and mastery sequence

[`benchmarks/unknotting-evidence-index-20260815.json`](../benchmarks/unknotting-evidence-index-20260815.json)
joins 546 archived Semantic-v2 round events to their frozen banks. Every stored
semantic action path is replayed from the exact bank word before admission. The
index currently contains verified native evidence for 100 knot identities and
reported zero replay failures.

For each scientist, “best” means the smallest crossing-change count and then
the smallest L10. A curriculum candidate must have a native witness longer than
50 moves whose crossing-change count strictly beats at least one peer in the
same event. Eligible knots are then ordered by L10. The first block is:

| # | knot | source scientist | cc | moves | L10 | best worse peer cc |
|---:|---|---|---:|---:|---:|---:|
| 1 | `4_1` | `raster-invariant-combined-film` | 1 | 101 | 111 | 3 |
| 2 | `8_21` | `cyclic-memory-12` | 3 | 89 | 119 | 5 |
| 3 | `12a_1246` | `cyclic-memory` | 5 | 70 | 120 | 6 |
| 4 | `8_20` | `cyclic-memory-12` | 3 | 93 | 123 | 4 |
| 5 | `11a_40` | `cyclic-memory` | 4 | 85 | 125 | 6 |
| 6 | `12n_684` | `cyclic-memory` | 3 | 99 | 129 | 4 |
| 7 | `11a_47` | `cyclic-memory` | 4 | 89 | 129 | 5 |
| 8 | `10_141` | `strand-graph` | 3 | 101 | 131 | 5 |
| 9 | `10_159` | `cyclic-memory` | 1 | 122 | 132 | 5 |
| 10 | `3_1` | `strand-graph` | 1 | 125 | 135 | 2 |

The next native five are `10_123`, `11a_170`, `12a_1288`, `11a_26`, and
`11a_79`. The requested five KnotInfo examples remain unfilled. The pinned
workbook has no replayable action paths, so the pseudo-scientist
`knotinfo-shortest-evidence` is present in the index with status `unavailable`
and is not rankable by L10. Those slots become eligible only after complete
external paths are imported and replay-verified.

Multiple initial representations are generated by bounded type-preserving walks
from every supplied braid. Their paths are stored, so a solution from any
generated start composes into one replayable witness from its declared root.
Externally supplied representations must carry provenance asserting why they are
representations of the named knot; the system never treats a shared name as a
proof.

The one-crossing-change frontier is finite only relative to the stored/reached
diagrams and the declared exploration bounds. It must not be described as the
complete Gordian neighbourhood of the knot type.

## Human-evidence distillation

External KnotInfo or paper witnesses enter through a separate file and are
strictly replayed before admission. They are ordered by

```text
L10 = 10 * crossing_changes + semantic_moves
```

Distillation remains off until the scientist has spent a declared number of
native attempts at the current target without reaching it. Even then, distilled
updates are bounded by a cumulative fraction of native updates (10% by default).
This prevents long, alien controller paths from dominating the representation.
Only portable semantic braid actions are distilled; UI operations, foreign
controller states, and unverified endpoint claims are rejected.

The pinned KnotInfo workbook itself contains no replayable action paths for
these 482 targets. It has explanatory-reference links on five rows; the catalogue
retains them as provenance but marks all 482 rows non-distillable. A scalar
interval or literature link is not a demonstration. Actual distillation data
must be collected from a source that exposes a complete path and must pass the
same semantic replay verifier as a native solution.

This is intentionally an AlphaGo-style optional knowledge channel around an
AlphaZero-style default. Every report must separate native and distilled
solutions and training doses.

## Implementation

The engine is
[`single_knot_mastery.py`](../../pgx-mcts-bench/src/pgx_mcts_bench/single_knot_mastery.py).
New runs write `single-knot-mastery-v2`. Legacy v1 states remain resumable under
their original strict-target, fixed-dose, uniform-replay behavior; resuming an
old artifact never opts it into this protocol. The v2 engine provides:

- a versioned mutable binary heap that supports score increases and decreases;
- leader-biased plus eventually-fair score refresh;
- deterministic equivalent-representation and one-CC frontier generation;
- `K` batched MCTS lanes with online factorized-head training;
- adaptive 50:50 outcome acquisition without changing the scientific target;
- three-seed negative admission, 25--50% rehearsal, and a 0.80 retention gate;
- paired simulation-dose probes and monotone compute promotion;
- verified witness composition and target ratcheting;
- delayed, L10-ordered, fraction-limited distillation;
- atomic JSON/scientist checkpoints and `--resume` support.

Example pilot:

```bash
uv run python -m pgx_mcts_bench.single_knot_mastery \
  --catalogue ../rf-knots/benchmarks/knotinfo-unknotting-gap-candidates-20260814.json \
  --knot 12n_140 \
  --scientist strand-graph \
  --checkpoint /path/to/pinned/strand-graph.pt \
  --output /path/to/artifacts/single-knot-mastery/12n_140 \
  --parallel-searches 8 \
  --simulations 256 \
  --steps 1000
```

The output directory contains resumable state, the exact inherited and updated
scientist state, an event log embedded in `state.json`, and immutable witness
files named by crossing-change count and L1000.

## Required reporting

Every run summary should include:

- target knot, KnotInfo snapshot hash, interval, lower-bound provenance, and
  exact starting-representation IDs;
- inherited checkpoint name, byte hash, architecture capacity, and calibration;
- current target, heap size, nodes by crossing distance, native attempts,
  native/distilled train steps, and wall/CPU time;
- solve probability calibration diagnostics and capped L1000 on a frozen
  rehearsal panel, so self-training cannot silently destroy the inherited skill;
- for each success: root representation, full semantic witness, crossing changes,
  semantic moves, L10, L1000, replay result, and whether any distilled training
  preceded discovery;
- a separate statement of what changed mathematically: new upper bound only, or
  exact value when and only when it meets a certified lower bound.
