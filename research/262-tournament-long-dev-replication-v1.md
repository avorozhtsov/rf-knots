# Tournament policy: independent long DEV replication

Frozen before observing any result on this panel, 2026-10-03.

The 32-condition short/medium DEV screen in protocol261 completed with
replay-verified solutions for A=29, F=32, G=32. On common solved conditions,
F often lowered objective cost against A; G was the closest of the
fully-solving variants to F on L10. Advance **only A, F, G**. This selection
is exploratory: the short screen has one search seed and word lengths 18–45.
G's additional high-CC tournament signal had only 17 contributing TRAIN
groups; a long DEV result cannot retroactively increase that sample size.

## New data and search contract

Build a fresh panel from deterministic `conjugate_witness` indices 0–511,
seed 20261117, `max_strands=20`, `min_depth=20`, `max_depth=60`. Rank indices
by SHA256 of `20261117:index`, then admit the first 16 distinct
validation-owned invariant query groups with start-word length 60–150 and a
fully replayed constructive unknot witness. Require at least four CC=2
references; if this fixed procedure cannot provide them, stop and report the
deficit, rather than alter the panel after search results. Exclude all groups
recorded by the long-general supply registry, the protocol261 short DEV
panel, and the previously viewed TEST150 panel. The generated witness sets
common resource caps for all actors; it is **not** supplied to the search.

Use L10 and L1000 separately. For each of 16 representations × 2 objectives,
run A/F/G at two predeclared search seeds 20271103 and 20271104, 128 tree
simulations each, `insertion_candidates=4`, `widening_batch=4`, `c_puct=1.5`,
`temperature=1.0`, distance tree prior, cached CPU float32 inference, and
the same horizon/resource caps from the panel. This is 192 conditions.
Preserve source/checkpoint/panel hashes and resumable shards. Independently
verify every returned witness by complete replay and compare solved fraction,
CC, moves, objective cost on common successes, neural evaluations, CPU/wall
time, both objectives, and seed-to-seed stability. A missing solution at this
budget is censored, not a negative action label.

No variant advances solely from this panel if differences are within
uncertainty. This is still rollout-guided protocol139 search, **not** the
AlphaZero-style multi-output value-head gate and **not** a fully non-neural
canonical `pgx-mcts-bench` superiority comparison. TEST150 is excluded from
selection. Do not change any prior source/checkpoint or overwrite artifacts.
