# TRAIN hard-stage F-supply budget pilot

2026-10-07. This is a new, score-blind TRAIN-only collection, not neural
training. The exact audited 64-simulation teacher yields zero F-usable
stage-3 roots among 287, while the whole bank has only 384 F-usable roots.
The purpose is to measure whether additional independent search and larger
*actual* tree budgets create verified different-cost solutions on hard roots,
and their CPU cost. Do not treat capped failures as negative actions.

Use the frozen roster-v3 SHA
`268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506`
and curriculum1024 checkpoint SHA
`03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8`.
From stage 3 in roster order, choose the first four roots with reference CC=1
and first four with reference CC=2. Their roster indices are
`[4,8,10,21,0,1,2,3]`. For each root run L10 and L1000, seeds
20271351 and 20271352, with 64 and 128 simulations: 64 search conditions.
Use canonical pgx-mcts-bench, fixed tree/continuation settings from
protocol289 (distance prior, insertion candidates=4, widening batch=4,
c_puct=1.5, temperature=1, tile cache, original caps). Search seed is
the fixed seed plus the task position, as in the existing runner; each task
has a fresh tree. Retain full episodes, best witness, ownership flags,
CPU/wall/neural work, and SHA-pinned resumable shards.

Before interpreting supply, independently check every shard SHA, full
replay and caps of found solutions, TRAIN ownership of each crossing-change
state, and exact F groups from admitted complete paths plus the constructive
reference, with research/273's margin 2, max three positives and K=4
negatives. Report per budget, objective and CC stratum: solved searches,
distinct complete paths, different-cost paths, exact F-usable root groups,
and CPU/neural work per usable group. Eight roots are a yield probe, not a
population estimate: if the outcome is sparse or inconsistent, a separately
frozen second pilot is required before any bulk collection. No DEV/TEST
checkpoint choice, and no multi-day R2/R4 timer until the training bank is
audited to the prerequisite size.
