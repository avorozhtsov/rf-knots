# Q-run-like curriculum rung-1 on-policy TRAIN teacher

2026-10-05. The independent DEV result in protocol288 gives a modest
positive short signal and a seed-dependent long signal. To continue the
Q-run-like loop, collect new verified solutions using the frozen
curriculum1024 endpoint SHA
`03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8`.
This is search data collection, not optimizer training.

Use every one of the 1,000 distinct TRAIN roster roots, both L10 and L1000,
one fixed search seed 20271300 and 64 simulations: 2,000 conditions. Use
canonical pgx-mcts-bench and the same tree, distance prior, continuation,
cache and resource caps as the matched DEV searches. Process roots in the
frozen roster stage order from simpler to harder. Store every search shard,
CPU/wall/neural counts, full episodes, best witness, and independently
computed TRAIN-ownership flags for episodes. Never turn a capped no-solve
into a negative action. Preserve the known constructive reference witness
for every root regardless of search outcome.

After collection, independently verify all SHA, replay every found path,
caps and TRAIN ownership. Measure by stage and objective: solved searches,
admitted solutions, unique paths, cost improvement over the known witness,
and CPU per admitted improved path. The next optimizer contract may distil
the cheapest admitted verified path per root/objective mixed with the old
reference witnesses, but must be frozen only after this audit. No DEV or
TEST data enter this teacher. This collection is not the separate F-dominant
R2/R4 experiment and does not start its 24-hour timer.
