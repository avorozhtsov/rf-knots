# Hourly check: new search40/128, old search80/128

2026-09-26,08:47–08:49 UTC. Read172, then inspected current manifests, actual
process commands, CPU counters, logs and artifact hashes. Both searches retained.

New155-v2 worker7097 committed its fifth block at08:38:05 UTC:32→40/128
conditions,192→240 episodes,90224→114808 policy evaluations,115008 full-path
moves,0 solutions. The block took7258.744669s (121.0 minutes). Latest shard
`conditions-00032.json` SHA256:
`3cd5c24a184bd89e534b1336251c4b2efd19898d2bccf3084d92c3b9fdf930c9`.
Log and manifest agree. Controller574/session17201 remains8/18 jobs; worker
97.5%CPU,513m53.73s CPU versus455m01.66s last hour. Still initial-model search,
not either trained endpoint. No new training or distance evaluation occurred.

Old139 remains80/128 conditions,480 episodes,278912 policy evaluations,
0 solutions. No new block since07:16:20 UTC. Controller20100 remains12/24 jobs;
live worker21103,97.9%CPU,842m05.46s CPU versus783m13.13s last hour. Both commands
use canonical pgx-mcts-bench, with no duplicate, restart or interruption.

Read-only verification session61965 exited0 at08:48:15 UTC. All1459/1273
controller input pins,8/12 completed-report hashes, both roots hashes,5/10
committed search shards and both final149 checkpoint hashes match. No matched
traceback/runtime/nonfinite/NaN indication in either log directory. No confirmed
error needed a fix. Full post-exit semantic replay remains pending; extra prefix
replay has only been completed for the first155 block (168).

Continue the unchanged registered matrix and hourly checks. Completed distance
results remain mixed (165); finite-budget failures are not impossibility labels.
Minimum-CC search benefit, superiority over fully non-neural heuristics and
public upper-bound improvements remain unestablished. No source mutation,
cloud purchase, database publication or Git publication occurred.
