# Hourly check:149 initial search72/128; trained135 prefix unchanged

2026-09-26,19:55–19:57 UTC. Read183 before inspecting manifests, actual process
commands, CPU counters, logs and hashes. No frozen conditions changed.

New155-v2 worker7097 committed a block at19:45:04 UTC:64→72/128 conditions,
384→432 episodes,205408→224328 policy evaluations,224688 full-path moves,
0 solutions. Block4262.137801s (71.0min). Log and manifest agree.
`conditions-00064.json` SHA256:
`034678a2a455d656de910f905c38b5910142e770b6f067521e92b9a5bbe93b61`.
Controller574/session17201 remains8/18 jobs; worker98.6%CPU,1159m39.98s CPU
versus1099m05.00s last hour. This remains the shared-initial149 model.

Old139 trained units-final-distance prefix is unchanged at16/128 conditions,
91 episodes,43871 policy evaluations and4 solved conditions:two roots each at
L10/L1000. Last block18:37:13 UTC; no new trained-weight outcome this hour.
Controller20100 remains14/24 jobs; worker47974 live98.2%CPU,251m58.21s CPU
versus191m22.45s. All four success certificates were verified in182/183; none
improved existing CC bounds, and one matched its reference's cost. Full search
episode/accounting replay remains pending. Initial135 search/replay is complete.

Read-only session7460 exited0 at19:56:04 UTC:1459/1273 controller input pins,
8/14 completed-report hashes, all three roots/search checkpoint hashes,
9/2/16 committed shards and both149 final checkpoints match. No matching
traceback/runtime/nonfinite/NaN indication in either log directory. No confirmed
issue required a fix. Both live commands use canonical pgx-mcts-bench.

Continue unchanged searches and hourly checks. No new optimizer updates or
distance results. Incomplete positive trained135 prefix is not evidence of
superiority over fully non-neural heuristics or an embedding-specific effect.
No restart, duplicate, source mutation, cloud purchase, database or Git publication.
