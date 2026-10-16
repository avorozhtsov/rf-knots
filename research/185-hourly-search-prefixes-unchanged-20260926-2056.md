# Hourly check: live searches, committed prefixes unchanged

2026-09-26, 20:55–20:56 UTC. Read research184 and current manifests before
checking actual processes, commands, CPU counters, logs and checksums.

155-v2 controller574 remains at8/18 completed jobs. Worker7097 is live at98.5%
CPU, accumulated CPU1219m03.08s versus1159m39.98s last hour. Its shared-initial
149 search remains72/128 conditions,432 episodes,224328 policy evaluations,
224688 full-path moves,0 solved conditions. Last committed shard/log update:
19:45:04 UTC. Nine committed shards; no new block this hour.

139 controller20100 remains at14/24 jobs. Trained135 worker47974 is live at98.2%
CPU, accumulated CPU311m21.00s versus251m58.21s last hour. Prefix remains16/128
conditions,91 episodes,43871 policy evaluations,43954 full-path moves,4 solved
conditions (two roots, each at L10/L1000). Last committed shard/log update:
18:37:13 UTC. Two committed shards. Earlier witness checks are in182/183;
no new successes to replay. Full trained search accounting replay is pending.
The completed initial135 search remains128/128,0 solutions, replay passed.

Read-only verification session31297 exited0 at20:55:55 UTC: all1459/1273
controller input pins,8/14 completed-report hashes, all three search roots and
checkpoint hashes, and9/2/16 committed shards match. A separate check matched
both149 final checkpoints at2048 updates each. No traceback/runtime/error/
nonfinite/NaN matches in evaluation logs. Both commands use canonical
pgx-mcts-bench. No confirmed defect required a fix.

CPU advancement establishes active computation, not an exact uncommitted
condition count. Workers publish complete eight-condition blocks; absence of
a new block is not a stop diagnosis. Continue the unchanged evaluation matrix
and hourly checks; no restart or duplicate worker. No optimizer updates this
hour, no new distance metrics, no established non-neural superiority, and no
improvement of public CC upper bounds. Frozen sources, artifacts and dirty work
preserved; no cloud purchase, database publication or Git operation.
