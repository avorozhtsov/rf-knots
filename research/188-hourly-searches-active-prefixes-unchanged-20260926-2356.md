# Hourly check: active searches, unchanged committed results

2026-09-26 23:55–23:56 UTC (27 September 00:55–00:56 London). Read187 and
current manifests before inspecting actual process commands, CPU counters,
logs and immutable artifact hashes.

155-v2 controller574 remains8/18 jobs. Shared-initial149 worker7097 is running
at96.7% CPU with1390m46.56s accumulated CPU, versus1333m11.53s at the previous
hourly check. Search remains80/128 conditions,480 episodes,248432 policy
evaluations,248832 full-path moves,0 solutions. Last committed manifest/log
update22:14:45 UTC;10 shards. Trained149 searches remain queued.

139 controller20100 remains14/24 jobs. Trained135 worker47974 is running at
97.6% CPU with483m04.20s accumulated CPU, versus425m29.55s last hour. Search
remains24/128 conditions,139 episodes,83239 policy evaluations,83362 full-path
moves,4 solved conditions (two roots, each at L10/L1000). Last committed
manifest/log update22:53:46 UTC;3 shards. No new successes to replay. Existing
witness checks are in182/183; full trained-search accounting replay is pending.
Completed135 initial search remains128/128,0 solutions, with passed replay.

Read-only session78928 exited0 at23:55:45 UTC. All1459/1273 controller input
pins,8/14 completed-report hashes, three search roots/checkpoint hashes,
10/3/16 committed shards, and both149 final2048-step checkpoints match. Logs
agree with manifests and contain no matching traceback/runtime/error/
nonfinite/NaN indications. Both commands use canonical pgx-mcts-bench.

Each worker gained about57.6 CPU minutes since the preceding hourly check;
this confirms computation but not exact within-block progress. No new saved
block this hour, no confirmed defect to fix, and no restart or duplicate.
Continue unchanged comparisons and hourly checks. No new optimizer updates,
distance reports, public CC improvements or established superiority over
fully non-neural heuristics. Frozen sources, dirty work and artifacts preserved;
no cloud purchase, database publication or Git action.
