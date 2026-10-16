# Hourly check: both trained searches active, no new saved block

2026-09-27 10:03–10:04 UTC (11:03–11:04 London). Read197 and current manifests
before inspecting actual commands, processes, CPU counters, logs and hashes.

155-v2 trained149 control worker36622 is live98.4% CPU, accumulated123m18.66s
versus63m03.38s last hour. No committed block yet: next_condition0, empty
counts,0 shards. Manifest07:59:30 UTC; startup log241 bytes,07:58:39 UTC.
This is not a measured zero-success rate. Controller574 remains10/18 jobs;
initial149 search/replay is complete128/128,768 episodes,0 solutions under
declared budgets, independently reconsumed in196. Long-data arm remains queued.

139 trained135 worker47974 is live98.4% CPU, accumulated1087m24.66s versus
1027m09.07s last hour. Controller20100 remains14/24 jobs. Saved prefix remains
64/128 conditions,377 episodes,220495 policy evaluations,220820 full-path moves,
9 successful conditions covering7 roots. Last manifest/log08:05:54 UTC;8 shards.
No new successes to replay; full trained-search episode/accounting audit pending.

Read-only session17995 exited0 at10:03:48 UTC. All1459/1273 controller input
pins,10/14 completed-report hashes, four search roots/checkpoint hashes,
0/16/8/16 committed shards, and both149 final2048-step checkpoints match.
No traceback/runtime/error/nonfinite/NaN matches in logs. Both live commands use
canonical pgx-mcts-bench. CPU increased by about60.3 minutes for each worker;
this establishes computation, not precise uncommitted condition progress.

No confirmed defect required a fix. Continue unchanged eight-condition batch
searches and remaining replay/comparison matrix; do not restart solely for
silent logs. No new optimizer updates, distance reports, public CC improvements
or established superiority over fully non-neural heuristics. No duplicate,
interruption, frozen-source mutation, cloud purchase, publication or Git action.
Dirty work and artifacts preserved.
