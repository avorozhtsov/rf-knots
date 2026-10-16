# Both searches advance after host hibernation

2026-09-27 15:36–15:37 UTC (16:36–16:37 London). Read202 and current manifests
before checking actual commands, processes, CPU counters, logs and hashes.

155-v2 trained149 control advanced24→32/128 at15:35:09 UTC:
142→190 episodes,88047 policy evaluations,88208 full-path moves,1 solution
(unchanged, certificate verified in201). Four committed shards;
conditions-00024.json SHA256:
40630c3484fdb2c9b72455db6577127e7dd0b9144080ff473f13a1868871fc26.
Recorded block timer4539.749054 seconds. Controller574 remains10/18 jobs;
worker36622 live76.3% CPU, accumulated333m31.26s versus276m06.66s in202.
Long-data149 arm remains queued; initial149 search/full replay is complete.

139 trained135 advanced80→88/128 at15:29:29 UTC:
473→521 episodes,298663 policy evaluations,299108 full-path moves,9 successful
conditions covering7 roots (unchanged). Eleven committed shards;
conditions-00080.json SHA256:
aa8207f6efaa0f9be811108464404c8cf32bdc3eb500beed9ea21395d8b870a6.
Recorded block timer6931.742248 seconds. Controller20100 remains14/24 jobs;
worker47974 live74.3% CPU, accumulated1297m38.38s versus1240m13.50s in202.
No new success certificates. Full trained-search accounting replay is pending.

Both workers gained about57.4 CPU minutes since202 and committed fresh files.
Power check: AC connected, battery67%, charging. Do not conflate saved timer
durations with timestamp-to-timestamp elapsed wall time: these blocks straddled
the118m55s sleep established in202. Their stored durations are substantially
shorter than UTC gaps; preserve raw counters and sleep evidence separately.
No performance-superiority claim from these interrupted/shared-host timings.

Read-only session32118 exited0 at15:37:03 UTC: all1459/1273 controller input
pins,10/14 completed-report hashes, four search roots/checkpoint hashes,
4/16/11/16 shards and both149 final2048-step checkpoints match. Logs and
manifests agree; no traceback/runtime/error/nonfinite/NaN matches. Both commands
use canonical pgx-mcts-bench. No confirmed defect required a fix.

Continue unchanged searches and remaining replay/comparison matrix. No new
optimizer updates, distance reports, public CC improvements or established
superiority over fully non-neural heuristics. No duplicate, restart, frozen-source
mutation, cloud purchase, database publication or Git action. Dirty work and
artifacts preserved.
