# Hourly check: seven trained135 roots solved;149 initial unchanged

2026-09-27 04:56–04:57 UTC (05:56–05:57 London). Read192 and current manifests
before checking actual processes, commands, CPU counters, logs and hashes.

139 trained135 search advanced48→56/128 conditions at04:29:56 UTC:
282→329 episodes,182039 policy evaluations,182324 full-path moves,7→9 solved
conditions. Nine successes cover SEVEN roots:3/4 both weights,16/22 L10,
23/24/27 L1000. Latest conditions-00048.json SHA256:
b1971d56afc1647114e59078876f411723640b511f764705a114e1112da00dca.
Block3533.532816 seconds (58.9 minutes). Controller20100 remains14/24 jobs;
worker47974 live93.4% CPU, accumulated782m40.54s versus722m10.13s last hour.

Read-only session18990 replayed both NEW successful UnknotWitness certificates,
verified exact starting word/strands, move/CC caps and recomputed costs:

| Condition / root / search | CC | Moves | L10 | L1000 |
| --- | ---: | ---: | ---: | ---: |
|49 /24 /L1000|1|88|98|1088|
|Reference root24|1|88|98|1088|
|55 /27 /L1000|1|65|75|1065|
|Reference root27|1|63|73|1063|

Both roots have certified lower_cc=1, so new solutions attain known minimum
CC. Root24 matches the reference cost, root27 uses two extra moves; neither
improves a CC upper bound or reference cost. Initial135 solved neither matching
condition. These are synthetic frozen TEST cases, not public named-knot bound
improvements. Full episode/tree/accounting replay remains pending after128
conditions; success replay alone is not the full audit. No TEST-guided tuning.

155-v2 shared-initial149 search remains96/128,576 episodes,306096 policy
evaluations,306576 full-path moves,0 solutions. Last saved manifest/log
03:53:11 UTC;12 shards. Controller574 remains8/18 jobs; worker7097 live91.8%
CPU, accumulated1690m21.65s versus1629m50.89s. Trained149 searches remain queued.
Activity does not establish an exact uncommitted condition count.

Session18990 exited0 at04:57:07 UTC. All1459/1273 controller input pins,
8/14 completed-report hashes, three search roots/checkpoint hashes,12/7/16
shards, and both149 final2048-step checkpoints match. Log timestamps agree
with manifests; no traceback/runtime/error/nonfinite/NaN matches. Both commands
use canonical pgx-mcts-bench. No confirmed defect required a fix.

Continue unchanged searches and complete replay/comparison matrix. No new
optimizer updates, distance reports or established superiority over fully
non-neural heuristics. No restart, duplicate, frozen-source modification, cloud
purchase, database publication or Git action. Dirty work and artifacts preserved.
