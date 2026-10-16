# Hourly check: five trained135 roots solved;149 initial96/128

2026-09-27 03:55–03:56 UTC (04:55–04:56 London). Read191 and current manifests
before checking actual processes, commands, CPU counters, logs and hashes.

139 trained135 search advanced40→48/128 conditions at03:31:01 UTC:
234→282 episodes,159394 policy evaluations,159639 full-path moves,5→7 solved
conditions. Seven successes cover FIVE roots:3/4 both weights,16/22 L10,
23 L1000. Latest conditions-00040.json SHA256:
41bca18520b3e6d09b94a9fa3f2d0adb31b34ee420f021a7c0a35f0468a6e2ce.
Block5289.120578 seconds (88.2 minutes). Controller20100 remains14/24 jobs;
worker47974 live100.0% CPU, accumulated722m10.13s versus662m48.40s last hour.

Read-only session7668 replayed both NEW successful UnknotWitness certificates,
verified exact starting word/strands, move/CC caps, and recomputed costs:

| Condition / root / search | CC | Moves | L10 | L1000 |
| --- | ---: | ---: | ---: | ---: |
|44 /22 /L10|2|178|198|2178|
|Reference root22|2|176|196|2176|
|47 /23 /L1000|1|91|101|1091|
|Reference root23|1|89|99|1089|

Both new solutions have the reference CC and two extra moves, so neither
improves the known upper bound or reference cost. Root22 has certified
lower_cc=1 and upper_cc=2: DO NOT label CC2 optimal. Root23 lower_cc=1 certifies
minimum CC for its new solution. Initial135 solved neither matching condition.
These are synthetic frozen TEST cases, not public named-knot bound improvements.
Full episode/tree/accounting replay remains pending after128 conditions;
success-certificate replay is not that full audit. No TEST-guided tuning.

155-v2 shared-initial149 search advanced88→96/128 at03:53:11 UTC:
528→576 episodes,306096 policy evaluations,306576 full-path moves,0 solutions.
Latest conditions-00088.json SHA256:
9a65a31c57b39b932f124f99af1e0fc0a753183de2feba330339092c5acddc75.
Block5046.363755 seconds (84.1 minutes);12 shards. Controller574 remains8/18
jobs; worker7097 live99.4% CPU, accumulated1629m50.89s versus1570m29.61s.
Trained149 searches remain queued. This is75% of the initial-model search,
not75% of the full evaluation matrix.

Session7668 exited0 at03:56:27 UTC. All1459/1273 controller input pins,
8/14 completed-report hashes, three search roots/checkpoint hashes,12/6/16
shards, and both149 final2048-step checkpoints match. Logs and manifests agree;
no traceback/runtime/error/nonfinite/NaN matches. Both commands use canonical
pgx-mcts-bench. No confirmed defect required a fix.

Continue unchanged searches and complete replay/comparison matrix. No new
optimizer updates, distance reports, or established superiority over fully
non-neural heuristics. No restart, duplicate, frozen-source modification,
cloud purchase, database publication or Git action. Dirty work and artifacts
preserved.
