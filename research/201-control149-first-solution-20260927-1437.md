# First trained149 control solution verified

2026-09-27 14:35–14:37 UTC (15:35–15:37 London). Read200 and current manifests
before inspecting actual commands, processes, CPU counters, logs and hashes.
The interval since the preceding check is approximately149 minutes, not one hour.

155-v2 trained149 control advanced16→24/128 conditions at12:21:00 UTC:
96→142 episodes,68311 policy evaluations,68432 full-path moves,0→1 solved
condition. Three committed shards; conditions-00016.json SHA256:
af799b796705bd5bce197b94ad5c19e4f98809c26e4b2dbc9fd7bc2f9fe063e6.
Block3967.387965 seconds (66.1 minutes). Controller574 remains10/18 jobs;
worker36622 live97.2% CPU, accumulated274m28.60s versus244m07.27s at12:06.

Session95609 fully replayed the NEW successful UnknotWitness, verified exact
starting word/strands, resource caps, and recomputed costs:

| Condition / root / search | CC | Moves | L10 | L1000 |
| --- | ---: | ---: | ---: | ---: |
|22 /11 /L10|1|126|136|1126|
|Existing reference root11|1|124|134|1124|

Certified lower_cc=1, so this achieves known minimum CC, but has two extra moves
relative to the reference. NO improvement of upper bound or reference cost.
Initial149 failed the matching condition under its budget. This is the CONTROL
arm, not long-data149; the long-data arm remains queued. Synthetic frozen TEST
root, not a public named-knot bound improvement. Full episode/tree/accounting
audit remains scheduled after128 conditions; this success replay is not that
full audit. Do not tune on TEST or infer general superiority from one success.

139 trained135 remains80/128,473 episodes,272447 policy evaluations,272852
full-path moves,9 successful conditions for7 roots. Last saved manifest/log
11:35:29 UTC;10 shards. Controller20100 remains14/24 jobs; worker47974 live
97.8% CPU, accumulated1238m35.33s versus1208m13.03s at12:06. No new success.

Both workers accumulated only about30.4 CPU minutes during the149-minute wall
interval despite being busy at the current observation. Cause of the unobserved
interval is not established; do not claim continuous full-speed execution or
derive isolated throughput from it. Current liveness is verified and no worker
was restarted. CPU advancement does not reveal exact within-block progress.

Read-only session95609 exited0 at14:36:28 UTC: all1459/1273 controller input
pins,10/14 completed-report hashes, four search roots/checkpoint hashes,
3/16/10/16 shards, and both149 final2048-step checkpoints match. Logs agree
with manifests; no traceback/runtime/error/nonfinite/NaN matches. Both live
commands use canonical pgx-mcts-bench. No confirmed defect required a fix.

Continue unchanged searches and remaining replay/comparison matrix. No new
optimizer updates, distance reports or established superiority over fully
non-neural heuristics. No duplicate, interruption, frozen-source mutation,
cloud purchase, database publication or Git action. Dirty work and artifacts
preserved.
