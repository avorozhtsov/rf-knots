# Hourly check: third trained135 root solved;149 initial at88/128

2026-09-27 02:56–02:57 UTC (03:56–03:57 London). Read190 and current manifests
before inspecting actual commands, processes, CPU counters, logs and hashes.

139 trained135 search advanced32→40/128 conditions at02:02:50 UTC:
187→234 episodes,133926 policy evaluations,134130 full-path moves,4→5 solved
conditions. Five conditions cover THREE roots:3 and4 at both L10/L1000,16 at
L10 only. New shard conditions-00032.json SHA256:
1c273c52826b1435ad4477cbd145f12d71f482557f691584e4e454878d794b42.
Block3315.516483 seconds (55.3 minutes). Controller20100 remains14/24 jobs;
worker47974 live100.0% CPU, accumulated662m48.40s versus601m23.04s last hour.

Session50989 fully replayed the NEW successful UnknotWitness for condition32,
verified exact starting word/strands, move and CC caps, and recomputed costs:

| Result for panel root16 | CC | Moves | L10 | L1000 |
| --- | ---: | ---: | ---: | ---: |
| Trained135, L10 search |1|224|234|1224|
| Existing reference |1|101|111|1101|

Certified lower_cc=1, so the new trajectory attains known minimum CC but uses
123 more moves than the reference; NO CC upper-bound or reference-cost
improvement. Initial135 failed the matching condition; trained135 also failed
root16's L1000 condition33 under its bounded search. These are synthetic frozen
TEST roots, not new public named-knot bounds. Witness replay is not the full
episode/tree/accounting audit, which remains scheduled after128 conditions.
Do not tune training on these TEST outcomes or claim an embedding-specific
effect from a jointly trained model.

155-v2 shared-initial149 search advanced80→88/128 at02:29:02 UTC:
480→528 episodes,284680 policy evaluations,285120 full-path moves,0 solutions.
Eleven shards; conditions-00080.json SHA256:
278146a9895cd6b78fe299501d24aa39d416d5f4d865504c8e3fe7309abc1df8.
Block15253.727349 seconds (254.2 minutes), confirming file advancement after
the long silent interval. Controller574 remains8/18 jobs; worker7097 live99.7%
CPU, accumulated1570m29.61s versus1509m05.31s. Trained149 searches remain queued.

Read-only session50989 exited0 at02:57:25 UTC. All1459/1273 controller input
pins,8/14 completed-report hashes, three search roots/checkpoint hashes,
11/5/16 shards, and both149 final2048-step checkpoints match. Log timestamps
agree with manifests; no traceback/runtime/error/nonfinite/NaN matches. Both
live commands use canonical pgx-mcts-bench. No confirmed defect required a fix.

Continue unchanged searches and complete replay/comparison matrix. No new
optimizer updates, distance reports or established superiority over fully
non-neural heuristics. No restart, duplicate, frozen-source modification, cloud
purchase, database publication or Git action. Dirty work and artifacts preserved.
