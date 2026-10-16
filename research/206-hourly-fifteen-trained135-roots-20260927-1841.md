# Trained135 reaches120/128 and15 solved roots; control149 second solution

2026-09-27 18:40–18:41 UTC (19:40–19:41 London). Read205 and current manifests
before checking real commands, processes, CPU counters, logs and hashes.

139 trained135 advanced96→120/128 conditions at18:39:25 UTC:
568→705 episodes,368079 policy evaluations,368688 full-path moves,10→18 solved
conditions covering15 distinct roots (previously8). Fifteen committed shards;
latest conditions-00112.json SHA256:
cf5198ea946ed9e7f8a3ccad02ffb88d021a1249891e643f16f7a3adbfc0060c.
Latest recorded block timer1520.313698 seconds. Controller20100 remains14/24
jobs; worker47974 live100% CPU, accumulated1476m13.22s versus1416m06.50s.
One block remains before full trained-search episode/tree/accounting audit.

Session44055 replayed all EIGHT new successful135 witness certificates, checked
exact starts, CC/move caps and recomputed costs. All have CC1 and certified
lower_cc1; all matching initial135 conditions failed under their budgets.

| Condition / root / search | Moves | L10 | L1000 | Reference moves |
| --- | ---: | ---: | ---: | ---: |
|97 /48 /L1000|68|78|1068|64|
|101 /50 /L1000|102|112|1102|100|
|103 /51 /L1000|64|74|1064|62|
|105 /52 /L1000|43|53|1043|43|
|110 /55 /L10|84|94|1084|84|
|114 /57 /L10|99|109|1099|97|
|118 /59 /L10|98|108|1098|98|
|119 /59 /L1000|100|110|1100|98|

Three new solutions match reference costs; the others use2–4 extra moves.
No reference-cost or CC upper-bound improvement. These are synthetic frozen
TEST cases, not public named-knot bound improvements. Success-witness replay
does not replace the forthcoming full episode/tree/accounting audit.

155-v2 trained149 control advanced40→48/128 at18:22:09 UTC:
238→286 episodes,132854 policy evaluations,133096 full-path moves,1→2 solved
conditions/roots. Six shards; conditions-00040.json SHA256:
56370144114265a4711269a7f474512d0f8d0c2520aa1a26dab1ee2997021f00.
Block4106.116354 seconds. Controller574 remains10/18 jobs; worker36622 live100%
CPU, accumulated512m07.03s versus452m00.03s. Long-data149 remains queued.

Session44055 replayed its NEW condition44/root22/L10 solution:CC2,moves112,
L10=132,L1000=2112. Reference CC2,moves110,L10=130,L1000=2110; certified lower_cc2.
Thus known minimum CC is attained, with two extra moves and no upper-bound
improvement. Initial149 failed the matching condition. Note: panel root22 in
149 is not root22 in135; panels differ and must not be conflated.

Read-only session44055 exited0 at18:40:49 UTC. All1459/1273 controller input
pins,10/14 completed-report hashes, four search roots/checkpoint hashes,
6/16/15/16 shards, and both149 final2048-step checkpoints match. No traceback/
runtime/error/nonfinite/NaN matches in logs. Both commands use canonical
pgx-mcts-bench. No confirmed defect required a fix.

Continue unchanged searches and remaining replay/comparison matrix. This
incomplete improvement relative to initial weights is not superiority over
fully non-neural heuristics or an isolated embedding effect. No new optimizer
updates or TEST-guided tuning. No restart, duplicate, frozen-source mutation,
cloud purchase, database publication or Git action. Dirty work and artifacts
preserved.
