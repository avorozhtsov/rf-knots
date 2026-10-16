# Hourly check: trained149 control16/128; trained135 at80/128

2026-09-27 12:06–12:07 UTC (13:06–13:07 London). Read199 and current manifests
before checking real processes, commands, CPU counters, logs and hashes.

155-v2 trained149 control advanced8→16/128 conditions at11:14:50 UTC:
48→96 episodes,48160 policy evaluations,48240 full-path moves,0 solutions.
Two committed shards; conditions-00008.json SHA256:
c23339586f1da7ae6d95bc7391564f20758af3aa0c777eaa5b9ee7dbc9f97caf.
Block3648.641493 seconds (60.8 minutes). Controller574 remains10/18 jobs;
worker36622 live100% CPU, accumulated244m07.27s versus183m53.56s last hour.
Initial149 search/replay remains complete128/128,768 episodes,0 solutions under
declared budgets. Long-data trained arm remains queued; this is control only.

139 trained135 search advanced72→80/128 conditions at11:35:29 UTC:
425→473 episodes,272447 policy evaluations,272852 full-path moves,9 solved
conditions covering7 roots. No new success to replay. Ten committed shards;
conditions-00072.json SHA256:
99baea2621a56a9990c4a33a0ddf8dd46a299585b6fda3f05d9d93dd9bb6f6e6.
Block4189.244536 seconds (69.8 minutes). Controller20100 remains14/24 jobs;
worker47974 live100% CPU, accumulated1208m13.03s versus1147m59.40s. Full
trained-search episode/accounting audit remains pending after128 conditions.

Read-only session11972 exited0 at12:06:48 UTC: all1459/1273 controller input
pins,10/14 completed-report hashes, four search roots/checkpoint hashes,
2/16/10/16 committed shards and both149 final2048-step checkpoints match.
Logs and manifests agree; no traceback/runtime/error/nonfinite/NaN matches.
Both commands use canonical pgx-mcts-bench. No confirmed defect required a fix.

Continue both unchanged searches and remaining replay/comparison matrix.
No new optimizer updates, distance reports, public CC improvements or established
superiority over fully non-neural heuristics. No restart, duplicate, frozen-source
mutation, cloud purchase, database publication or Git action. Dirty work and
artifacts preserved.
