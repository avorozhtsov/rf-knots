# Hourly check: trained149 control first block; trained135 at72/128

2026-09-27 11:05–11:06 UTC (12:05–12:06 London). Read198 and current manifests
before checking real commands, processes, CPU counters, logs and hashes.

155-v2 trained149 control committed its first block at10:13:58 UTC:
8/128 conditions,48 episodes,28568 policy evaluations,28608 full-path moves,
0 solutions. First-block duration8065.299839 seconds (134.4 minutes).
conditions-00000.json SHA256:
d667ca0bbf4e0a9db006d47679dd8c354948162f4b58ec10c374d46009c2a31b.
Controller574 remains10/18 jobs; worker36622 live97.8% CPU, accumulated
183m53.56s versus123m18.66s last hour. Initial149 search and full replay remain
complete128/128,768 episodes,0 solutions under declared budgets. Long-data
trained arm remains queued; this control prefix is not that arm's result.

139 trained135 search advanced64→72/128 conditions at10:25:38 UTC:
377→425 episodes,249735 policy evaluations,250100 full-path moves,9 solved
conditions covering7 roots. No new success to replay. Nine committed shards;
conditions-00064.json SHA256:
73a553b70c15a939c229a02dad0926b085fb747f3bf9c6a964ff2886ade25bed.
Block8382.008433 seconds (139.7 minutes). Controller20100 remains14/24 jobs;
worker47974 live98.0% CPU, accumulated1147m59.40s versus1087m24.66s. Full
trained-search episode/accounting audit remains pending after128 conditions.

Read-only session63743 exited0 at11:05:48 UTC. All1459/1273 controller input
pins,10/14 completed-report hashes, four search roots/checkpoint hashes,
1/16/9/16 shards, and both149 final2048-step checkpoints match. Log timestamps
agree with manifests; no traceback/runtime/error/nonfinite/NaN matches. Both
commands use canonical pgx-mcts-bench. No confirmed defect required a fix.

Both searches show actual saved-file advancement after previous silent periods.
Continue unchanged searches and remaining replay/comparison matrix. No new
optimizer updates, distance reports, improved public CC bounds or established
superiority over fully non-neural heuristics. No restart, duplicate, frozen-source
mutation, cloud purchase, database publication or Git action. Dirty work and
artifacts preserved.
