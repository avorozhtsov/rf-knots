# Hourly check: live searches, saved prefixes unchanged

2026-09-28 00:46–00:48 UTC (01:46–01:48 London). Read211 and current
controller/search manifests before checking real processes, commands and hashes.

155-v2 control-final-distance remains56/128 conditions,334 episodes,
171118 policy neural evaluations,171400 full-path moves,2 solutions;7 shards.
Manifest/log last saved22:30:08 UTC. Controller574 remains10/18 jobs.
Worker36622 actually running98.1% CPU, accumulated875m28.52s versus
814m40.46s last check:60m48.06s additional CPU. The long-data149 search arm
remains queued, so the paired evaluation is not complete.

139 raw-initial-distance remains16/128 conditions,96 episodes,
48064 policy neural evaluations,48144 full-path moves,0 solutions;2 shards.
Manifest/log last saved23:38:44 UTC. Controller20100 remains16/24 jobs.
Worker96642 actually running98.8% CPU, accumulated327m46.59s versus
266m59.64s last check:60m46.95s additional CPU. This is a neural variant,
not a fully non-neural control. No saved block advancement this hour; CPU
advancement confirms activity, not exact unsaved condition progress.

Integrity check completed00:47:31 UTC: all1459/1273 controller input pins,
10/16 completed-report hashes, five roots/checkpoint hashes,7/16/2/16/16
committed shards and both149 final2048-step checkpoints match. No
traceback/runtime/error/nonfinite/NaN matches in logs. Authorized read-only
process inspection confirmed both workers and controllers and actual use of
canonical pgx-mcts-bench. No new success needs replay; no confirmed defect
required a fix. Current work is search evaluation, not optimizer training.

Completed units-final135 remains128/128 evaluated,20 successful conditions
on16 roots, with previously passed full replay and no CC/reference-cost
improvement. Completed initial units135 and initial control149 remain0 successes.

Keep healthy searches unchanged, then complete remaining queued arms and the
preregistered matched comparison with fully non-neural algorithms. No restart,
duplicate worker, frozen-source mutation, cloud purchase, database publication,
commit, merge or push. Existing dirty work/checkpoints preserved. Superiority
over heuristic search remains unestablished.
