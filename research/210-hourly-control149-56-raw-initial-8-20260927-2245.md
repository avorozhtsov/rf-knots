# Hourly check: trained control149 56/128; raw initial135 active

2026-09-27 22:44–22:46 UTC (23:44–23:46 London). Read209 and current
manifests before checking real processes, commands, CPU counters, logs and
hashes.

155-v2 trained149 control-final-distance advanced48→56/128 conditions:
334 episodes,171118 policy evaluations,171400 full-path moves,2 solutions
unchanged. Seventh committed shard `conditions-00048.json` SHA256
790d430ff6e2652c59e866ddfc9d0fbabcbc7dfe5260e0647d9d1d39d88da558;
manifest/log mtime22:30:08 UTC. Controller574 remains10/18 jobs; worker36622
actually running, accumulated755m03.29s CPU versus693m04.42s last hour.
Long-data149 arm remains queued, so the pair is not yet evaluated.

139 raw-initial-distance remains8/128 conditions,48 episodes,22808 policy
evaluations,22848 full-path moves,0 solutions; one committed shard, mtime
21:06:19 UTC. Controller20100 remains16/24 jobs. Worker96642 actually
running, accumulated207m23.27s CPU versus145m23.34s last hour. This is a
neural resource-input variant, not a fully non-neural heuristic control.

Completed units-final135 remains128/128,753 episodes,20 successful conditions
on16 roots, with full replay passed and no new CC or reference-cost
improvement; initial units remains0/128 solved. Original initial149 is
128/128 with0 solutions and full replay passed. No optimizer updates this hour.

Read-only integrity check at22:45:38 UTC passed all1459/1273 controller input
pins,10/16 completed-report hashes, five search roots and checkpoint hashes,
7/16/1/16/16 committed shards, and both149 final2048-step checkpoints. No
traceback/runtime/error/nonfinite/NaN match in search logs. Both live workers
and controllers use canonical pgx-mcts-bench. The first sandboxed process
query was denied; authorized read-only process inspection succeeded. No
confirmed defect needed a fix.

Continue the two searches unchanged, then evaluate the queued long-data149
arm and complete matched fully non-neural comparison/replay. No duplicate,
restart, frozen-source mutation, cloud purchase, database publication or Git
action. Dirty work and artifacts preserved. Neural superiority over heuristic
search is not established.
