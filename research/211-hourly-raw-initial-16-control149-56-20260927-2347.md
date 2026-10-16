# Hourly check: raw initial135 16/128; trained control149 active

2026-09-27 23:45–23:47 UTC (2026-09-28 00:45–00:47 London).
Read210 and current manifests before checking actual processes and commands.

139 raw-initial-distance advanced8→16/128 conditions,96 episodes,
48064 policy neural evaluations,48144 full-path moves,0 solutions.
Manifest/log mtime23:38:44 UTC. Second shard `conditions-00008.json` SHA256
f74fcc9005da47c34bb8a2ed272dff2c0998fce4a58624efd92c538e7b03bf92.
Controller20100 remains16/24 completed jobs. Worker96642 is actually running
at100% CPU, accumulated266m59.64s versus207m23.27s at the previous check:
about59m36s new CPU time. This raw variant is neural, not heuristic-only.

155-v2 control-final-distance remains56/128 conditions,334 episodes,
171118 policy neural evaluations,171400 full-path moves,2 solutions.
Seven shards; last saved manifest/log22:30:08 UTC. Worker36622 is actually
running at100% CPU, accumulated814m40.46s versus755m03.29s last check:
about59m37s new CPU time. Controller574 remains10/18 completed jobs.
The long-data149 search arm remains queued. CPU advancement establishes
activity but cannot quantify unsaved progress inside the current block.

Integrity check completed23:46:36 UTC: all1459/1273 controller input pins,
10/16 completed-report hashes, five search roots/checkpoint hashes,
7/16/2/16/16 committed shards, and both149 final2048-step checkpoints match.
Search log mtimes agree with manifests; no traceback/runtime/error/nonfinite/
NaN matches. Authorized read-only process inspection confirmed both workers
and both controllers, with canonical pgx-mcts-bench in their actual commands.
No new successful trajectory needs replay. No confirmed defect required a fix.

Previously completed units-final135 remains20 successful conditions on16
roots,128/128 evaluated, full replay passed, with no new CC or reference-cost
improvement. Initial units135 and initial control149 completed with0 successes.
No optimizer updates this hour: current computation is search evaluation.

Continue healthy searches unchanged, then consume remaining queued arms and
complete the preregistered comparison including fully non-neural algorithms.
No duplicate, restart, frozen-source edit, cloud purchase, database publication,
commit, merge or push. Dirty work and checkpoints preserved. Superiority over
heuristics remains unestablished.
