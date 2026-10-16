# Hourly check: trained control149 72/128; raw initial135 active

2026-09-28 02:48–02:49 UTC (03:48–03:49 London). Read213 and current
manifests before checking actual processes, commands, logs and hashes.

155-v2 control-final-distance advanced64→72/128 conditions,430 episodes,
221102 policy neural evaluations,221464 full-path moves,2 successful
conditions unchanged. Nine shards; manifest/log mtime01:50:00 UTC.
Latest `conditions-00064.json` SHA256:
18f4acfa163654b667e64348b99b58a67e5a12c40d9bb5a8cf352fa095b79278.
Worker36622 actually running99.1% CPU, accumulated996m52.64s versus
935m53.33s last check:60m59.31s additional CPU. Controller574 remains10/18
jobs; long-data149 search remains queued, so the paired evaluation is incomplete.

139 raw-initial-distance remains16/128 conditions,96 episodes,
48064 policy neural evaluations,48144 full-path moves,0 successful conditions.
Two shards; manifest/log mtime23:38:44 UTC. Worker96642 actually running
99.1% CPU, accumulated449m10.05s versus388m10.94s last check:60m59.11s
additional CPU. Controller20100 remains16/24 jobs. CPU advancement confirms
activity but does not quantify the unsaved current block. No timeout-based
restart is warranted. This variant is neural, not a fully non-neural control.

Integrity verification at02:49:04 UTC passed all1459/1273 controller input
pins,10/16 completed-report hashes, five roots/checkpoint hashes,9/16/2/16/16
committed shards and both149 final2048-step checkpoint hashes. Log mtimes
agree with manifests and no traceback/runtime/error/nonfinite/NaN matches
were found. Authorized read-only process inspection confirmed both workers
and controllers and canonical pgx-mcts-bench in their actual commands.
No new success requires replay and no confirmed defect requires a fix.

Previously completed units-final135 remains128/128,20 successful conditions
on16 roots, with full replay passed and no CC/reference-cost improvement.
Initial units135 and initial control149 remain completed with0 successes.
No optimizer training occurred this hour; these computations are search evaluation.

Continue healthy searches unchanged, then complete remaining queued arms and
the preregistered matched fully non-neural comparison. No duplicate, restart,
frozen-source change, cloud purchase, database publication, commit, merge or
push. Dirty work and checkpoints preserved. Superiority over heuristics is not
established; limited-budget failures are not proven negative examples.
