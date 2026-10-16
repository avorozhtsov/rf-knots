# Hourly check: trained control149 80/128; raw initial135 long block active

2026-09-28 03:48–03:50 UTC (04:48–04:50 London). Read214 and current
manifests before actual process, command, log and integrity checks.

155-v2 control-final-distance advanced72→80/128 conditions,478 episodes,
245206 policy neural evaluations,245608 full-path moves,2 successful conditions
unchanged. Ten committed shards; manifest/log mtime03:17:21 UTC.
Latest `conditions-00072.json` SHA256:
5902bf64757ffe2b5dde90e50c4c11440fbf4e0c0a0ac22d0998a15beddbdd47.
Worker36622 actually running98.9% CPU, accumulated1057m14.29s versus
996m52.64s last check:60m21.65s additional CPU. Controller574 remains10/18
jobs; long-data149 search remains queued, so the paired evaluation is incomplete.

139 raw-initial-distance remains16/128 conditions,96 episodes,
48064 policy neural evaluations,48144 full-path moves,0 successful conditions.
Two shards; manifest/log mtime23:38:44 UTC. Worker96642 actually running
99.1% CPU, accumulated509m31.42s versus449m10.05s last check:60m21.37s
additional CPU. Controller20100 remains16/24 jobs. CPU advancement confirms
activity but cannot quantify unsaved progress. Read-only source inspection
confirms saving/logging only after all eight conditions in run_batch return.
Pending registered conditions16–23 cover roots8–11 at both L10/L1000 weights,
word lengths328/383/393/403,9/20/22/24 strands,move caps704/836/860/884;
the batch horizon is884. This describes the pending workload, not proof of
internal progress or an exact explanation of wall time. No restart warranted
solely by observation timeout. This variant is neural, not heuristic-only.

Integrity verification at03:49:31 UTC passed all1459/1273 controller input
pins,10/16 completed-report hashes, five roots/checkpoint hashes,10/16/2/16/16
committed shards and both149 final2048-step checkpoint hashes. Log mtimes
agree with manifests; no traceback/runtime/error/nonfinite/NaN matches.
Authorized read-only process inspection confirmed both workers/controllers
and canonical pgx-mcts-bench in actual commands. No new success needs replay,
no confirmed defect needs a fix, and no optimizer training occurred this hour.

Completed units-final135 remains128/128 evaluated,20 successful conditions
on16 roots, with full replay previously passed and no CC/reference-cost
improvement. Completed initial units135 and control149 remain0 successes.

Continue healthy searches unchanged; then complete queued arms and the
preregistered matched fully non-neural comparison. No duplicate, restart,
frozen-source edit, cloud purchase, database publication, commit, merge or push.
Dirty work/checkpoints preserved. Superiority over heuristics remains
unestablished; limited-budget failures are not proven negative examples.
