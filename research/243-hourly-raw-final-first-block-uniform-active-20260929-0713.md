# Hourly check: trained raw first block; neural uniform search active

2026-09-29 07:11–07:13 UTC. Read242 and current controller/search manifests
before actual process commands, CPU advancement, logs and checksum checks.

139 raw-final-distance advanced0→8/128 conditions (4 roots),48 episodes,
22808 policy neural evaluations,22848 full-path moves,0 successes;1 saved
shard. Latest manifest/log06:59:05 UTC; conditions-00000.json SHA256
3808d90871fd34ee1b74f4e854d86eb3be3c73de88665593ffc4afa9b8e33ac4;
saved block3481.764725 seconds. Worker31474 verified running99.1% CPU,
73m36.86s accumulated versus12m40.51s in242:60m56.35s additional CPU.
Controller20100 remains18/24 jobs, tracking this existing worker. Completed
raw-initial128/128 with0 solutions/full audit passed remains unchanged.

149 control-final-uniform remains0/128 saved conditions,0 shards; fresh
manifest05:26:28 UTC, last log05:25:37 UTC. Worker23761 verified running
99.3% CPU,107m01.43s versus46m04.48s in242:60m56.95s additional CPU.
Controller574 remains14/18 jobs. Activity is confirmed, not stopped; no
exact unsaved block progress is inferred. First block is not yet committed,
so do not report0 solved roots as a measured outcome of a completed sample.
Uniform tree priority still uses neural policy continuations; not heuristic.

Completed long-final-distance/full audit remain128/128,8 distinct roots/64
(12.5%),L10=6/64,L1000=3/64, versus control distance6/64 (9.375%),
L10=5/64,L1000=1/64. No new success or replay required this hour. No new
CC/reference-cost improvement. Goal and heuristic superiority not achieved;
user-requested heuristic deferral remains in force below greater15% orientation.

Integrity verification07:12:44 UTC passed1459/1273 input pins,14/18 completed
report hashes, eight roots/checkpoint hashes,16/0/16/16/1/16/16/16 saved shards
and both149 final2048-step checkpoint hashes. Actual commands for workers
and controllers use canonical pgx-mcts-bench. Logs contain no traceback/
runtime/error/nonfinite/NaN matches. No confirmed defect requires fixing.
No optimizer updates; MCTS here is evaluation, not training.

Next: continue healthy searches unchanged, inspect saved blocks at hourly
cadence and replay new successes. Preserve dirty work, frozen sources,
artifacts/checkpoints. Finite-budget failures are censored, not proven
negatives. No duplicate, restart on observation timeout, heuristic launch,
cloud purchase, database publication, commit, merge or push.
