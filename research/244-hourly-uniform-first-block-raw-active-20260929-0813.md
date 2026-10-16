# Hourly check: neural uniform first block saved; raw active

2026-09-29 08:11–08:13 UTC. Read243 and current controller/search manifests
before live process commands, CPU advancement, logs and checksum checks.

149 control-final-uniform advanced0→8/128 conditions (4 roots),48 episodes,
28568 policy neural evaluations,28608 full-path moves,0 successes;1 shard.
Latest manifest/log07:34:06 UTC; conditions-00000.json SHA256
8ea0046cb9a0b5c660a445aad4c6694dc10eaf062e7a0e4c18634ef207bd54d7;
saved block7655.573441 seconds. Worker23761 actually running100% CPU,
166m04.77s accumulated versus107m01.43s in243:59m03.34s additional CPU.
Controller574 remains14/18 jobs. Uniform tree priority eliminates tree
embedding evaluations (0 here) but still has28568 NEURAL policy evaluations;
this is not a fully non-neural heuristic control.

139 raw-final-distance remains8/128 conditions,48 episodes,22808 policy
neural evaluations,22848 full-path moves,0 successes,1 shard. Last
manifest/log06:59:05 UTC; shard SHA256
3808d90871fd34ee1b74f4e854d86eb3be3c73de88665593ffc4afa9b8e33ac4.
Worker31474 actually running100% CPU,132m40.73s versus73m36.86s:
59m03.87s additional CPU. Controller20100 remains18/24 jobs. Both actual
search/controller commands use canonical pgx-mcts-bench. CPU advancement
establishes activity, not exact unsaved progress; no timeout-based restart.

Completed long distance result/full audit unchanged128/128,8/64 distinct
roots (12.5%),L10=6/64,L1000=3/64 versus control distance6/64 (9.375%),
L10=5/64,L1000=1/64. Completed raw-initial128/128,0 successes/full audit
passed unchanged. No new solution needs replay, no new CC/reference-cost
improvement. Heuristic launch remains deferred per user's greater15%
orientation; pilot completion does not establish overall goal or superiority.

Integrity verification08:12:13 UTC passed1459/1273 input pins,14/18 completed
report hashes, eight roots/checkpoint hashes,16/1/16/16/1/16/16/16 saved shards
and both149 final2048-step checkpoint hashes. Logs contain no traceback/
runtime/error/nonfinite/NaN matches. No confirmed defect requires fixing.
No optimizer updates; current MCTS is evaluation, not training.

Next: continue healthy frozen neural searches unchanged, inspect new blocks
hourly and replay saved solutions. Finite-budget unsolved cases remain
censored, not proven negatives. Preserve dirty work, artifacts, frozen sources
and checkpoints. No duplicate, restart, heuristic launch, cloud purchase,
publication, commit, merge or push.
