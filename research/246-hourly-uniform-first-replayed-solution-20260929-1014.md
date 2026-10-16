# Hourly check: neural uniform first replayed solution

2026-09-29 10:12–10:14 UTC. Read245 and current controller/search manifests
before real commands, CPU advancement, log/hash checks and full witness replay.

149 control-final-uniform advanced16→24/128 conditions (12 roots),96→143
episodes,48160→69225 policy neural evaluations,48240→69346 full-path moves,
0→1 successful condition on1 root;3 shards. Latest log09:38:14 UTC;
conditions-00016.json SHA256
83938d99bce6c909bc5c15130ccf4a6759adc057ccff2a8f3ebe4e3876a13597;
saved block4181.294271 seconds. Independently replayed full successful witness,
semantic identity at each prefix, exact start, action-space dimensions, caps
and L10/L1000 arithmetic. Condition22/root11/L10:CC1,moves152,L10=162,
L1000=1152. Reference134/1124 implies124 moves:28 extra moves at equalCC.
Same control distance condition has126 moves, so uniform uses26 extra moves.
On identical first12 roots, control distance and uniform each solve root11;
long distance solves roots3/11. Uniform L10=1/12,L1000=0/12,either1/12
(8.33%). Neural policy evaluations69225 despite0 tree embedding evaluations;
uniform tree priority is not a fully non-neural heuristic comparator.

139 raw-final-distance unchanged16/128 conditions (8 roots),95 episodes,
47237 policy neural evaluations,47319 full-path moves,1 successful condition,
2 shards;last log08:40:36 UTC. Its successful condition9/root4/L1000 witness
was independently replayed again:CC1,moves161,L10=171,L1000=1161,
reference89/1079. L10=0/8,L1000=1/8,either1/8. No new CC/reference-cost
improvement for either search; full episode/tree auditing pending completion.

Worker23761 running100% CPU,286m04.60s accumulated versus225m22.92s
in245:60m41.68s additional CPU. Worker31474 running99.9% CPU,
252m40.62s versus191m58.55s:60m42.07s additional CPU. Controllers574/20100
remain14/18 and18/24 jobs. Actual commands use canonical pgx-mcts-bench.
CPU activity confirms running workers, not exact unsaved block progress;
no duplicate or timeout-based restart.

Integrity verification10:13:48 UTC passed1459/1273 input pins,14/18 completed
report hashes, eight roots/checkpoint hashes,16/3/16/16/2/16/16/16 saved shards
and both149 final2048-step checkpoint hashes. Logs contain no traceback/
runtime/error/nonfinite/NaN matches. No confirmed defect needs fixing or new
optimizer updates; current MCTS is evaluation, not training.

Completed149 distance/full audit unchanged long8/64 (12.5%),L10=6/64,
L1000=3/64 versus control6/64 (9.375%),L10=5/64,L1000=1/64. Heuristic
launch remains deferred per user's greater15% orientation on independent
evaluation; current prefixes are diagnostic, not promotion or superiority.
Next: continue healthy frozen searches, replay new saved successes and audit
at completion. Overall goal unmet; finite-budget failures remain censored,
not proven negatives. Preserve dirty work, sources, artifacts/checkpoints.
No restart, heuristic launch, cloud purchase, publication, commit, merge or push.
