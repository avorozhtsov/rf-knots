# Hourly check: trained raw first replayed solution; both prefixes16

2026-09-29 09:11–09:13 UTC. Read244 and current controller/search manifests
before real process commands, CPU advancement, logs, hashes and witness replay.

139 raw-final-distance advanced8→16/128 conditions (8 roots),48→95 episodes,
22808→47237 policy neural evaluations,22848→47319 full-path moves,
0→1 solved condition on1 distinct root;2 shards. Latest log08:40:36 UTC;
conditions-00008.json SHA256
f4900640ef8bbb0313601c84c4e956c35a4e7440df698cdd9ec37df5aaeec46e;
block6089.122864 seconds. Independently replayed full successful witness,
semantic identity at every prefix, exact start, action-space dimensions,
CC/move caps and costs. Condition9/root4/L1000:CC1,moves161,L10=171,
L1000=1161. Reference89/1079 implies79 moves atCC1;82 extra moves,
not a CC/reference-cost improvement. L10=0/8,L1000=1/8,either1/8 (12.5%).
Matched units-final first8 roots: roots3/4 solved, L10=2/8,L1000=2/8,
either2/8; same condition9 units costs91/1081 imply81 moves atCC1,
80 fewer moves than raw. Distinct139 panel: do not pool with149 results.
Completed raw-initial remains128/128,0 solutions,full audit passed.

149 control-final-uniform advanced8→16/128 conditions (8 roots),48→96
episodes,28568→48160 policy neural evaluations,28608→48240 full-path moves,
0 successes;2 shards. Latest log08:28:29 UTC; shard SHA256
5b7c37e41e6f1dec5fba5d70b37204c28c1ecc56b8eea05c1a5f1aa2ff81624b;
block3260.625741 seconds. Uniform TREE embedding evaluations0 but NEURAL
policy evaluations48160: this is not a fully non-neural heuristic control.

Worker23761 running99.4% CPU,225m22.92s accumulated versus166m04.77s
in244:59m18.15s additional CPU. Worker31474 running99.4% CPU,
191m58.55s versus132m40.73s:59m17.82s additional CPU. Controllers574/20100
remain14/18 and18/24 jobs. Actual commands use canonical pgx-mcts-bench.
No exact unsaved progress inferred from CPU; no timeout-based restart.

Integrity verification09:12:22 UTC passed1459/1273 input pins,14/18 completed
report hashes, eight roots/checkpoint hashes,16/2/16/16/2/16/16/16 saved shards
and both149 final2048-step checkpoint hashes. Logs contain no traceback/
runtime/error/nonfinite/NaN matches. No confirmed defect needs fixing or new
optimizer updates. Current MCTS is evaluation, not training.

Completed149 distance full audit unchanged: long8/64 (12.5%),L10=6/64,
L1000=3/64; control6/64 (9.375%),L10=5/64,L1000=1/64. Heuristic launch
remains deferred per user's greater15% orientation on an independent panel;
the raw8-root prefix is diagnostic, not a promotion result. Overall goal and
heuristic superiority not established. Continue healthy frozen neural searches,
inspect new blocks hourly and replay solutions. Finite-budget unsolved cases
are censored, not proven negatives. Preserve dirty work, sources, artifacts
and checkpoints. No duplicate, restart, heuristic launch, cloud purchase,
publication, commit, merge or push.
