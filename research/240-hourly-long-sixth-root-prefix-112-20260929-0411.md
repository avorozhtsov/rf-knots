# Hourly check: sixth distinct long root; both searches112/128

2026-09-29 04:09–04:11 UTC. Read239 and current manifests before actual
process commands, CPU advancement, logs, checksum verification and replay.

Long-final149 advanced96→112/128 conditions (48→56 roots),570→665 episodes,
300904→335498 policy neural evaluations,301386→336060 full-path moves.
Solved conditions6→7, distinct solved roots5→6;14 shards. Latest
log03:51:54 UTC; conditions-00104.json SHA256
64c7b11aea900eb4ba90d3f03988ab32304debb45e7b6359877da1db3d18ff0f;
latest block1209.067720 seconds.

Independently replayed all7 long and5 control successful witnesses on the
first112 conditions, checking complete native replay, each semantic prefix,
exact starts, action-space dimensions, CC/move caps and L10/L1000 arithmetic.
New long condition99/root49/L1000:CC1,moves218,L10=228,L1000=1218.
Reference130/1120 implies120 moves atCC1, so new route is98 moves longer
and not a reference-cost improvement. Control has no solution on root49
within budget. Control adds its already completed root54/L1000 to this
expanded comparison:CC1,moves49,L10=59,L1000=1049,reference55/1045.

Matched56-root prefix: long L10=5/56 (8.93%),L1000=2/56 (3.57%),
either=6/56 (10.71%); control L10=4/56 (7.14%),L1000=1/56 (1.79%),
either=5/56 (8.93%). Long roots3,11,22,28,47,49; control11,22,45,47,54.
Long-only3,28,49; control-only45,54. Coverage advantage1 root, not2,
and objective successes are not independent knots. Completed control149
remains128/128,6 distinct roots,L10=5/64,L1000=1/64,full audit passed.
Full all-episode/tree audit for long still pending completion. No new
CC/reference-cost improvement or demonstrated heuristic superiority.

Worker79573 actually running99.0% CPU,1212m19.88s accumulated versus
1151m55.55s in239:60m24.33s additional CPU. Controller574 remains12/18
jobs. Raw-initial139 advanced96→112/128 conditions,576→672 episodes,
330096→363664 policy neural evaluations,330576→364224 full-path moves,
still0 successes;14 shards. Latest log04:01:45 UTC, shard SHA256
6b9a9137c84f03d866f24f3802a764f9d04127e9d793c09290fb3b53e1264c93;
latest block2942.423932 seconds. Worker96642 actually running99.3% CPU,
1960m50.49s versus1900m25.48s:60m25.01s additional CPU. Controller20100
remains16/24 jobs. Actual commands use canonical pgx-mcts-bench. CPU activity
does not quantify unsaved progress; no timeout-based restart or duplicate.

Integrity check04:10:20 UTC passed1459/1273 input pins,12/16 completed-report
hashes, six roots/checkpoint hashes,16/16/14/14/16/16 saved shards and both149
final2048-step checkpoint hashes. Logs show no traceback/runtime/error/
nonfinite/NaN matches. No confirmed defect needs fixing. No optimizer updates;
MCTS here is evaluation, not training. Raw is neural, not heuristic.

Heuristic launch remains deferred per user's substantial-neural-signal
criterion, e.g. greater15% on an independent panel. Current distinct-root
prefix10.71% is below that orientation and not a full-panel promotion result.
Continue healthy searches unchanged, replay new saved solutions and complete
full episode/tree auditing. Overall goal unmet; finite-budget failures are
censored, not proven negatives. Preserve dirty work, frozen inputs and
checkpoints. No heuristic launch, cloud purchase, publication, commit,
merge or push.
