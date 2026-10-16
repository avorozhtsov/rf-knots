# Hourly check: long fifth distinct root; first L1000 success

2026-09-29 03:08–03:10 UTC. Read238 and current controller/search manifests
before actual process commands, CPU advancement, logs, hashes and full replay.

Long-final149 advanced88→96/128 conditions (44→48 roots),524→570 episodes,
281118→300904 policy neural evaluations,281560→301386 full-path moves,
4→6 solved CONDITIONS but4→5 distinct solved ROOTS. Both new conditions
refer to root47, so they must not be counted as two newly solved knots.
Twelve saved shards; latest log02:34:48 UTC. conditions-00088.json SHA256
38b3827e5d94749f94839557ac37b21dd342fa7ff87ab13a17545806d3daea19;
latest saved block2299.868788 seconds.

Independently replayed every successful witness on the identical first96
conditions:6 long and4 control. Checked complete native witness validity,
semantic identity at every prefix, exact starts, action-space dimensions,
CC/move caps and L10/L1000 arithmetic. New root47:
- condition94/L10: CC1,moves182,L10=192,L1000=1182;
- condition95/L1000: CC1,moves90,L10=100,L1000=1090.
Reference costs96/1086 imply86 moves atCC1. Neither long route improves
reference costs; L1000 route is4 moves above reference. Control root47
L10 route hasCC1,moves102,costs112/1102; its L1000 condition was unsolved.
Long's L1000 route is12 moves shorter than control's L10 route, but these
are different objective conditions; same L10 comparison is182 versus102.
Full all-episode/tree audit remains pending long search completion.

Matched48-root prefix: long L10=5/48 (10.42%),L1000=1/48 (2.08%),
either objective=5/48 (10.42%). Control L10=4/48 (8.33%),L1000=0/48,
either=4/48. Long solved roots3,11,22,28,47; control11,22,45,47.
Thus long adds roots3/28 but misses control root45 within current budget.
Coverage advantage now1 distinct root, not2. Completed control149 still
128/128,6 distinct solved roots,L10=5/64,L1000=1/64,full audit passed.
No new demonstrated CC/reference-cost improvement or heuristic superiority.

Worker79573 actually running99.0% CPU,1151m55.55s accumulated versus
1090m20.59s in238:61m34.96s additional CPU. Controller574 remains12/18
jobs, tracking the same worker. Raw-initial139 advanced88→96/128,
528→576 episodes,305128→330096 policy neural evaluations,
305568→330576 full-path moves,still0 successes;12 shards. Last log02:18:38
UTC, conditions-00088.json SHA256
caff9513ed7c705e4c13fcf8f9d57926bec0d59494b8f94309414e72e5e587f7;
latest saved block8492.082646 seconds. Worker96642 running98.7% CPU,
1900m25.48s versus1838m50.24s:61m35.24s additional CPU. Controller20100
remains16/24 jobs. Real commands use canonical pgx-mcts-bench.

Integrity check03:09:49 UTC passed1459/1273 input pins,12/16 completed-report
hashes, six roots/checkpoint hashes,16/16/12/12/16/16 saved shards and both149
final2048-step checkpoint hashes. Logs show no traceback/runtime/error/
nonfinite/NaN matches. No confirmed defect needs fixing or optimizer updates.

Heuristic launch remains deferred per user's substantial-neural-signal
criterion, e.g. greater15% on an independent panel. Current distinct-root
prefix10.42% does not reach that orientation. Do not pool objective successes
as independent roots. Continue healthy searches unchanged, replay newly saved
solutions and full episode/tree auditing at completion. Overall goal unmet;
finite-budget failures are censored, not proven negatives. MCTS is evaluation,
not training. Preserve dirty work, checkpoints and frozen conditions. No
duplicate, restart, heuristic launch, cloud purchase, publication, commit,
merge or push.
