# Hourly check: long prefix80; no new solution

2026-09-28 23:05–23:07 UTC (2026-09-29 00:05–00:07 London).
Read234 and current controller/search manifests before real process commands,
CPU advancement, log timestamps and checksum verification.

Long-final149 advanced72→80/128 conditions (36→40 roots),428→476 episodes,
220766→244870 policy neural evaluations,221128→245272 full-path moves.
Four successful conditions on4 roots unchanged;10 saved shards. Latest
log22:54:08 UTC; conditions-00072.json SHA256
ded5acfe67367b03682c447eed79e3bb262bfa4cf8f371acc7f0fd134f487482;
saved block5432.580620 seconds. Matched40-root prefix: long L10=4/40 (10%),
control L10=2/40 (5%); both L1000=0/40. Success identities unchanged:
long roots3,11,22,28, control11,22. All successful witnesses replayed in233;
no new success requires replay. No new CC/reference-cost improvement.

Worker79573 verified running100% CPU,910m21.32s accumulated versus
849m20.04s last hour:61m01.28s additional CPU. Controller574 remains12/18
jobs, tracking the existing worker. Completed control149 remains128/128,
6 distinct solved roots,L10=5/64,L1000=1/64,full replay audit passed.

Raw-initial139 unchanged80/128 conditions,480 episodes,278912 policy
neural evaluations,279312 full-path moves,0 successes;10 saved shards,
last log21:28:42 UTC. Latest shard SHA256
378a2f432c7f5c37e61a1dcb9394c71055ddd7bc62d05481b1930e8d480272f7.
Worker96642 verified running100% CPU,1658m50.77s versus1597m49.16s:
61m01.61s additional CPU. Controller20100 remains16/24 jobs. Actual
commands use canonical pgx-mcts-bench. CPU activity does not quantify
unsaved block progress; no restart based on observation timeout.

Integrity verification23:06:37 UTC passed1459/1273 controller input pins,
12/16 completed-report hashes, six roots/checkpoint hashes,16/16/10/10/16/16
saved shards and both149 final2048-step checkpoint hashes. Logs show no
traceback/runtime/error/nonfinite/NaN matches. No confirmed defect needs
fixing. No optimizer updates; current MCTS is evaluation, not training.

User-requested heuristic deferral remains: seek a substantial independent
neural signal, e.g. greater15%, before matched fully non-neural evaluation.
Current10% incomplete prefix does not reach that orientation or establish
superiority. Raw variant is neural, not a heuristic control. Limited-budget
unsolved cases remain censored, not proven negatives; overall goal unmet.

Next: continue healthy neural searches unchanged, replay new saved successes,
then full episode/tree audit at completion. Preserve dirty work, artifacts,
checkpoints and frozen inputs. No duplicate worker, restart, heuristic launch,
cloud purchase, database publication, commit, merge or push.
