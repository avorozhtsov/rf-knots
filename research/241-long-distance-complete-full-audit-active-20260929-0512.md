# Hourly check: long distance search completed; full audit active

2026-09-29 05:09–05:12 UTC. Read240 and current manifests before live
process commands, logs, checksum verification and independent witness replay.

Long-final149 distance completed128/128 conditions at05:07:21 UTC:
759 episodes,369482 policy neural evaluations,370125 full-path moves,
9 successful CONDITIONS on8 distinct ROOTS,16 saved shards. Manifest SHA256
9aab0f63ed6423c67d253e8a3a07976685d435fafcfbe3ae71eae03eea9ad3d4.
Latest conditions-00120.json SHA256
4ec7a7f077edf08b23243ec2ee605312ffb7609ee7614eb6f16e28f3b3edd9ad.
Old worker79573 absent, matching normal completed state; do not resume it.
Controller574 advanced12→13/18 jobs and launched the declared full audit.
Audit19649 actually running99.0% CPU,3m14.91s, command
audit_long_general_search.py --search long-final-distance --out replay-audit.json.
Full all-episode/tree audit is ACTIVE, not yet passed; no duplicate audit.

Independently replayed all9 long and6 control successful full witnesses,
checking native replay, every semantic prefix, exact starts, action-space
dimensions, CC/move caps and L10/L1000 arithmetic. New long successes:
- condition116/root58/L10:CC1,moves247,L10=257,L1000=1247;
  reference103/1093 implies93 moves,154 extra at equalCC;
- condition121/root60/L1000:CC1,moves306,L10=316,L1000=1306;
  reference100/1090 implies90 moves,216 extra at equalCC.
Neither improves reference CC or costs. All nine long routes have equalCC
to their reference and higher reference costs. Successful-witness replay
does not replace the still-running all-episode/tree/accounting audit.

Full matched64-root comparison: long L10=6/64 (9.375%),L1000=3/64 (4.6875%),
either objective=8/64 (12.5%); control L10=5/64 (7.8125%),L1000=1/64
(1.5625%),either=6/64 (9.375%). Long solved3,11,22,28,47,49,58,60;
control11,22,45,47,54,63. Long-only3,28,49,58,60; control-only45,54,63.
Coverage difference2 roots, not3 successful conditions; shared root47 counts
once despite two objective successes. Shared L10 paths remain longer for long
on roots11/22/47 by52/120/80 moves at equalCC. Logical compute counts:
long369482 policy evaluations/370125 moves versus control371078/371722.
This is a matched neural comparison, not a fully non-neural benchmark or
proof of superiority; runtime comparisons need separate compute accounting.

Raw-initial139 advanced112→120/128 conditions,672→720 episodes,
363664→381384 policy neural evaluations,364224→381984 full-path moves,
still0 successes;15 shards. Latest log04:53:46 UTC, latest shard SHA256
f09a99d5a402304e3b6f8534b103c73d1c5a989f3e57dce65967d573f6708c57.
Worker96642 actually running98.8% CPU,2021m18.71s versus1960m50.49s
in240:60m28.22s additional CPU. Controller20100 remains16/24 jobs.
Actual controllers/search commands use canonical pgx-mcts-bench.

Integrity verification05:10:53 UTC passed1459/1273 input pins,13/16
completed-report hashes, six roots/checkpoint hashes,16/16/16/15/16/16
saved shards and both149 final2048-step checkpoint hashes. Logs contain no
traceback/runtime/error/nonfinite/NaN matches. No confirmed defect needs
fixing or new optimizer updates; MCTS is evaluation, not training.

Heuristic launch remains deferred: complete long panel12.5% is below user's
greater15% orientation. Next: consume and independently validate the full
audit when finished; let existing controllers advance the remaining frozen
neural matrix without duplicate workers. Overall goal not achieved. Raw is
neural, not heuristic; finite-budget failures remain censored. Preserve dirty
work, frozen conditions, artifacts and checkpoints. No restart, cloud purchase,
heuristic launch, publication, commit, merge or push.
