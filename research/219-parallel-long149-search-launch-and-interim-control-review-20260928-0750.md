# Interim review and parallel registered long149 evaluation

2026-09-28 07:48–07:50 UTC (08:48–08:50 London). User questions waiting for
128/128 and proposes parallel neural unknotting evaluation. Both existing
workers already perform neural-guided unknotting; long-final149 had remained
queued behind control-final149. No need to postpone diagnostic interpretation
until all128 conditions; retain full frozen denominator for definitive results.

Current control-final-distance prefix is96/128 conditions,573 episodes,
301467 policy neural evaluations,301950 full-path moves,4 successful conditions
covering4 distinct roots. Conditions are64 roots at two objectives, not128 knots.
On the evaluated48-root prefix: L10=4/48, L1000=0/48, either=4/48. Initial
weights on identical first96 conditions had0 successes. Small positive evidence
relative to initialization, weak absolute solve rate, not a heuristic comparison
and not isolated evidence for the embedding. Registered budget is8 MCTS
simulations per condition, CPU single-thread. Prefix root order is not a random
interim statistical stopping rule; do not extrapolate or tune on this TEST.

Read-only independent replay verified ALL four successful full witnesses against
their exact roots, action-space dimensions and CC/move caps, recomputing costs:

| Condition | Root | Objective | CC | Moves | L10 | L1000 |
| --- | --- | --- | --- | --- | --- | --- |
|22|11|L10|1|126|136|1126|
|44|22|L10|2|112|132|2112|
|90|45|L10|2|158|178|2158|
|94|47|L10|1|102|112|1102|

Shard hashes verified before witness replay. This is successful-witness replay,
not the eventual full all-episode/tree/accounting audit. Completed units135
previously solved16/64 roots on a DIFFERENT frozen panel; not a direct149 comparator.

Launched the EXISTING registered search-long-final-distance job in parallel,
not a new protocol, duplicate or altered source/configuration. Checked1459
controller input pins and actual matching commands; no existing matching worker,
output directory or log. Exact command is manifest jobs entry with arm=long,
stage=final,prior=distance,canonical pgx-mcts-bench, output:
artifacts/raster-long-general-evaluation-v2/mcts/long-final-distance.
Fresh log: artifacts/raster-long-general-evaluation-v2/logs/search-long-final-distance.log.
Detached worker PID79573 LAUNCHED; actual command verified and CPU advanced
26.07→68.64s,99.7% CPU at74s elapsed. VERIFIED ACTIVE in initialization;
no search manifest or completed block yet at07:50:02 UTC. Optional Plink GUI
warning only. Do not claim evaluated conditions or completed launch prerequisites
beyond the checks actually reported. Next hourly run must inspect this worker,
fresh log and manifest even if the manifest has not yet appeared.

Existing control36622/raw96642 continue actually running; no interruption.
Controller574 remains processing control; it already supports matching-worker
adoption and completed-output reuse when reaching the long job. Do NOT launch a
second controller or duplicate long worker. No dispatcher manifest or frozen
inputs were edited. Concurrent CPU workers may contend for resources; wall-time
claims need separate accounting, not naive comparison of shared-host speed.

Next: verify long worker reaches a fresh search manifest, monitor all three at
hourly cadence, compare paired prefix diagnostically and complete full replay
and fully non-neural matched evaluation before a superiority claim. No new
optimizer training, cloud purchase, database publication, commit, merge or push.
