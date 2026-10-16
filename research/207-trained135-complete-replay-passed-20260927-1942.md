# Trained135 search complete and full replay passed:16/64 roots

2026-09-27 19:40–19:42 UTC (20:40–20:42 London). Read206 and current manifests
before inspecting actual commands, processes, logs, CPU and artifact hashes.

## Completed135 units-final-distance

Search completed18:56:56 UTC, all128 conditions (64 roots ×2 search weights),
753 episodes,381160 policy evaluations,381811 full-path moves. Twenty successful
conditions cover16 distinct roots. Full episode/tree/accounting replay PASSED;
controller20100 advanced16/24 jobs. Read-only session12180 independently
reconsumed both completed search and replay reports with
run_resource_evaluation.verify_job, including pinned inputs/model/coverage.
This consumes the full replay certificate, not a second full episode replay.

Manifest SHA256:
7f554daac847c33cdfd9103ac6869b67594987a69824fc5cc8f11b41f3c2926a.
Full replay report SHA256:
37145747f3087fb40b7e8ea68cf671f58cc6464af52848b8aee00e8784d85b99.

| Search configuration | Initial weights solved | Trained units solved |
| --- | ---: | ---: |
|L10|0/64|9/64|
|L1000|0/64|11/64|
|Either weight, distinct roots|0/64|16/64|

This is a complete descriptive paired result on the declared synthetic TEST
panel, not superiority over fully non-neural heuristics, nor an isolated
embedding effect. Raw-resource and uniform-tree comparisons remain unfinished.
Uniform tree priors with neural continuations are not non-neural controls.
Do not tune training on this TEST result. Initial search used768 episodes and
395312 policy evaluations, so observed work is not identical despite the matched
declared protocol. Host sleep/shared execution also prevents naive speed claims.

Of20 successful trajectories,19 haveCC1 and oneCC2. Nineteen match certified
lower_cc; root22 remains interval[1,2], so itsCC2 is not proven optimal.
Six match existing reference L10/L1000 costs; zero improve reference costs or
CC upper bounds. No public named-knot bound improvement.

Session12180 additionally replayed the final TWO new success witnesses against
their exact roots and resource caps, recomputing costs:

| Condition / root / search | CC | Moves | L10 | L1000 |
| --- | ---: | ---: | ---: | ---: |
|124 /62 /L10|1|87|97|1087|
|125 /62 /L1000|1|85|95|1085|
|Reference root62|1|85|95|1085|

Root62 lower_cc1; L1000 run matches reference, L10 run uses two extra moves.

## Actual next workers

Controller20100 automatically launched raw-initial-distance PID96642 around
19:16:22 UTC. Fresh manifest19:18:56 UTC,0 committed conditions/shards; actual
command uses canonical PGX. Live100% CPU with23m44.74s CPU after24m13s elapsed.
Old units-final worker47974 is absent after normal completion; do not restart.
Raw-initial is the next preregistered resource-input variant, not a heuristic.

155-v2 trained149 control remains48/128,286 episodes,132854 policy evaluations,
133096 full-path moves,2 solutions. Last manifest18:22:09 UTC;6 shards.
Controller574 remains10/18 jobs; worker36622 live100% CPU, accumulated571m25.59s
versus512m07.03s last hour. No new result; long-data149 arm still queued.

## Integrity and continuation

Read-only session12180 exited0 at19:41:31 UTC. All1459/1273 controller input
pins,10/16 completed-report hashes, five search roots/checkpoint hashes,
6/16/0/16/16 shards and both149 final2048-step checkpoints match. No traceback/
runtime/error/nonfinite/NaN matches in logs. No confirmed defect required a fix.
Both running worker commands use canonical pgx-mcts-bench.

Continue the unchanged matrix, full replays and final comparisons. No optimizer
updates this hour. Completion of this configuration does not complete the overall
goal. No duplicate, interruption, frozen-source mutation, cloud purchase,
database publication or Git action. Dirty work and artifacts preserved.
