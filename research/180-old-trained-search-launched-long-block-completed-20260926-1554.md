# Hourly check: full139 replay passed; trained search active;155 block advanced

2026-09-26,15:53–15:55 UTC. Read179; checked manifests, real process commands,
CPU counters, logs, checkpoint/report hashes and the completed replay consumer.

##139 advanced to trained weights

Full `units-initial-distance` replay passed:128 conditions,768 episodes,
395312 continuation moves,395952 full-path moves,167955508 policy action scores.
All successful/censored paths (here no successes) were replayed and reference
witness costs certified by the completed auditor. Recorded timing/tree/cache
counters are not an independent compute rerun. Initial result remains0 solutions;
finite-budget failure is not an impossible-state or bad-action label.

Replay report SHA256:
`67130c69cbdd7798b32d991d3b97d677508e42c2579440ad70dd2dc134a1fc7e`.

Controller20100 advanced to14/24 completed jobs and automatically launched
`search-units-final-distance` at15:32:25 UTC, worker47974. Actual command uses
canonical pgx-mcts-bench and the trained135 units checkpoint after2048 updates:
`cd1fe65ec47a48658a9c892852c63d464bfe4c9d52bc26a833e428e1fd1e365a`.
This is the earlier resource-units experiment's trained arm, NOT149's long-data
model. Manifest published15:35:05;0 committed conditions and empty counts so far.
Live worker59.5%CPU,19m15.88s CPU in20m59s elapsed. No scientific result yet.

##155-v2 delayed block really completed

Worker7097 committed conditions48..55 at15:00:34 UTC:48→56/128 conditions,
288→336 episodes,136080→174344 policy neural evaluations,174624 full-path moves,
0 solutions. Block17845.245028s (297.4min,4.96h). This confirms the previous
long pause between file updates ended normally; no restart occurred.
`conditions-00048.json` SHA256:
`f8536e19aa5345fae7ec3f5c2f352e00ff48a680827b5c82961474c31a6a4dbb`.

Controller574/session17201 remains8/18 jobs, current control-initial-distance.
Worker59.7%CPU,927m01.13s CPU versus869m14.89s last hourly initial observation.
These remain149 shared-initial weights. CPU percentages are instantaneous
shared-host observations, not controlled speed comparisons. No new optimization.

## Verification / next step

Read-only session47756 exited0 at15:54:08 UTC: all1459/1273 controller input pins,
8/14 completed report hashes, all7/16 committed search shards, all three search
roots/checkpoints, and both149 final checkpoints match. Reconsumed the full139
replay report through the registered verifier, matching controller's receipt.
No traceback/runtime/nonfinite/NaN indication matched in either log directory.
No confirmed issue needed a fix; no second audit/search was launched manually.

Continue the frozen trained135 search and155 initial search, with hourly checks.
Complete remaining comparisons and replays before claiming utility. General
distance calibration gains from149 do not establish minimum-CC solving benefit;
superiority over fully non-neural heuristics and public-bound improvement remain
unestablished. No source mutation, cloud purchase, database or Git publication.
