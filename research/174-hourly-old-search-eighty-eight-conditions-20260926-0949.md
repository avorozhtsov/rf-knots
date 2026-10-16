# Hourly check: old search88/128, new search40/128

2026-09-26,09:48–09:50 UTC. Read173, then inspected actual process commands,
CPU counters, manifests, logs and hashes. Both existing searches retained.

Old139 worker21103 committed its eleventh block at09:35:03 UTC:80→88/128
conditions,480→528 episodes,278912→305128 policy neural evaluations,
305568 full-path moves,0 solutions. Block8320.927907s (138.7 minutes).
`conditions-00080.json` SHA256:
`664c1c887afd62e87110ba63652346edd46a27c5d72945a008ee7db4e661335a`.
Log and manifest agree. Controller20100 remains12/24 jobs; worker99%CPU,
902m41.89s CPU versus842m05.46s last hour.

New155-v2 controller574/session17201 remains8/18 jobs, with live worker7097
at99%CPU and574m31.34s CPU versus513m53.73s previously. Prefix remains40/128,
240 episodes,114808 policy evaluations,0 solutions. No new block since08:38:05
UTC. Both searches still evaluate INITIAL weights, not either149 trained
endpoint. Canonical pgx-mcts-bench commands unchanged; no duplicate or restart.

Read-only verification session35618 exited0 at09:49:14 UTC:1459/1273 controller
input pins,8/12 completed report hashes, both roots hashes,5/11 committed
search shards and both final149 checkpoint hashes match. No matching traceback,
runtime error, nonfinite or NaN indication in either log directory. No confirmed
issue needed a fix. Full post-exit semantic replay remains pending; first155
block's separate prefix replay passed in168. Hash checks do not replace replay.

No new training or distance results this hour. Continue the unchanged registered
matrix and hourly checks. Finite-budget failures are not impossibility labels;
search utility, minimum-CC advantage over fully non-neural heuristics and public
upper-bound improvements remain unestablished. No source mutation, cloud
purchase, database publication or Git publication occurred.
