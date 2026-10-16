# Hourly check: new search48/128, old search88/128

2026-09-26,10:49–10:51 UTC. Read174, then checked current manifests, live
commands, CPU counters, logs and hashes. Frozen conditions remain unchanged.

New155-v2 worker7097 committed its sixth block at10:03:06 UTC:40→48/128
conditions,240→288 episodes,114808→136080 policy neural evaluations,
136320 full-path moves,0 solutions. Block5097.299599s (85.0 minutes).
`conditions-00040.json` SHA256:
`934af459d3e4d5e30b77d479b5cf88720d41d44e290d9425c5f57a95510bac66`.
Log and manifest agree. Controller574/session17201 remains8/18 jobs; live worker
98.5%CPU,635m38.86s CPU versus574m31.34s last hour. This is still shared-initial
distance-prior search, not a trained endpoint or a fully non-neural control.

Old139 controller20100 remains12/24 jobs; worker21103 live at100%CPU and
963m50.27s CPU versus902m41.89s previously. Prefix stays88/128 conditions,
528 episodes,305128 policy evaluations,0 solutions; last block09:35:03 UTC.
Both actual commands use canonical pgx-mcts-bench; no restart or duplicate.

Read-only verification session11495 exited0 at10:50:28 UTC:1459/1273 controller
input pins,8/12 completed-report hashes, both roots hashes,6/11 committed shards
and both final149 checkpoint hashes match. No matched traceback/runtime error,
nonfinite or NaN indication in either log directory. No confirmed issue needed
a fix. Full post-exit semantic replay remains pending; only the first155 block
has extra prefix replay evidence (168). No new training or distance metrics.

Continue the unchanged registered evaluation matrix and hourly checks. The mixed
distance result in165 is not a search-utility claim. Minimum-CC superiority to
fully non-neural heuristics and public-bound improvement remain unestablished.
No source mutation, cloud purchase, database publication or Git publication.
