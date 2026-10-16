# Hourly check: old search112/128; larger new-search batch still active

2026-09-26,13:51–13:53 UTC. Read177 before inspecting manifests, live commands,
CPU counters, logs, roots and checksums. No frozen conditions changed.

Old139 worker21103 committed a block at13:12:47 UTC:104→112/128 conditions,
624→672 episodes,347048→363664 policy neural evaluations,364224 full-path moves,
0 solutions. Block2650.169894s (44.2 minutes). Log and manifest agree.
`conditions-00104.json` SHA256:
`b5a1705d178cc4e1430dd38bbed25d0543fb4ef48710619de43d45f42d268275`.
Controller20100 remains12/24 jobs; worker98.5%CPU,1142m39.59s CPU versus
1084m10.57s last hour. Sixteen conditions remain in this configuration only.

New155-v2 controller574/session17201 remains8/18 jobs; worker7097 live at98.7%CPU,
814m27.65s CPU versus755m59.27s last hour. Prefix remains48/128,288 episodes,
136080 policy evaluations,0 solutions. No committed block since10:03:06 UTC.
Continued CPU activity does not reveal within-batch completion.

Read-only inspection of frozen roots shows pending conditions48..55 have input
lengths347..401 and native move caps748..868, versus120..245 and288..540 for
the preceding40..47 block. Larger words/caps plausibly increase runtime, but
this is not a measured causal decomposition or proof of healthy inner-loop
progress. No stopped process or logged error was found; stale batch timestamps
alone do not justify restarting. Keep both unchanged searches.

Verification session71625 exited0 at13:51:42 UTC:1459/1273 controller input pins,
8/12 completed-report hashes, both roots hashes,6/14 committed search shards
and both149 final checkpoint hashes match. No matched traceback/runtime error,
nonfinite or NaN indication in either log directory. No confirmed bug needed
a fix. Both commands use canonical pgx-mcts-bench and initial weights.

No trained-endpoint search result, new training step or distance metric this
hour. Full post-exit semantic replay remains pending; first155 prefix replay
passed in168. Continue registered evaluation and hourly checks. Minimum-CC
advantage over fully non-neural heuristics and public-bound improvements remain
unestablished. No restart, duplicate, source mutation, cloud purchase, database
publication or Git publication occurred.
