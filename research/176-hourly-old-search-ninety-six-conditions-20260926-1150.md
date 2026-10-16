# Hourly check: old search96/128, new search48/128

2026-09-26,11:49–11:51 UTC. Read175 before inspecting current manifests, live
commands, CPU counters, logs and checksums. No changes to frozen conditions.

Old139 worker21103 committed its twelfth block at11:39:16 UTC:88→96/128
conditions,528→576 episodes,305128→330096 policy neural evaluations,
330576 full-path moves,0 solutions. Block7451.697457s (124.2 minutes).
`conditions-00088.json` SHA256:
`ce9ead4748a5ae8274e8a323b82c9a1fec66da96250a4cdccd84287bb0ddd3e9`.
Log and manifest agree. Controller20100 remains12/24 jobs; live worker100%CPU,
1023m15.79s CPU versus963m50.27s last hour. This is75% of this configuration's
conditions, not75% of the complete evaluation.

New155-v2 controller574/session17201 remains8/18 jobs; live worker7097 at100%CPU,
695m04.62s CPU versus635m38.86s previously. Prefix stays48/128,288 episodes,
136080 policy evaluations,0 solutions. No new block since10:03:06 UTC. Both
searches retain canonical pgx-mcts-bench commands and still use initial weights;
neither trained149 endpoint's search result is available. No duplicate/restart.

Read-only verification session70250 exited0 at11:50:09 UTC:1459/1273 controller
input pins,8/12 completed report hashes, both roots hashes,6/12 committed search
shards and both final149 checkpoint hashes match. No matching traceback/runtime
error, nonfinite or NaN indication in either log directory. No fix was needed.
Full post-exit semantic replay remains pending; first155 prefix replay passed
in168, which does not certify subsequent blocks.

Continue the unchanged search matrix and hourly checks. No new training or
distance result this hour. Minimum-CC benefit, superiority over fully non-neural
heuristics and public-bound improvements remain unestablished. No source change,
cloud purchase, database publication or Git publication occurred.
