# Hourly check: old search104/128, new search48/128

2026-09-26,12:50–12:52 UTC. Read176, then inspected current manifests, actual
commands, CPU counters, logs and hashes. Frozen conditions remain unchanged.

Old139 worker21103 committed another block at12:28:35 UTC:96→104/128 conditions,
576→624 episodes,330096→347048 policy neural evaluations,347568 full-path moves,
0 solutions. Block2956.673870s (49.3 minutes). Log and manifest agree.
`conditions-00096.json` SHA256:
`b5169361864479d712500cd05c1f1fa506138c2a32a5d5c9d981e2ff48c25e22`.
Controller20100 remains12/24 jobs; worker99.6%CPU,1084m10.57s CPU versus
1023m15.79s last hour. There are24 conditions left in this configuration,
not24 conditions left in the complete evaluation.

New155-v2 controller574/session17201 remains8/18 jobs; live worker7097 at99.8%CPU,
755m59.27s CPU versus695m04.62s last hour. Prefix stays48/128 conditions,
288 episodes,136080 policy evaluations,0 solutions. No new block since10:03:06
UTC. Within-batch outcomes are not exposed; continued CPU use is not a claim of
new file progress. Both searches still use initial weights and canonical PGX.
No duplicate, restart or interruption occurred.

Read-only verification session96766 exited0 at12:51:15 UTC:1459/1273 controller
input pins,8/12 completed report hashes, both roots hashes,6/13 committed shards
and both final149 checkpoint hashes match. No matched traceback/runtime error,
nonfinite or NaN indication in either log directory. No confirmed issue required
a fix. Full post-exit semantic replay remains pending; first155 prefix replay
passed in168. No new optimizer updates or distance metrics this hour.

Continue both unchanged searches and hourly checks. Minimum-CC search benefit,
superiority over fully non-neural heuristics and public-bound improvements are
not established. No source mutation, cloud purchase, database publication or
Git publication occurred.
