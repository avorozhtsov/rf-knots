# Hourly check: new search24/128, old search64/128

2026-09-26,05:46–05:48 UTC. Read169 before inspecting actual processes,
manifests, logs and checksums. Corrected155 evaluation remains in the v2 output;
do not launch the original v1 controller mentioned in the heartbeat template.

New155-v2 worker7097 committed its third block at05:10:04 UTC:16→24/128
conditions,96→144 episodes,48160→70488 policy neural evaluations,
70608 full-path moves,0 solutions. Latest block4707.181786s (78.5 minutes).
Shard `conditions-00016.json` SHA256:
`b0ce9869d742cacee96fa9c408a4f0be28825dcf1945bfa36b8b3b2b7c85c6ae`.
Log and manifest agree. Controller574/session17201 remains8/18 jobs because
this first configuration is unfinished. It uses shared INITIAL weights, not
either trained endpoint. Worker99.3%CPU,334m11.27s CPU versus274m42.09s last hour.

Old139 controller20100/worker21103 remain live at64/128 conditions,384 episodes,
226960 policy evaluations,0 solutions. No new block since03:37:38 UTC;
controller12/24 jobs. Worker99.6%CPU,662m24.42s CPU versus602m55.79s last hour.
Both actual commands use canonical pgx-mcts-bench. No duplicate or restart.

Read-only verification session93811 exited0 at05:47:11 UTC. All1459/1273
controller input pins,8/12 completed-report hashes, both roots hashes,3/8
committed search shard hashes and both final149 checkpoint hashes match.
No traceback/runtime/nonfinite/NaN indication matched in either log directory.
No confirmed problem needed a fix. Only the first155 block has the additional
prefix semantic replay documented in168; full post-exit replay remains required
for all conditions. Hash verification is not a substitute for that replay.

No new training updates or distance results this hour. Continue the registered
search matrix under unchanged conditions and inspect again on the hourly
schedule. The mixed distance result in165 remains the latest model-quality
evidence; improved minimum-CC unknotting, superiority over non-neural heuristics
and public upper-bound improvements are not established. No cloud purchase,
source mutation, database publication or Git publication occurred.
