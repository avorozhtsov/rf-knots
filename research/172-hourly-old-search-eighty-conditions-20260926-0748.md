# Hourly check: old search80/128, new search32/128

2026-09-26,07:47–07:49 UTC. Read171, then inspected current manifests, actual
process commands, CPU counters and logs. No restart, duplicate or source edit.

Old139 worker21103 committed another block at07:16:20 UTC:72→80/128 conditions,
432→480 episodes,256200→278912 policy evaluations,279312 full-path moves,
0 solutions. Block duration4425.177287s (73.8 minutes). Log and manifest agree.
`conditions-00072.json` SHA256:
`e78cf638dab2466eb3151beaa6eea5dd6a61a6f643ffa47c72f6e5800e426e4d`.
Controller20100 remains12/24 jobs. Worker99.3%CPU,783m13.13s CPU versus723m47.74s
last hour; same canonical pgx-mcts-bench command.

New155-v2 controller574/session17201 remains8/18 jobs, with live worker7097
at98.4%CPU and455m01.66s CPU versus395m35.09s last hour. Prefix stays32/128
conditions,192 episodes,90224 policy evaluations,0 solutions. No new block
since06:37:04 UTC. Both searches are still initial-model configurations; no
trained-endpoint search outcome or new optimizer step is reported.

Read-only verification session23332 exited0 at07:48:15 UTC:1459/1273 controller
input pins,8/12 completed report hashes, both roots hashes,4/10 committed shards
and both final149 checkpoint hashes match. No traceback/runtime/nonfinite/NaN
indication matched in either log directory. No confirmed issue needed a fix.
Full post-exit replay remains pending; first155 prefix replay passed in168.

Continue both unchanged searches and hourly checks. Completed distance results
remain those in165; improved minimum-CC solving, superiority to fully non-neural
heuristics and public upper-bound improvements remain unestablished. No cloud
purchase, database publication, checkpoint deletion or Git publication occurred.
