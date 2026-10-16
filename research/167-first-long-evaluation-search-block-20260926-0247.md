# Hourly check: first155 search block committed

2026-09-26, 02:44–02:48 UTC. Read166 and inspected current manifests, actual
commands, CPU counters, logs and checksums. Both existing searches continue.

## New155-v2 result

Controller574/session17201 remains at8/18 completed jobs. Worker7097 committed
the first `control-initial-distance` block at02:44:06 UTC:8/128 conditions,
48 episodes,28568 policy neural evaluations,28608 full-path moves,0 solutions.
This is the shared INITIAL model, not either trained endpoint. Budget-limited
failure is not an impossible-state label or evidence that training is useless.

The block took9031.276221 seconds (150.5 minutes). Shard
`conditions-00000.json` SHA256:
`1c5c9c23c257a64d357be70b4abb8040c8d9119c378dfbae85f0228f5d72b70f`.
The log and manifest agree; actual file advancement is now established.
Worker100%CPU,152m17.89s CPU versus92m55.32s last hour; same canonical PGX command.

## Other work and integrity

Old139 controller20100/worker21103 remain live; worker100%CPU,480m33.83s CPU
versus421m11.85s previously. Its prefix stays56/128,336 episodes,188504 policy
evaluations,0 solutions; latest committed block remains23:27:11 UTC. The roughly
197-minute pending batch is below its historical260.8-minute maximum, which is
context, not a guarantee or a fixed timeout. No restart is warranted from this
observation alone.

Read-only verification session26742 exited0 at02:45:07 UTC: all1459 v2 and1273
old-controller input pins, all8 and12 completed report hashes, both roots hashes,
the new shard, all7 old shards and both149 final checkpoints match. No matching
traceback/runtime/nonfinite/NaN error in either log directory. Scientific
distance results are unchanged from165; no new training occurred.

## Additional finite prefix replay

Started a read-only replay check in session6914 using the existing
`audit_long_general_search.inspect_rows` on the captured eight-condition prefix
and its matching roots. This verifies path semantics/counts without running a
new search or changing files. At this note's creation the check is still running;
do NOT claim it passed, launch a second copy, or confuse it with the required
complete post-exit128-condition audit. Retrieve this session's result on the
next check if it has not returned before this turn ends.

Continue the unchanged registered search matrix and full post-exit audits.
Neural superiority over fully non-neural heuristics and public upper-bound
improvement remain unestablished. No source changes, cloud purchase, database
publication or Git publication were performed.
