# Hourly check: old search64/128; first155 prefix replay passed

2026-09-26,03:46–03:48 UTC. Read167; inspected current manifests, actual commands,
logs, CPU counters and checksums. All frozen sources and experiments preserved.

## New completed work

The read-only eight-condition prefix replay started last hour in session6914
returned exit0. Existing `audit_long_general_search.inspect_rows` reconstructed
all48 episodes and28568 continuation moves (28608 full-path moves), matching
10005258 policy action scores and the captured search counters. This is only
the first155-v2 block, not the required post-exit128-condition audit, not an
additional MCTS run, and not an extra optimizer update. No need to rerun it.

Old139 worker21103 committed another eight-condition block at03:37:38 UTC:
56→64/128,384 episodes,226960 policy evaluations,227280 full-path moves,
0 solutions. Added48 episodes and38456 policy evaluations. Latest shard
`conditions-00056.json` SHA256:
`5cb9cbf6ca81e68178d74cd6179c23f7be4c69c8b2e7cf122d69c29962e5b598`.
Its elapsed batch time15025.664288 seconds (250.4 minutes) explains the preceding
hours without file advancement. This is halfway through this one configuration,
not halfway through the whole experiment. Controller20100 remains12/24 jobs.

## Current live state

New155-v2 controller574/session17201 remains8/18 completed jobs; worker7097 is
still `control-initial-distance`,8/128 conditions,48 episodes,0 solutions.
No new155 block since02:44:06. Worker99.3%CPU,214m17.39s accumulated CPU versus
152m17.89s last hour. These are shared initial weights, not a trained endpoint.

Old139 worker100%CPU,542m31.18s CPU versus480m33.83s last hour. Both workers are
the same live processes with canonical pgx-mcts-bench commands; no duplicates.

## Integrity / errors / next step

Read-only verification session8070 exited0 at03:47:09 UTC:1459/1273 controller
input hashes,8/12 completed report hashes, both roots hashes,1 new-evaluation
and8 old-evaluation committed shards, and both final149 checkpoints all match.
No matched traceback/runtime/nonfinite/NaN error in either log directory; log
counts agree with the newly committed old139 manifest. No fix was needed.

Continue both searches unchanged and let controllers perform their registered
full post-exit replay sequence. Distance results remain those reported in165:
general calibration improves with long-pair training, short-u worsens. Search
utility, minimum-CC advantage over fully non-neural heuristics and public-bound
improvement remain unestablished. No source mutation, cloud purchase, database
publication or Git publication occurred. Next routine check remains hourly.
