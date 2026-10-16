# First trained135 search witnesses verified; incomplete comparison

2026-09-26,17:54–17:56 UTC. Read181 before checking manifests, actual processes,
logs and hashes. Both existing workers retained; frozen conditions unchanged.

## First positive trained-search signal

Old139 units-final-distance committed its first block at17:06:10 UTC:8/128
conditions,47 episodes,21810 policy evaluations,21851 full-path moves,
2 solved conditions. Block5463.132187s (91.1min). Shard SHA256:
`7a498686544a8166c00c614a516241455eef8da25e29043b9dfec9b77e34da22`.

The two successes are the SAME panel root (index3) at two cost weights, not two
distinct roots/knots. There are four roots in this eight-condition prefix.
Initial weights solved neither matching condition. Read-only session43739
replayed both saved `UnknotWitness` certificates, checked exact root identity,
move/CC caps and recalculated L10/L1000. Both are valid unknotting paths:

| Condition | Search weight | CC | Moves | L10 | L1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
|6|10|1|97|107|1097|
|7|1000|1|94|104|1094|
|existing reference for root3|—|1|92|102|1092|

The previously audited root has certified lower_cc1, so the found CC count1
attains its known minimum. However the reference already had one CC and fewer
moves: these do NOT improve its CC upper bound or its L10/L1000 costs. This is
an early successful-solving difference versus initial weights, not a new public
theorem or a complete statistical/model comparison. Both successes come from
the earlier135 trained units model, NOT149 long-data training. Do not attribute
the gain specifically to embeddings; multiple model parameters were trained.

Only the two success certificates were additionally replayed now, not every
episode/tree prefix of this new block. Full semantic search/accounting replay
remains scheduled after all128 conditions finish. No output was promoted and no
TEST trajectory entered training. Fully non-neural comparison remains required.

## Live state and integrity

Old controller20100 remains14/24 jobs, worker47974 live at99.5%CPU with132m59.75s
CPU versus75m43.02s last hour. New155-v2 controller574/session17201 remains8/18
jobs; worker7097 live at99.5%CPU with1040m43.34s CPU versus983m26.72s. Its initial
search remains56/128,336 episodes,174344 policy evaluations,0 solutions; no new
block since15:00:34 UTC. Both commands use canonical pgx-mcts-bench.

Session43739 exited0 at17:55:10 UTC:1459/1273 controller input hashes,8/14
completed-report hashes, all three roots/search checkpoint hashes and7/1/16
committed shard hashes match. No traceback/runtime/nonfinite/NaN error matched
in either log directory. No confirmed issue needed a fix.

Continue both frozen searches and complete paired comparisons/replays. No new
optimizer updates, source changes, duplicate/restart, cloud purchase, database
publication or Git publication occurred. Next routine observation is hourly.
