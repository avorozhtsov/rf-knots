# Tight-cap hard-stage audit and move-slack launch

2026-10-07. Protocol295's 64/64 TRAIN searches completed; manifest SHA
`ff9ef48f6677e3928dc519699b4cc112eac0a45d210fcf4fa1421c52530499cd`.
The independent audit in
`artifacts/f-hard-stage-budget-yield-pilot-audit-v1/report.json` SHA
`42d476bcb45bcdc91ed7677820c0a92bdff7992d54eb05828e418922e18721c7`
checked every shard SHA, task/cap identity, full replay of every found
solution and TRAIN ownership of every crossing-change state. Across 64
searches there were 45 raw solved attempts, 33 attempts with at least one
admitted complete path, 4,725 episodes, and 3,953.0 measured CPU seconds.
Neither 64 nor 128 simulations produced any different-cost root/objective
group or exact F-usable group. The 45 raw best witnesses all matched the
constructive reference CC and move count exactly.

The key data-contract finding is not a model failure: in **all 287**
stage-3 roster roots, `cap_moves == reference.moves` and
`cap_crossing_changes == reference.crossing_changes`. Therefore an equal-CC
completed path cannot be more expensive in moves; the task resources
structurally suppress the intended F cost comparison. Stages 0/1 have
move slack and stage 2 is mixed. More tree simulations under the tight
stage-3 caps were a poor way to create F negatives.

A separate protocol297 changes only the move cap by +16 or +32 and keeps
the CC cap unchanged. Source
`scripts/collect_f_hard_stage_move_slack_pilot_v1.py` SHA
`8e498c91a0b8b358d6b6c6859b4b3c48fb458ef24aec4474888f5a5bae08228b`
started in `artifacts/f-hard-stage-move-slack-pilot-v1`, contract SHA
`a528d2872fb45a6a1f773c2b4b545d9dc751a354c125b5abc800c2bc53f9387a`.
Its first shard SHA was independently checked, and its found witness,
start/caps and all 90 episode ownership flags independently replayed.
With +16 moves, one CC1 L10 root had 25 admitted completed paths with
observed costs 87, 89, 91, 92, 95, 96, 98, 101; the exact F rule yielded
one nonzero group with six action terms. This is one diagnostic condition,
not population yield or downstream improvement. The verified one-shard
process exited and the same source/artifact was resumed; one worker was
checked consuming one CPU core. Full 64-condition audit remains pending.

Neither pilot is optimizer training; the R2/R4 24-hour clock has not
started. Do not infer from more expensive synthetic-search paths that the
network will improve on held-out search. Need full audited yield/CPU before
any bulk teacher collection, then independent DEV and matched non-neural
control before a superiority claim.
