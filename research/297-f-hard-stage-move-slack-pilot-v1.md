# TRAIN hard-stage move-slack pilot

2026-10-07. Protocol295 finished 64/64 and its independent audit
(`artifacts/f-hard-stage-budget-yield-pilot-audit-v1/report.json`, SHA
`42d476bcb45bcdc91ed7677820c0a92bdff7992d54eb05828e418922e18721c7`)
found zero different-cost or F-usable groups at either 64 or 128
simulations. This result exposed a data-contract issue: all 287 stage-3
roster roots have `cap_moves == reference.moves` and
`cap_crossing_changes == reference.crossing_changes`. On the eight pilot
roots, all 45 raw found best witnesses had exactly the reference CC and
moves. The pilot consumed 3,953 CPU seconds. Increasing the tree budget
alone cannot test cost diversity well under these tight resource caps.

This separate TRAIN-only diagnostic retains the same eight score-blind
stage-3 roots from protocol295 (four CC1 and four CC2, indices
`[4,8,10,21,0,1,2,3]`), the same frozen roster and curriculum1024 actor,
and the canonical pgx-mcts-bench search settings. Change only the move
resource cap to `reference.moves + 16` or `reference.moves + 32`; keep
the crossing-change cap fixed. For each cap, run L10 and L1000, two fresh
independent seeds 20271361/20271362 and 128 actual tree simulations:
8 roots × 2 objectives × 2 seeds × 2 move slacks = 64 conditions.
Every search has one fresh tree and seed plus task index. Log CPU, wall,
neural evaluations, full episodes, found witnesses and TRAIN ownership.
Do not conflate the relaxed-cap cost distribution with protocol295 or
with a downstream evaluation at the original resource budget.

The independently audited output must verify every shard SHA, replay all
found paths, caps, and ownership of every crossing-change state. Within
the same root/objective/slack combine the two seeds' distinct admitted
completed paths with the constructive reference, then apply the exact F
rule (margin 2, max three positives, K=4 negatives, identical state and
remaining-resource contexts). Report exact F-usable roots per slack,
cost gaps, solve/admission fractions, and CPU/neural work per usable
root separately by L10/L1000 and CC1/CC2. Capped no-solve is censored.
This is a small diagnostic, not bulk teacher creation or network training.
Any scale-up requires a separately fixed contract and a representative
yield/CPU estimate. No DEV/TEST selection or superiority claim follows.
