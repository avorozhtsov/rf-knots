# Clean paired iteration: completed MCTS conditions

2026-09-20. This is an intermediate report for completed whole MCTS conditions
from note33/35. Both CC weights10/1000 are complete, as are all controller-owned
comparisons and the charged rollout portfolio. The frozen protocol is unchanged.
This note covers the MCTS conditions; final42-job results are in note39 and
`artifacts/embedding-clean-paired-summary-v1/report.md`. No model is promoted.

## Completed CC weight10 conditions

All conditions use the same55 query fingerprint groups, model-blind shortest
reference witness per group, two seeds,512 simulations per tree, widening4,
root noise0.25, default catalogue proposals and witness-feasible resource caps.
Parent uses constant0.5. The two trained arms are each tested with constant0.5
and learned leaf value. These are the same two completed2048-step checkpoints,
not intermediate weights selected using this panel.

| Model | Leaf value | Solved / trees | Simulations | Neural-state evaluations | Action scores | Search seconds |
|---|---|---:|---:|---:|---:|---:|
| Parent | constant0.5 | 0/110 | 56320 | 56306 | 489452 | 192.656711 |
| Value-supervised | constant0.5 | 0/110 | 56320 | 56299 | 511925 | 191.849059 |
| Value-supervised | learned | 0/110 | 56320 | 56299 | 431431 | 193.868018 |
| Control | constant0.5 | 0/110 | 56320 | 56307 | 507735 | 192.890876 |
| Control | learned | 0/110 | 56320 | 56305 | 615326 | 182.572796 |

Each completed report's bytes were checked against the controller's stored
hash, its checkpoint SHA against the pinned final model, and all110 query/seed
rows using the independent MCTS comparison validator. That validator checks
panel roots, CC/resource caps, coverage, counts and failure censoring. There
are no successful witnesses in these conditions to replay or compare by cost.

The same-checkpoint Value constant-to-learned comparison passed the explicit
`same_model_leaf_value` gate. It changed the measured number of action scores
511925 ->431431 but did not change solved coverage. This establishes a change
in measured search work, not a benefit in solutions and not a claim about the
identity of every visited state. Parent-constant vs Value-constant also passed
the matched-model comparison gate, with110 both-unsolved pairs. Their
common-solved cost delta is UNAVAILABLE, not zero.

The five conditions contain281600 simulation visits and281516 neural-state
evaluations in total. They are NOT281600 distinct knot representations or550
independent knot tests: the same55 groups recur across models, seeds and modes.
All failures are finite-budget observations, not evidence of unsolvability.

## Immutable completed report hashes

Paths are inside `artifacts/embedding-clean-paired-evaluation-v1/mcts/`.

| Directory (report.json) | SHA256 |
|---|---|
| mcts-parent-constant-10 | 622711b1eef3dc2cba143e33b01b416787d8bcdcedddaf2e1dd5d1b6fe55aa20 |
| mcts-value-constant-10 | 1c947fb6d63ea47b1819d0dba95dfd8ed9345e61f99b94e98a32a32112d58380 |
| mcts-value-learned-10 | 087d6f6d2600284e975c0f40d27730beba5ceff8fd5069aa32927659e1d6041a |
| mcts-control-constant-10 | 01a1f0f553f808ecddadf42a814ea51b78a294d8086713066592ac1e5175641f |
| mcts-control-learned-10 | af4e1f6568a9c5e41469264e989c66df8870fa6bc46b45603464ad29e1f30144 |
| mcts-parent-constant-1000 | 867950b8860d678e7b87b8ab88b9440350343dfd856b50f4544b28d699bbe02a |
| mcts-value-constant-1000 | 7d07fef72c1b41d1dfc43904e94fd9448f671eac22d124111b89d96f00f2a5fe |
| mcts-value-learned-1000 | 06a99f3d9497fa33e468c7a2f886d4dffc2da2d3b8cea105e56afcea99c20a7c |
| mcts-control-constant-1000 | caa3d4f9f4c6a0d5797c422e77f2c2e294a1b705cea1925c35c9a17c9e0afec3 |
| mcts-control-learned-1000 | 2a36818dc6224b399c137e331d831e17511e73b7a8181dbf951b6a2a2d4839a2 |

## Completed CC weight1000 conditions

The same roots, seeds, simulation counts and validation checks apply.

| Model | Leaf value | Solved / trees | Simulations | Neural-state evaluations | Action scores | Search seconds |
|---|---|---:|---:|---:|---:|---:|
| Parent | constant0.5 | 0/110 | 56320 | 56277 | 457510 | 189.672624 |
| Value-supervised | constant0.5 | 0/110 | 56320 | 56297 | 460300 | 191.204392 |
| Value-supervised | learned | 0/110 | 56320 | 56298 | 469177 | 186.254865 |
| Control | constant0.5 | 0/110 | 56320 | 56283 | 460439 | 188.883542 |
| Control | learned | 0/110 | 56320 | 56293 | 518053 | 190.813505 |

All ten reports were independently revalidated after the last condition ended:
report bytes match controller hashes, checkpoint hashes match the respective
final pins, complete110-row coverage and censored failures pass the comparator.
Both within-model constant-to-learned comparisons at weight1000 passed the
`same_model_leaf_value` gate, with110 both-unsolved and common-only cost delta
UNAVAILABLE in each case. Changing leaf value changes measured search work,
but there is no solution gain at this budget.

The full ten-condition series consumed563200 simulation visits and562964
neural-state evaluations. All1100 trees were unsolved. These are repeated
conditions on55 fingerprint groups, NOT1100 independent knot tests or563200
unique representations.

## Interpretation boundaries and next steps

No MCTS solved-coverage benefit has appeared at either cost weight and this budget.
Because all conditions hit a zero-solve floor, cost-quality comparisons cannot
be estimated. This does not invalidate the replayed cost-search solutions in
note37: that is a different search algorithm, selection and compute contract.
It also does not prove the absence of useful information in the global
embedding, or that larger joint RL training cannot exploit it.

Value was trained to predict ONE rollout of the pinned parent behaviour within
specified resources. It was not trained as the exact success probability of
this updated-policy MCTS. Lack of transfer here must not be inflated into a
general impossibility statement or used to relabel these roots as negatives
for mathematical solvability.

All predeclared paired comparisons and the parent-rollout portfolio completed
at the charged neural budget. Retain this negative result; do not change
budgets, roots or seeds to obtain a preferred outcome. Controller99732/
session13956 completed42/42 and exited with code0. This is completed work,
not an active-process claim. The portfolio outcome is documented in note39.

Source inspection confirms that `Tree._record_solution` stops at the first
solution and terminal backup defaults to1 regardless of path cost. CC weight
is an input context, not an anytime cost-minimization objective. This is a
disclosed scope limitation, not a diagnosed cause of the zero-solve outcome.
Changing this contract requires a new search experiment; it must not modify
the active frozen evaluation.
