# Completed common-domain witness-access gate

2026-09-21. Protocol80 completed, exit0. No network-selected search was run.

Artifacts: `artifacts/qrun-common-witness-access-v1/`.
Panel SHA256: `912c2f5402195469f7fd4ca47842dbb817b065515355c24f5040425b2edf22eb`.
Report SHA256: `3bbd9ddbeb71209258cee1b6ad606a62abaefff56b1807f4976710ed3d31fa29`.
All134 input hashes and64 individual route hashes were rechecked after exit.
The new helper's two focused tests passed and its Ruff check passed.

The score-blind panel contains16 distinct cyclic representations, lengths8..40,
with four roots in each witness-CC1/2 x length<=20/>20 stratum. Every reference
path stays in RF TRAIN ownership. Its six invariant groups have multiplicities:
(3,2):6; (5,0):4; (9,4):2; (11,2):2; (13,0):1; (119,2):1.
These groups are not knot identities, and the16 rows are not independent knot
types. Q training exposure remains unknown; this is explicitly a diagnostic.

Both exact native checkpoints loaded without changing their tensors. The
native horizon was128, capacities48/12, and both L10/L1000 contexts were checked.
All64 combinations (16 roots x two native controllers x two objectives) routed
successfully. Their resulting semantic witnesses replayed to the empty one-braid.
RF's ordinary-plus-four-insertions proposal pool contained every reference action.

| Controller | Semantic witness moves | Native plies including head movement |
| --- | --- | --- |
| Q254 raster | 6..24 | 6..31 |
| Q304 strand graph | 6..24 | 11..47 |

This rules out absence of an accessible reference solution for these particular
roots under this128-ply envelope. It does NOT establish that either policy can
find it, that its witness is cost-optimal, that all useful actions are proposed,
or that long inputs transfer successfully. Routing explicitly supplied the
reference semantic actions; no policy scores determined the route.

Next: a prospectively specified native-policy/search comparison on these roots,
with actual neural-evaluation counters, preserved controller state, portable
witness replay, and clearly distinguished semantic/native clocks. Avoid an
unqualified equal-simulations comparison: Q's controller-level MCTS and RF's
tree-plus-policy-continuations are different search algorithms. Also declare
the crossing-change cap and treatment of PASS before scoring; changing only
RF's resource features can itself create an out-of-training-distribution test.
Keep this diagnostic separate from the independent protocol70/74 evaluation.

At completion of this gate the original experiment was still healthy: graph
4096/4096 completed; control1616/4096 and advancing; the two evaluators were
waiting for the final paired-training gates. No training source, checkpoint,
frozen evaluation protocol, or existing artifact was modified by this check.
