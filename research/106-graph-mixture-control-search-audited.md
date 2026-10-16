# Protocol 70: trained-control search completed and replay-audited

2026-09-21. This records the completed `control` arm of protocols 69/70,
not the final 69/70/74 comparison. Protocol 97 remains frozen: none of its
training or evaluation settings is changed using these TEST results.

## Evidence and scope

Directory: `artifacts/graph-mixture-evaluation-v1/mcts/control/`.

- Final manifest SHA256:
  `548ed4ebdacd51c9f1b81dddfce0c37873fff100b4bbaea3d3eb45a775fa36c3`.
- Passed replay-audit SHA256:
  `536a0f3583159aeaf137e6a5d3a71f808d2f944fc63b5c90fe42c00bd9a5ef13`.
- All 128 conditions and 764 episodes replayed: 375,336 continuation moves,
  375,980 full-path moves, and 65,821,249 recorded action scores.
- Both committed jobs were independently checked with
  `run_graph_mixture_evaluation.verify_job`; all 16 shard hashes and ordered
  condition coverage were checked with `committed_rows`.
- References and successful witnesses were independently reconstructed with
  `summarize_graph_mixture_experiment.certify_costs` for this complete arm.
  The incomplete-matrix gate was not bypassed; no final summary was emitted.
- Recorded committed search time: 5,794.759 seconds. The full replay audit
  checks resources, RNG context and costs, not independently rerun timings
  or tree-work counters.

This is the 4096-update trained control descended from the RF protocol-57
initial checkpoint, not the protocol-97 Q254-local raster model. Search uses
no learned value head. Its checkpoint SHA256 is
`e72b159d7ddb3db27b1cc491cde23203b91cfb6ec7f66749f1e25abc52607689`.

## Confirmed outcomes

| Objective | Initial (report 104) | Graph mixture (report 105) | Trained control |
| --- | ---: | ---: | ---: |
| L10 | 3 / 64 | 2 / 64 | 3 / 64 |
| L1000 | 2 / 64 | 1 / 64 | 2 / 64 |

All three arms share the fixed panel, search settings and seeds. Equal initial
and control counts do not mean identical behavior: their successful conditions
are disjoint. All three graph-mixture successes are also control successes;
control additionally solves panel 1 at L1000 and panel 50 at L10. The sample
is small, and the required uniform search and replay audit are still pending.

For each control success, the corresponding reference attains the certified
semantic cost floor. Therefore the last column below is certified regret
within the declared action system, not merely a gap to an arbitrary witness.
Here `L_lambda = moves + lambda * CC`.

| Panel index | Objective | CC | Moves | Cost | Certified optimum | Excess cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | L1000 | 1 | 100 | 1100 | 1098 | 2 |
| 3 | L10 | 1 | 283 | 293 | 101 | 192 |
| 32 | L1000 | 1 | 41 | 1041 | 1035 | 6 |
| 50 | L10 | 1 | 296 | 306 | 81 | 225 |
| 55 | L10 | 1 | 391 | 401 | 117 | 284 |

All five solutions use one crossing change; none reaches the certified cost
optimum. Excess costs decompose as 2, `2*96`, `2*2 + 2`,
`2*91 + 37 + 6`, and `2*142`, respectively, from counted insertions,
stabilizations, commutations and braid moves. These are action counts, not
chronological suffixes or a causal explanation of failed searches.

On the three shared successful conditions, graph-mixture costs are better
by 10 and 4 for panels 3 and 32, but worse by 36 for panel 55. Thus the
control's higher success count is not a uniform cost improvement either.
Failures remain budget-censored; they are not proof of unsolvability or
negative labels for every action on a failed trajectory.

## Continuation and limits

The controller accepted both control artifacts (12 of 14 prescribed jobs
completed) and automatically started the `uniform` search. Its full 128
conditions and replay audit must finish before the combined summary.

These results, together with report 103, do not justify promotion of the
graph-mixture model. They do not establish that embeddings are useless,
and do not evaluate adding an embedding channel to an intact best Q-Run
solver. No database discovery or production replacement is claimed.

Separately, protocol-97 Q254-local training is complete; its matched random
local initialization is still training. Its paired audit and independent
distance/search evaluation remain pending. The current results do not
replace those evaluations or justify changing their frozen protocol.
