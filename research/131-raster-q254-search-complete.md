# Protocol97: Q254-final search complete and fully audited

2026-09-22. The first of the TWO matched final-model searches is complete.
This is not a completed paired comparison or an embedding promotion decision.

## What ran

Scalable shared raster-tile/strand-graph policy and global embeddings, trained
under97 for2048 updates. Q254 supplied the local raster trunk, not its original
policy/value heads; upper layers were newly initialized and jointly trained.
This was the frozen Q254 lineage, not a claim to the best current Q-Run model.

All64 frozen70 short-u roots, each at lambda10 and1000:128 conditions. Eight
cost-tree simulations, learned policy continuations and embedding-derived tree
priors, no scalar value head, no oracle incumbent. CPU float32, one thread,
microbatch2, unchanged seeds/resources and no neural cache. Both final arms have
embeddings; this experiment does not isolate their causal contribution.

## Evidence and verification

Root: `artifacts/pretrained-raster-log-evaluation-v1/mcts/q254/`.

| Artifact | SHA256 |
| --- | --- |
| Final trained Q254 checkpoint | `c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f` |
| Final search manifest | `a88c3f09f54bdee789670df0cdd37eed3d2c45b139955b45b87d3803227d88a6` |
| Full replay/cost audit | `9ccc083ea71f2ae7378d03beea5f4471b8545ac16222ebd41b0ad84416e0291f` |
| Exact common128-root contract | `c1da996e1e6e95251986dd04bdad1467d45ad3f07055fc2b6ca79424131d7967` |

The controller recorded the search and its full audit as completed. The audit
replayed every successful and censored path, all64 reference witnesses and all20
solutions; checked native resource/cost identities; and independently recomputed
integer determinants. Signature lower bounds come from the audited frozen panel.
All1287 audit input pins were subsequently rechecked. A separate arithmetic pass
checked exact query coverage, solution identities, reference lower floors,
regret semantics and all per-cost totals directly from the immutable JSON records,
without launching a duplicate trajectory replay.

## Search outcomes

| Measure | L10 search | L1000 search |
| --- | ---: | ---: |
| Conditions | 64 | 64 |
| Successful | 9 | 11 |
| Not solved within the frozen budget | 55 | 53 |
| Found cost proven optimal | 2 | 2 |
| Better than reference witness cost | 0 | 0 |
| Equal to reference cost | 2 | 2 |
| Worse than reference cost | 7 | 9 |

Twenty successful conditions cover16 distinct raw representations; four roots
succeeded under both cost weights. These are not20 distinct knots or independent
games. A witness can be evaluated under both costs without creating a second
successful SEARCH result.

Certified regret histograms among successes:

- L10:0 moves x2,2 x5,8 x1,82 x1.
- L1000:0 moves x2,2 x7,32 x1; one solution has unknown optimality regret.

The unknown case is root19: u is bounded by[1,2], found2CC/125 moves versus a
2CC/121-move reference. Gap to reference4 is NOT proven regret4. The reference
itself is not certified optimal. No completed search improves an existing
reference bound, so this result does not justify a proof-database import/release.

## Four optimal-cost search results

| Root | Input length / strands | Search objective | Found cost | CC / moves |
| --- | ---: | --- | ---: | ---: |
| 30 | 392 / 25 | L1000 | 2210 | 2 / 210 |
| 32 | 64 / 5 | L10 | 45 | 1 / 35 |
| 48 | 194 / 17 | L10 | 116 | 1 / 106 |
| 50 | 136 / 5 | L1000 | 1071 | 1 / 71 |

For this native semantic move system, q0=(length+strands-1)/2 and every complete
witness satisfies moves=q0+CC+excess, with excess>=0. Each listed witness has
zero excess and reaches the certified CC lower bound; it therefore attains
the cost floor q0+(lambda+1)*lowerCC. Optimality concerns this representation
and cost/action contract, not every possible presentation or move convention.

Root30 and the successfully solved407-letter/28-strand root31 exceed BOTH actual
protocol97 training maxima254 letters/21 strands. Positive policy examples
reached208/18 and all complete relative groups reached at most128/17. This is
bounded size extrapolation with replayed solutions. Root31 costs remain above
their certified optima, by82 moves at L10 and32 at L1000. Neither result proves
arbitrary-size reliability, unseen knot identity relative to all Q pretraining,
or that embeddings caused the success. See120 and the hash-verified121 census.

The final two successful queries, previously awaiting full audit, now passed:
root60 (189 letters/6 strands), lambda10,1CC/100 moves,L10=110, regret2;
root62 (193/22), lambda1000,1CC/110 moves,L1000=1110, regret2. Their excess2
comes respectively from INSERT and negative STABILIZE, not a new CC improvement.

## Compute and censoring limits

Recorded746 episodes,351448 policy continuation-state evaluations and352103
full-path moves;55953078 action scores. Policy neural batches181676; tree
embedding evaluations3072/batches1584, tree action scores5589. Committed-batch
time85743.710965s (about23.82 hours). These are recorded work/timing counters,
not unique representations, new training games or independent neural reruns.

Reference costs are certified optimal for40/64 roots. At L1000,24 roots have a
necessary move envelope for beating/matching reference that exceeds the frozen
search move cap. This does not prove a cheaper path exists, but limits what a
budget failure says. Unsuccessful searches remain censored observations, not
proof of impossibility or negative labels for all actions on their trajectories.

## Decision and remaining work

Keep the protocol unchanged. The random-initialized model was ALSO trained for
2048 matched updates; its final checkpoint is
`17433748fac7308cf4c36ef0c40650c7ec83d58dee8625834ce531c2f5bb40c6`.
Its128-condition search started automatically after this audit, using the exact
same root contract. At19:41–19:42 UTC the unique worker2020 under controller98832
had advancing CPU and no committed first batch;11/13 controller jobs completed.

Wait for that search, its complete replay/cost audit and the paired summary.
Distance outcomes109 were weak; these search successes do not reverse that
finding or establish training/embedding benefit. There is no initial-model MCTS
or no-embedding arm in this matrix, and70/74 are shared TEST panels, not fresh
replications. Do not retune on them or import their trajectories into training.
No new training, source modification, checkpoint promotion, GPU rental or release.
