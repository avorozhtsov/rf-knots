# Clean paired iteration: completed cost-search stage

2026-09-20. This note records completed whole-arm comparisons as the frozen
note33/35 evaluation proceeds. It is NOT the final42-job report or a promotion.
All four cost-search arms are now complete. MCTS conditions and the rollout
portfolio still remain. Their prescribed budgets and source files are unchanged
by these observations. The controller-owned paired-comparison jobs also remain
queued; the descriptive audits below did not overwrite those planned outputs.

## Completed parent versus value-supervised arm

Both reports contain exactly64 original-panel roots at each CC weight10/1000,
with128 expansions per condition and matching witness-feasible resource caps.
All solutions were independently replayed with the comparison loader; initial
states, CC budgets, move/length/strand limits and both recorded costs were
checked. Failures retain null witnesses/costs rather than invented penalties.
Full query coverage, identical evaluation configurations/code hashes and each
report's pinned checkpoint SHA were checked independently of controller status.

The comparison below was computed read-only from the completed reports. It did
not create or overwrite the comparison outputs owned by the running controller;
the controller will still execute its prescribed later comparison jobs.

| CC weight | Parent solved | Value solved | Newly solved | Lost solutions | Common solved, same cost | Both unsolved |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 34/64 | 33/64 | 1 | 2 | 32 | 29 |
| 1000 | 34/64 | 33/64 | 1 | 2 | 32 | 29 |

There are no lower-cost or higher-cost outcomes among the32 common solved roots;
their mean cost delta is exactly0 in both conditions. Repeated cost conditions
are correlated, not128 independent knot tests. The finite-budget losses of
solutions do not establish that those roots are unsolvable.

### Changed roots

These labels are panel indices, not catalogue knot identities.

| Panel root | Parent L10 | Value L10 | Parent L1000 | Value L1000 |
|---|---:|---:|---:|---:|
| short-u-1 | 75 | unsolved | 2055 | unsolved |
| short-u-17 | unsolved | 88 | unsolved | 2068 |
| short-u-40 | 52 | unsolved | 2032 | unsolved |

### Actual compute

| Model | CC weight | Neural-state evaluations | Recorded search seconds |
|---|---:|---:|---:|
| Parent | 10 | 36179 | 303.052722 |
| Parent | 1000 | 36179 | 300.139072 |
| Value | 10 | 37466 | 298.626125 |
| Value | 1000 | 37466 | 297.446871 |

Equal expansion budgets are not equal neural work. The value arm used1287 more
neural-state evaluations per condition. Small wall-time differences on a shared
host are not evidence of a speed improvement. These are anytime bounded-search
results, not optimal-cost certificates.

## Provenance

All paths below are inside `artifacts/embedding-clean-paired-evaluation-v1/`.

- `cost/parent/report.json` SHA:
  `e4ef270e74475fc56213b02e812672cd834ee71e9aed6ae754c2ee08622f5ffa`.
- `cost/value/report.json` SHA:
  `371da47b5f2e9cd371bf870aab46b81e8c2867ea65f885614e9b4afad90aaea7`.

Both hashes match the controller's completed-job records. The controller moved
to `cost-control` automatically (observed live worker26062, durable rows0 ->12),
with11/42 jobs completed and no error. Recheck live processes/files rather than
treating these recorded PIDs or counts as permanently current.

## Interpretation so far

This completed comparison gives no improvement in solved coverage or common
L10/L1000 costs from the value-supervised iteration. It changes weights and
training objectives together relative to Parent; the matched Control results
below address the effect of adding value supervision separately from the
relative-policy/metric updates. Search here uses the embedding heuristic, not
the learned MCTS leaf-value mode. No conclusion about that MCTS channel can be
drawn before its own fixed evaluation finishes. This bounded result also does
not prove that global embeddings are useless after substantially larger joint
training.

## Completed four-arm comparison

Control and Length subsequently completed all128 conditions each. The same
independent loader replayed their witnesses and checked exact64-root/two-cost
coverage, hashes, caps and failure censoring. All arms share the same evaluation
code and search configuration apart from the declared learned/length mode.

| Heuristic | Solved at CC weight10 | Solved at CC weight1000 |
|---|---:|---:|
| Parent embedding | 34/64 | 34/64 |
| Value-supervised embedding | 33/64 | 33/64 |
| Control embedding | 34/64 | 34/64 |
| Word length, no neural heuristic | 1/64 | 1/64 |

The two cost conditions give the same following paired outcome counts; they
remain correlated observations, not a doubled independent sample.

| Candidate vs baseline | New solves | Lost solves | Common equal cost | Common lower cost | Both unsolved |
|---|---:|---:|---:|---:|---:|
| Control vs Parent | 1 | 1 | 33 | 0 | 29 |
| Value vs Control | 0 | 1 | 33 | 0 | 30 |
| Value vs Length | 33 | 1 | 0 | 0 | 30 |
| Parent vs Length (post-hoc descriptive) | 33 | 0 | 0 | 1 | 30 |
| Control vs Length (post-hoc descriptive) | 34 | 1 | 0 | 0 | 29 |

There are no higher-common-cost cases in these rows. Control vs Parent and
Value vs Control have mean common-solved cost delta0. Parent vs Length improves
the only common solution by4 semantic moves at each CC weight. Value and Control
lose the one Length solution, so their common-solved cost deltas versus Length
are UNAVAILABLE, not zero. The Parent/Length and Control/Length paired summaries
are additional post-hoc descriptions, not predeclared hypothesis tests or extra
independent evidence; no model is selected using them.

### Remaining compute and provenance

| Model | CC weight | Neural-state evaluations | Recorded search seconds |
|---|---:|---:|---:|
| Control | 10 | 36235 | 292.724740 |
| Control | 1000 | 36235 | 290.701981 |
| Length | 10 | 0 | 44.958607 |
| Length | 1000 | 0 | 45.676543 |

- `cost/control/report.json` SHA:
  `22845cd7bf40a706f98e9717fa1a2b357f4ddecbfa68b6cf8f9045bd62bbbb5c`.
- `cost/length/report.json` SHA:
  `14b24a7a4e7622d80bb7e6ea9027b8d1e8b4853e57a4d218644567f024d69f84`.

The learned-heuristic searches have much better solved coverage than the weak
length heuristic at the same128-expansion allowance, but take substantially
more neural work and wall time. This demonstrates utility of these embedding
heuristics relative to THAT baseline on this panel. It does not isolate the
benefit of learned weights versus architectural/normalization bias: a random
initial-weight search was not part of this frozen panel protocol. It is not
proof of useful exact-u calibration, superiority to all non-neural heuristics,
or generalization to arbitrarily large nontrivial knots.

The incremental value-supervised iteration still gives no cost-search gain over
Parent or matched Control. Its learned MCTS leaf value is a different use of the
network and remains to be evaluated. Controller99732 reached13/42 completed
jobs and launched the first MCTS condition (observed worker29311): Parent,
constant0.5 leaf, CC weight10,55 groups, two seeds and512 simulations. The live
log advanced41 ->82 ->124 simulations per tree, with0/110 solves at that partial
point. This is not a completed MCTS result; keep the full sequence running.
