# Research publication checkpoint — 2026-09-21

This publishes the scalable embedding/policy architecture, research protocols,
training/evaluation controllers and tests as a reproducible research snapshot.
It is not a production model promotion or a claim of million-trajectory RL.
Generated databases, checkpoints, logs and source snapshots remain local under
the ignored `artifacts/` directory. No active training sources were reformatted.

## Validation and scientific status

- Full suite: **1179 passed in 281.75 seconds** (`.venv/bin/python -m pytest -q`).
- `git diff --check`: passed before staging.
- Full Ruff check: **520 findings**, not a clean lint gate: 269 E501, 117 E702,
  98 E701, 29 I001, 3 B905, 2 F401, 1 E731 and 1 F841. The three F findings
  are unused imports/variable in older factorized-proposal research scripts.
  Do not imply these are all pre-existing on committed main. Formatting and
  cleanup are deferred because active experiments pin source contents.
- Protocol69 graph arm completed 4096 optimizer updates; its matched control
  is still training. Final paired quality evaluation is pending. Two data
  audit jobs in protocol70 are complete; no partial checkpoint was promoted.
- The completed metric-only experiment improved greedy-frontier coverage
  from 34/64 to 57/64, not MCTS coverage; costs on common learned-model
  successes did not improve. See note47.
- The completed joint experiment solved only one of 64 distinct MCTS query
  roots; constant distance1 still beats aggregate distance calibration.
  See note66. Shape scaling is not demonstrated large-knot solving.

## What 4096 training steps mean

Each step is one AdamW update from 128 sampled metric pairs, 32 sampled
positive-policy states and one available relative-success comparison group.
No new MCTS collection occurs inside that optimizer loop. Across one arm this
is 524288 pair presentations and 131072 policy-state presentations, with
replacement, not unique knots or search evaluations.

The graph arm uses 31 old pairs, 32 suffix-to-unknot pairs, 64 new graph pairs
and one fixed exact-distance2 chiral sentinel. The old sampler is uniform
over eight (task, bound-class) bins; the suffix sampler over four bound bins;
the graph sampler over exact1, exact2 and interval. Within a bin it samples
an invariant group uniformly, then a record. Thus expected batch counts are:

| Label | Graph arm | Control arm |
| --- | ---: | ---: |
| Exact0 | 15.75 | 31.75 |
| Exact1 | 37.083333 | 31.75 |
| Exact2 | 38.083333 | 32.75 |
| Interval only | 37.083333 | 31.75 |

These are sampling expectations, not measured class totals for the saved run
or fixed quotas in each batch. Interval supervision does not assert an exact
distance. The repeated sentinel is one pair, not additional independent data.

## Publication scope

The user authorized commit + push and preparation of a new database release.
All three repositories were on main: no merge is necessary. RF Knots had
35 commits ahead of origin/main plus this working-tree research package;
those commits were not all produced in the last 28 hours. UnknotDB has a
separate source/preflight publication; PGX has no new local changes.
This authorization does not turn research results into validated database
proofs. Drive release replacement is not performed by this checkpoint.
