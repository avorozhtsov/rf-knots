# Pure-braid short-distance test: supply and execution

2026-09-21. Execution record for frozen [protocol74](74-pure-braid-short-distance-test-protocol.md).
It supplements, and does not erase, the inadequate sidecar/neighborhood outcomes
in71/73. No model has been scored on this panel at the first recorded snapshot.

## Complete construction

All3,072 attempts completed. The immutable panel contains126 ordinary-distance
pairs in120 conservative query fingerprint groups:

| Certified distance interval | Pairs | Query strands above21 |
| --- | ---: | ---: |
| [0,0], cyclic coordinate certificates | 32 | 13 |
| [1,1] | 32 | 16 |
| [2,2] | 30 | 12 |
| [0,1] or[1,2] | 32 | 10 |

The last row consists of four[0,1] and28[1,2] intervals. These are not exact1/2
labels. All retained references have a determinant/signature certificate of
nontriviality. They are not assumed to be unknots and do not need full unknotting
witnesses to certify their local query/reference distance.

Exactly2,776 attempts had a query with other invariant ownership,58 collided with
the frozen prior-query exclusions,126 were accepted and112 exceeded a class quota.
The effort and seed were not extended to make a score or category look better.

Raw query/reference sizes are L41..209 and N3..33. There are51 pairs above the
metric TRAIN strand envelope N21 and **none** above its word-length envelopeL254.
Thus this panel can test strand-count extrapolation, not length extrapolation.
The earlier arbitrary-size shape checks and the separate long-word short-u panel
must not be conflated with this distribution.

Panel: `artifacts/pure-braid-short-distance-test-v1/panel.json`, SHA
`5fc6fc9c9e17f1ab3b8bf393b0a8f247dcbcbabccaf80869ba89159c812cf3e2`.

## Nontrivial topology versus inflated representation

The signed-signature lower bound on unknotting number, abs(signature)/2, ranges
from0 to28 for queries and0 to29 for references. It exceeds2 for107 query slots
and107 reference slots. In the audited current optimizer group union the largest
absolute signature is14 (hence the largest such lower bound is7);76 query slots
and81 reference slots in this panel exceed that bound7.

These are **certified lower bounds**, not actual unknotting numbers, and slots
are not independent knot types. A TRAIN knot with a smaller signature may still
be difficult or have a large unknotting number. Nevertheless, the high-signature
members are not merely long zero-CC presentations of knots with u<=2. Their
nearby pair distances remain only0..2 by the explicit certificates. This is the
intended distinction between general short Gordian distance and distance to the
unknot. Exact0 rows are still only coordinate-invariance checks.

## Audit and actual execution

The independent auditor was actually started as
`audit_pure_braid_distance_panel.py` (PID91075 at the recorded snapshot). It
independently reconstructs every RNG draw and pure block, checks actual closure
permutations, recomputes all signed invariants and independent determinants,
replays literal CC paths/cyclic certificates, and repeats the complete first-
eligible selection without calling the producer's selection function. It must
finish and publish a passed certificate before model scoring. A live process or
adequate producer counts alone do not pass that gate.

The three-model evaluation controller was also actually launched:
`artifacts/pure-braid-short-distance-evaluation-v1`, PID92699 at the snapshot.
It waits on the exact live data-auditor command and the live base evaluation
controller until the final training-pair certificate appears. It does not start
or restart training, data construction, or auditors. Missing dependencies,
duplicates, inadequate data or altered provenance stop the transition for
explicit inspection. Source snapshots and hashes are preserved.

After both certificates pass, the controller serially scores initial joint57,
final graph69 and final old-data control69. The evaluator refuses partial weights,
different checkpoints or TEST-fitted calibration. It reports constant0/1/2 and
equal-query-group error, exact0 separately from exact1/exact2, positive-distance
lower bounds separately, and every raw-size stratum (empty strata stay empty).

The final comparison verifies the raw predictions and their summaries, compares
the same query/label order in all arms, and reports any group overlap with the
simultaneously constructed short-u panel. It does not pool the two panels as
independent replications or declare an automatic promotion. Search quality still
requires the separately executed full-policy MCTS/witness audits from70.

The new construction/audit/evaluation/sequence tests cover pure permutations,
all31 strand counts, zero-coordinate versus CC transcripts, corruption of source,
positions, signatures and bounds, query selection/deduplication, nontrivial
references, class/group caps, finite prediction summaries, honest constant
controls, exact command liveness, fixed final models and missing/failed gates.
The focused suite for these four components passed28 tests at this snapshot.

## Independent audit completed

The auditor subsequently completed all3,072 attempts and published `status=passed`,
`evaluation_ready=true`,126 selected pairs and empty query intersections with the
audited optimizer and historical/diagnostic exclusions. Its1,070 pinned inputs
were re-hashed after completion and all matched.

Certificate: `artifacts/pure-braid-short-distance-test-v1/independence-audit.json`,
SHA `50b47f2ba4eafb845b28d46ffe9b2743fd576b6e8e7ef2a5e840b558a724d917`.

The live three-model controller recognized the certificate and removed the
data-audit dependency from its waiting list. It is now waiting only for the
final training-pair audit in the base queue; it has not scored partial models.
The combined focused suite including the prior training/search guards passed87
tests (the only warning was unavailable optional Plink tkinter GUI support).

These are construction/integrity results, **not model accuracy results**.
