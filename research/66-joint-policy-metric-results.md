# Joint policy/metric: completed experiment and next training decision

2026-09-21. Final descriptive results of protocols57/58/60, including all four
search arms and independent replay audits. No production promotion or GPU
launch. The goal of useful distance embeddings and arbitrary-size knot
search remains open.

## Provenance

Controller `artifacts/mcts-policy-metric-evaluation-schema-v2/manifest.json`
is completed,15/15 jobs; SHA256
`b349db17906ac74f77b7ac593f5724aee58b30c7c38116a17c1fdcdd2840e180`.
The original schema-failure lineage remains preserved and hash-linked; see
execution note59. No failed outputs were overwritten or silently reused.

Final comparison: `artifacts/mcts-policy-metric-summary-v1/report.json`, SHA256
`f336ee84f8baa41c4af7a7ea85240a9a5a5004001fb0a72c6a4f2ebea5120c06`;
human-readable `report.md`, SHA256
`7defb471e2f770c15ceb2cce30b28542803e6be1889e9b0629750cd2273b14ba`.
It binds all final models, audits, query roots, resource limits, metric
outputs, resource-envelope diagnostic, semantic-cost certificates and recovery
receipt. Re-run verification only, not another worker against the same output.

## Distance quality: improved over parent, not yet good enough

Equal-fingerprint-group mean absolute interval violation, lower is better:

| Model | Short-u,64 groups | General Gordian,83 groups |
|---|---:|---:|
| Parent | 0.341766 | 0.530641 |
| Joint policy+metric | 0.291944 | 0.475879 |
| Metric-only control | 0.295382 | 0.504659 |
| Constant distance1 | **0.125000** | **0.289157** |

The constant is a sanity check on these distributions, not a useful universal
embedding or evidence of no ranking signal. All three learned models lose
to it on the aggregate metric. Joint's short-u exact2 MAE is0.131779 versus
parent0.101137 (8 rows): this class worsened. General exact2 remains poor:
joint MAE1.566747, control1.550040, parent1.616816 (only4 rows).
Do not hide these deficiencies behind the improvement in the aggregate.

The lifted views have numerically identical summary scores because the
normalization removes those particular reversible expansions. They do not
demonstrate new knot complexity or extrapolation beyond that normalization.

## Actual search: one query, not broad success

Each model searched64 roots at each CC weight,8 simulations per condition,
with full policy continuations and matching resource caps. All reported
witnesses independently replayed; cost=Lλ=λ*CC+semantic_moves.

| Arm | Solved at λ10 | Solved at λ1000 |
|---|---:|---:|
| Parent | 0/64 | 0/64 |
| Joint | 1/64 | 1/64 |
| Metric-control | 0/64 | 1/64 |
| Uniform | 0/64 | 0/64 |

All successes concern **the same query53**, L70/N3, inside the TRAIN size
envelope. Joint has L10=49 on its λ10 condition; at λ1000, joint has1045
and control1055. The latter is a real ten-move improvement on one common
success, not a mean over64 solved knots. Against parent/uniform there is
no common-success cost difference to estimate.

The reference for this query is certified optimal under the exact semantic
action system: L10=47, L1000=1037. Joint therefore remains +2/+8 above that
optimum, control +18 on its λ1000 result. Neither beats the reference.

The audited bounds certify40/64 reference optima, but not the other24.
For the latter, paths with fewer CC could legitimately use more semantic
moves than the current cap. The current protocol is resource-constrained;
it neither finds global minima generally nor gives matched relative budget
headroom across size strata. Every unsuccessful continuation ended at the
move cap. This describes the search outcome, not absolute bad-action labels
or proof of unsolvability.

Equal simulations are not equal compute. Final JSON retains neural states,
batches, action scores and times. Concurrent engineering diagnostics mean
wall times must not be sold as isolated speed benchmarks.

## Decision and next concrete work

Preserve the joint model as a completed experimental checkpoint, not as a
validated general replacement. There is a small observed search benefit and
some distance improvement, but insufficient coverage, weak exact2 calibration
and no demonstrated large-size solution quality. This is **not** grounds to
discard embeddings after this small run.

The next metric-data step now has actual evidence: notes64/65 reconstructed
ordinary one/two-CC paths from the proof-graph sidecar without treating mirror
as free. The512-path probe and independent partition/replay audit yielded283
eligible traces in34 invariant groups, including13 exact2 pairs. These are
small graphs/words, not the arbitrary-size test and not full unknotting paths.

Proceed to deterministic full-source preparation of the62086 local CC edges
and28857 two-edge paths, with checkpointed work, literal replay, recomputed
bounds, all-state owner filtering, historical query exclusions, and canonical
pair/group deduplication. Publish actual retained support before deciding a
new balanced metric mixture. Keep uncertain intervals as intervals and keep
general distance distinct from u(K): fingerprint(1,0) alone is no unknot proof.

After that audit, jointly train the same full scalable network, with a
matched data/control design and a prospectively fixed fresh evaluation.
Current successful-policy supervision can remain an explicit source; do not
turn arbitrary graph-local CC edges into optimal unknotting policy labels.
A separate next search protocol should specify size-relative resource
budgets prospectively and test L-only, N-only and joint growth on genuinely
new query families. Do not retune current TEST or call lifts independent
large knots. No head-only residual adapter is implied by these steps.

Further scaling to a million evaluations/trajectories needs end-to-end
throughput and data diversity, not only repeated draws from these same
examples. Million-scale RL, general large-knot solving, and useful calibrated
distance prediction remain requirements to work toward, not completed claims.
