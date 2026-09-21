# Ordinary graph-local Gordian pair supply probe

2026-09-21. Fixed before sampling or computing new fingerprints; no optimizer
or model scores. This follows the source-semantic inspection in note 64.

Read the SHA-pinned embedding-pairs-v4 sidecar, source TRAIN endpoints only.
Only provenance `replayed_cc_upper_bound` or
`replayed_cc_distinct_attested_knots` with witness_semantic_length=1 can enter
the pool. Do not import graph-edge bounds or mirror-orbit zero relations.
Strictly reject UKB0 flags other than ordinary Artin zero; independently
reconstruct one sign flip at its exact physical position and replay it.

Seed20261030: choose 256 single edges uniformly without replacement from
pair-ID order. Then choose 256 distinct middle representation IDs uniformly
from the sorted degree>=2 list; for each choose two distinct neighbors.
This creates actual two-CC paths through graph representations, not claims
of exact distance two. Sampled paths may share endpoints or knot types.
No retry to obtain a desired label balance or fresh type count.

For all selected endpoints and midpoints recompute determinant and signed
signature; independently check determinant with the Burau implementation.
Endpoint signed-signature difference / 2 and distinct determinant provide
ordinary Gordian lower bounds. The literal replay supplies upper=1 or 2.
Every midpoint participates in the existing invariant-owner partition check.
Keep rejected-by-owner paths in the diagnostic report; do not emit TRAIN data.

Report class support and invariant groups separately for all paths and for
all-owner-TRAIN paths. The latter is **not** a completed historical identity
exclusion audit. No source split, knot name or fingerprint is promoted to an
identity certificate. This probe cannot establish policy usefulness, u(K),
large-size generalization, or sufficient exact-2 supply before execution.

Output `artifacts/ordinary-graph-pair-supply-v1/report.json`, with immutable
inputs, configuration, selected source IDs, exact replay positions, states,
fingerprints and durable progress every16 paths. One writer lock; resume only
matching source/config. Final input hashes checked. No duplicated worker,
checkpoint update, source database write, training-ready export or TEST score.
