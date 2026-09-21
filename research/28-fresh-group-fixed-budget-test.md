# Fresh fingerprint groups: fixed-budget evaluation construction

2026-09-20, declared while the two clean multisize trainings are still running,
before this new generator is run and before any model scores on its outputs.
This does not amend or complete the previous partial 31/32 panel.

The eligible population is a fixed stream of 40000 replayed unknot-origin
walks, seed 20261008. Alternate one and two CC; isotopy walk length 24..48,
settling 12, target length 32..64, target strands 3..9, caps 80/10,
strand band one. Retain the first four distinct circular representations of
each eligible fingerprint. Exhaust the entire stream: no acceptance quota,
early success stop, seed search or model-based filtering.

Eligibility: owner=test under the existing immutable split, and
`(det, abs(signature))` absent from all five pre-existing panel files:

- embedding-short-u-independent-test-v1/panel.json
- embedding-multisize-clean-test-v1/panel.json
- embedding-catalogue-independent-test-v1/panel.json
- embedding-short-u-heldout-panel-v1/panel.json
- embedding-external-independent-test-v1/panel.json

Paths above are under artifacts/. The builder pins and copies these exclusion
inputs and records every excluded group, source hashes, selection counters,
accepted provenance and source code. Fail on unexpected generation/replay
errors; do not silently skip corruption. Resume must use identical source,
configuration and exclusions. Train-path ownership is separately verified by
the frozen multisize corpus builder and both trainers; new roots in owner=test
cannot be isotopic to train roots or intermediate train states if those
invariant/split computations are correct. Fingerprints are not unique knot IDs.
This is fresh relative to these five panels, not an exhaustive audit of every
historical legacy experiment in the project.

No target count is promised for a fixed-effort sample. Construction completion
means all 40000 attempts are processed. Scientific adequacy is separate:
require at least eight eligible fingerprint groups and sixteen representations
before scoring. Otherwise retain the exhausted result as insufficient and do
not weaken this gate. No network evaluation enters construction.

Keep original signed-signature lower bounds and exactly replayed CC upper
bounds. Also produce zero-CC lifts to at least L256,N33, beyond this run's
training dimensions, as a separate stability view, not independent knot types
or intrinsically hard large knots. A broad interval is not an exact label.

If adequate, use the final step-4096 clean joint and policy checkpoints and
their identical SHA-pinned initial state. Apply note 27's fixed distance and
search measurements to this separate cohort, preserving group-wise results,
censoring, returned-witness replay, resource caps and compute accounting.
Report original and lifted distance errors; search every original query at
128 expansions for both L10/L1000, then one model-blind shortest reference
witness per group for four-seed sparse MCTS at 512 simulations. No checkpoint,
hyperparameter or root selection based on these results. This measures u and
search; general non-unknot pair distances remain a separate evaluation need.
