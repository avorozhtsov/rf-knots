# Full ordinary graph-pair preparation

2026-09-21, fixed before the full-source pass. Extends the successful
512-path feasibility/audit, not its sampling stream or scientific claims.

Use the same SHA-pinned source as protocol65. Enumerate every eligible
source-TRAIN single-CC edge in pair-ID order (expected62086). Then every
unordered pair of distinct neighbors of every middle representation, with
middles and neighbors sorted (expected28857 paths). No sampling, topping-up,
model scores or optimizer. Total expected90943 candidate paths.

For every visited representation: require a source-TRAIN ordinary Artin
UKB0 encoding, retain the exact word and strands, recompute signed signature
and determinant, cross-check determinant with independent Burau arithmetic,
and record invariant owner. Replay every literal crossing change, including
paths that will be excluded. Reject the whole path if any state is not
owner-TRAIN, quarantined, or in a historical query group from the15 panels
pinned by the probe's successful independent audit. Known TRAIN references
may be shared; do not convert source physical-state splits into identity claims.

Lower bound: max(signed-signature difference/2, distinct determinant).
Upper bound: one or two literal CCs. A special exact0 tightening is permitted
only when the two endpoint words are equal modulo cyclic rotation with the
same strand count; retain the canonical equality certificate and the original
replayed path's CC count separately. No free mirror, no guessed knot equality,
and no conversion of fingerprint(1,0) into an unknot certificate.

Deduplicate unordered endpoint pairs by signed cyclic-word canonical form,
not by fingerprint or mirror quotient. Keep all candidate-path provenance;
the representative uses the smallest proven upper bound, then earliest path
index. Lower bounds must agree between equivalent endpoint pairs. Report
counts before and after deduplication, exact/interval support, invariant groups,
raw L/N and both-endpoint group support. Do not call any count independent knots.

Output `artifacts/ordinary-graph-metric-corpus-v1`: a transactional SQLite
working corpus, source snapshots and an atomic manifest. Commit progress every
256 paths; metadata in the same transaction is authoritative for resume after
preemption. One writer lock, exact input/config validation, no duplicate workers.
Existing artifacts are never overwritten by a new run. On completion close the
DB, verify source pins, and save its SHA in the final manifest.

Completion means **prepared, not yet training-ready**. A separate source,
replay, partition, canonical-dedup and bound audit is required before training.
These general Gordian pairs are not full unknotting trajectories, optimal policy
actions, large-knot generalization evidence, or a million-rollout dataset.

Subsequent training will use the full shared-weight scalable network, retain
explicit policy-supervision provenance, and balance actual available metric
classes/groups. Its mixture and comparison protocol must be fixed separately,
after measured support and before optimizer updates and fresh TEST scoring.
