# Train-only constructive short-u supply probe

2026-09-20. Fixed before generating this attempt stream. This follows the
completed negative clean-paired iteration, not a retest of its held-out panels.
No model scores, optimizer, GPU launch or search-budget tuning are involved.

Run1024 attempts, seed20261017, existing `conjugate_witness`, max_strands17,
min_depth8, max_depth48. The deterministic constructor cycles one/two CC and
strand counts3..17. Exhaust the stream; do not replace the seed or top up a
desired category. No artificial size lifts. Preserve rejected-reason counts.

Read only the pinned existing metric TRAIN corpus to identify already-supported
endpoint fingerprint groups. Exclude these groups at the candidate root; this
means new relative to TRAIN endpoints, not globally unseen knot types. All
candidate CC endpoints must have the existing train partition owner. Exclude
determinant-one roots for this particular short nontrivial-u probe. Record at
most four roots per fingerprint group, with distinct normalized root states.
These are dataset design choices, not unbiased sampling of knot space.

Every accepted candidate has a fully replayed witness. Store the original
signature lower bound and, separately, the bound strengthened by nonunit
determinant. The witness CC count remains an upper bound. Classify exact1,
exact2 and intervals honestly; never relabel a two-CC witness as exact u=2.
This probe does not emit a training-ready corpus. Independent determinant,
lineage exclusion and record-format audits remain gates before training.

Measure accepted records, fingerprint groups, normalized states, original and
normalized lengths/strand counts, and exact/interval class support. Repeated
views and fingerprint groups are not distinct knot identities. No acceptance
gate is chosen after looking at results: this is a feasibility measurement,
whose complete output informs a separate, prospective training protocol.

Output `artifacts/embedding-diverse-short-u-supply-v1`. One writer lock; immutable
eight-attempt shards; source/input SHA pins; durable next-index progress. Resume
only matching inputs and committed shards. No overwrites, training restarts,
deletion, publication or reading of evaluation scores.
