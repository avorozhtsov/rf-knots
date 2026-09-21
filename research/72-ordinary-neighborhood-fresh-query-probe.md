# Prospective ordinary-neighborhood supply probe

2026-09-21. Fixed before this probe runs; no checkpoint scores are used.

The source-local held-out sidecar paths in70/71 supplied zero fresh invariant-TEST
query groups. Preserve that inadequate result. This is a **separate diagnostic
construction**, not a top-up or a substitute success result for70. It does not
change training69 or its short-u/search comparison.

Take the unique physical source representations from the fully audited 1,635
non-TRAIN source paths in `graph-mixture-fresh-gordian-test-v1/progress.json`.
All are ordinary Artin words. Choose uniformly over source representation IDs
whose word has at least two letters. Source split alone is not independence:
references may belong to TRAIN groups and be known, but fresh queries may not.

Run exactly1,024 attempts, seed20261035. For attempt i use an independent
`random.Random(seed+i)`; choose one source representation, then sample without
replacement k=1+(i mod2) positions in its word. Flip those signed letters in
sample order. Reverse the resulting path to orient the new endpoint as query
and the original source state as reference. Do not mirror, normalize, change
strand count, use quotient labels, or choose flips by model score/invariant gain.

Reconstruct every literal CC step, recompute every signed signature/determinant,
and check each determinant independently. The lower bound is the maximum of
the signed-signature difference divided by2 and the determinant-inequality
indicator. The upper bound is the replayed kCC steps, except signed cyclic
coordinate equality can separately certify zero. Record the actual k even then.

Count a query as eligible only when its invariant owner is TEST, its conservative
(det,abs(signature)) group is absent from all15 historical query panels and the
complete audited protocol69 optimizer lineage, and its reference is not
quarantined. Intermediate states need not be TEST: this is evaluation supply,
not a TRAIN corpus. Report all attempts and rejection reasons, unique eligible
query groups, physical signed-cyclic endpoint pair deduplication, and interval
support; do not equate records/groups to independent knot types.

This probe is not a model test, does not have a promotion gate, and must not be
scored as an independent accuracy result. Audit its source membership,
deterministic draws, physical replay, bounds, and ownership separately. If the
supply is promising, a later declared test contract must freeze its own seed,
budget and selection and exclude these diagnostic query groups as well. If it
fails, keep the failed supply outcome without retuning this seed/budget.
