# Prospective evaluation of the graph-mixture experiment

2026-09-21, fixed before training69. Evaluate initial joint57, final graph,
and final old-data control. Do not select checkpoints or alter data generation
from loss/TEST scores. Comparison is descriptive with one training seed.

## Fresh constructive short-u/search panel

Use `build_constructive_short_u_panel.py` with seed20261032,10000 attempts,
max_strands33, min_depth24, max_depth96. Use the existing first-eligible rule,
max4 per fingerprint group,32 roots per witness-CC class1/2. Exclude query
groups from all15 current historical panel.json files, plus full audited
TRAIN path groups from all metric/policy inputs in69. Keep partial/failed
historical panels in the exclusion set. Do not top up or change seed to
obtain desirable scores. Require at least32 roots,8 groups and8 examples per
witness-CC class; otherwise preserve the failed supply result and do not
claim a completed independent test.

Replay all witnesses, recompute signed-signature/determinant bounds, verify
query owner=TEST and nonoverlap with optimizer groups. Report exact1, exact2,
intervals and constant0/1/2 controls, equal weighting by query fingerprint.
Use initial raw size strata relative to combined metric TRAIN envelope
L<=254/N<=21; separately note policy-supervision envelope L<=208/N<=17.
Do not replace raw complexity by normalized length or zero-CC lift success.

Cost MCTS: same roots, λ10 and1000 separately,8 simulations, temperature1,
c_puct1.5, widening4, four insertion candidates. Seed20261034+batch start,
eight conditions per batch. CC cap=reference witness CC count. Let
q0=(L+N-1)/2; move cap=max(128,4*q0+32,reference moves), horizon=move cap,
growth allowance at least8 letters/4 strands and at least the reference
requirements. This is prospective additional headroom, not a change to58.
Report actual moves/evaluations/action scores, not just equal simulation count.
Evaluate all three learned arms; uniform is a fourth algorithmic control.
Use no value-head returns. Replay every episode and solution independently.

Report coverage, new/lost solutions, common-success L10/L1000 deltas and cost
relative to known references; no zero/infinite imputation for failures. This
still does not prove global minima or unlimited-size solution quality.

## Fresh ordinary graph distance panel

Before scoring, enumerate source-local one/two-CC paths from non-TRAIN source
splits of the same pinned graph sidecar. Reconstruct ordinary Artin steps,
never import mirror-quotient upper bounds. Deterministic permutation seed20261033;
orient the first eligible endpoint as query with invariant owner=TEST and absent
from all historical query/optimizer groups, reference not quarantined. Reject
any held-out query identity collision conservatively by invariant grouping.

Keep first32 pairs per exact0/1/2 or interval category, max4 per query group.
Exact0 requires an explicit cyclic-coordinate or replayed zero-CC certificate;
do not invent it if the supply lacks such pairs. Require at least32 pairs,
8 query groups and8 exact2 pairs for the aggregate test gate. Preserve category
shortfalls, do not replace them with easy examples or source upper bounds.
Audit the complete selection/source/bounds before any model evaluation.

Report this general d(K1,K2) test separately from u(K). Fingerprint(1,0) is
not proof of an unknot. Keep raw and independently replayed zero-CC size-lifted
views separate if lifts are generated; lifted views are not new knot types.
These graph and constructive families do not constitute universal coverage.

## Decision

No automatic promotion. Distance evidence must beat meaningful controls and
not conceal exact2 regression. Search evidence must include replayed costs,
coverage and resource limitations. Report negative/insufficient outcomes.
Arbitrary-size inference compatibility, learning useful invariants, and
solving genuinely larger knots remain separate requirements.
