# Prospective short Gordian-pair evaluation

Declared 2026-09-20 before generation or model scoring. This complements, not
replaces, the short-u/search evaluation. Use the two step-4096 clean multisize
models and their pinned initial state, no checkpoint selection.

Generate a fixed stream of 4000 seeded positive-braid walks, seed 20261009.
Index i uses CC=i%3, walk length 8, settling 6, target strands
3+(i//3)%3, target length 18+(i//9)%8, caps40/7 and strand band one.
The positive-braid seed has strands2+(i//27)%4, genus1+(i//108)%6,
seed20261009+i and churn4. Mirror both endpoints/witness when (i//2)%2.
The pair ends at the seeded knot; its replayed suffix to the unknot is retained
for auditing, not used to label the pair distance.

Retain only owner=test queries absent from the five old panels listed in
note28. Reference endpoints may be train, validation or test, but never
quarantine. Allowing test references is necessary for zero-distance pairs:
isotopic endpoints necessarily have the same split. This permission is confined
to a separate evaluation validator; the training validator stays unchanged.

Take the first four distinct query representations per (fingerprint, interval
class) and first32 per class 0,1,2,interval. Always exhaust4000 attempts; no
model-based choice or extension. Construction is adequate only with at least
eight query groups,32 pairs and four examples in each exact class0,1,2.
Otherwise report insufficient, do not weaken the gate. Nonunique fingerprints
are grouping safeguards, not certified identities. Known reference endpoints
are disclosed; independence is query-side relative to training and listed
previous panels, not a complete legacy project exposure audit.

Record signed-signature lower and replayed prefix-CC upper bounds without
mirror quotienting. Mirror images are not declared distance zero. Produce
replayed zero-CC lifts of the left endpoint to L160,N25 as a separate view.
Report distance interval violations, class counts/exact-class errors and
per-fingerprint summaries for both trained models, initial model and constants
0,1,2. This tests general short pair distances, not only distances to unknot.
No L10/L1000 inference follows from this metric-only cohort; note28 search
evaluation remains required. Correlations or overlaps between the two new
cohorts must be disclosed instead of pooled as independent trials.
