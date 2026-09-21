# Prospective synthetic ordinary-distance test with nontrivial references

2026-09-21. Freeze before generating/scoring this test. This is a separate
synthetic-family experiment, not a replacement success for the inadequate70
sidecar test or the diagnostic72. Training69 and its14-job short-u/search
controller remain unchanged. Evaluate the same three final checkpoints: initial
joint57, final graph69 and final old-data control69; do not select checkpoints.

## Construction, fixed effort and ownership

Exactly3,072 attempts, seed20261036. Attempt i uses `random.Random(seed+i)`.
Let N=3+(floor(i/3) mod31), so all three distance constructions cover every N3..33.
Start with the positive spanning chain sigma1...sigma(N-1), whose permutation
is an N-cycle. Append B=8+rng.randrange(25) pure blocks. Each block is

```
w sigma_j^(2*epsilon) w^(-1)
```

where j is uniform1..N-1; conjugator length is uniform0..3 and each of its
letters chooses a generator uniformly1..N-1 and sign uniformly+/-; inverse
reverses order and negates letters. Draw order per block: conjugator length,
conjugator letters (generator then sign), central generator, central sign.
Consume the central-sign draw in both modes. For even floor(i/93), epsilon=+1;
for odd floor(i/93), use the drawn sign. The two modes therefore sample positive
central twists versus mixed central twists, not uniformly distributed knots.
Every block has identity permutation, so the reference closure remains one
component. Verify the actual permutation/closure too; do not rely only on this
construction argument. Word lengths and knot complexity are distinct.

Let k=i mod3. For k1/2, sample k distinct blocks uniformly without replacement,
flip the **first** of the two central letters in sampled order, and reverse the
resulting path so the modified word is query and the original reference is the
right endpoint. Each is an actual ordinary Artin crossing change. No mirrored
quotient edges or hidden strand relabeling are used. For k0, rotate the reference
by a uniform offset in0..L-1 to form the query, and retain an explicit inverse
cyclic-coordinate certificate to the reference. Do not call it a0CC path with
an invented literal CC transcript.

Recompute signed signature and determinant of every path state and independently
check determinant via the integer Burau calculation. Lower bound is
max(abs(sigma(query)-sigma(reference))/2, determinant inequality). Upper bound is
k unless signed cyclic-coordinate equality independently certifies0. Retain
actual CC witness count separately. Do not treat a(1,0) fingerprint as proof of
unknottedness. Reject references with fingerprint(1,0) as *uncertified nontrivial*
for this panel; the retained references are nontrivial by determinant/signature.

Query ownership must be invariant TEST, reference not quarantine. Query groups
must be absent from the complete audited protocol69 optimizer lineage, the15
historical query panels frozen there, and all eligible query groups from the
completed diagnostic72. The simultaneous unseen short-u panel70 is a different
task distribution, not a historical scored test; report any query-group overlap
with it separately and do not pool the two as independent replications.

In attempt order, keep the first32 pairs per category exact0/exact1/exact2/interval,
at most4 per query invariant group across all categories, with unordered signed
cyclic endpoint-pair deduplication. Reject quota/duplicate candidates without
changing orientation or creating additional attempts. Keep and audit all attempt
records/rejection reasons. No per-size selection or post-hoc score filtering.

Require at least32 retained pairs,8 query groups, and8 pairs in **each** exact0,
exact1 and exact2 class. Otherwise preserve an inadequate supply result; no top-up
or seed/size change in this experiment. Before scoring, independently audit the
construction, replay, bounds, full selection, ownership and frozen inputs.

## Evaluation and reporting

Use the final frozen checkpoints, the existing ordinary-distance objective scale,
and no fitted TEST calibration/normalization. Compare constant0/1/2. Report
per-row and equal-query-group interval violation, exact-class MAE (especially2),
and raw query/reference size distributions separately. Explicitly retain that
exact0 pairs are coordinate invariance checks, not separate knot-distance tasks.
No lifted views are generated. Report raw size strata relative to L254/N21 TRAIN;
an axis with no observed out-of-envelope pairs has no extrapolation evidence.

This panel measures general d(K1,K2) around a declared pure-block family. It does
not measure u(K), provide shortest full unknotting witnesses, prove a catalogue
distribution gain, or demonstrate unlimited-size solving. Pair count is not a
count of independent knot types. Search evidence remains the separately replayed
short-u/full-policy comparison from70. No automatic scientific promotion.
