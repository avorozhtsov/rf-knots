# Diverse short-u: audit and deterministic corpus conversion

2026-09-20. Follows the fully exhausted supply probe in note40. This fixes data
conversion before running it; a training/evaluation protocol remains separate.
No checkpoint is selected or trained during this step.

The probe accepted344 roots in300 fingerprint groups:163 exact-u1,53 exact-u2
and128 intervals[1,2], using determinant-strengthened bounds. Raw length reaches
208 and conservative normalization leaves lengths up to204; strands reach17.
Fingerprint groups are not knot IDs, and normalized length is not minimal crossing
number. These are new relative to the declared previous TRAIN endpoint groups.

The independent audit replays EVERY accepted witness and its CC endpoints,
checks train ownership and disjointness from old validation/current paired-test
queries, recomputes normalization with the replay reference, and independently
checks every root determinant. The initial full-polynomial audit was stopped
after32 verified roots solely because of unnecessary polynomial interpolation;
it produced no final audit artifact. The replacement computes the same exact
integer through an unreduced Burau cofactor at t=-1, with no Seifert dependence.
Its identity is [Lemma2.10, section2.4.4](https://people.tamu.edu/~rowell/DRWv10.pdf).
Tests compare it with the old polynomial implementation on small random knots,
known knots, even-strand stabilizations, and exact determinant expansion.
The complete replacement audit must pass before conversion; no reduced sample.

## Fixed conversion

Keep all4096 existing metric TRAIN records and keep validation/size-shift
records byte-for-byte equal as JSON objects. For EACH accepted candidate:

1. Add the full root-to-unknot anchor.
2. If its witness has two CC, also add root-to-first-post-CC as a general pair
   with upper bound1, using the full witness and the appropriate prefix length.
3. For that two-CC case add first-post-CC-to-unknot using the remaining witness.

Deduplicate ordered RAW endpoint pairs against old and new records, not by knot
ID. Do not add artificial size lifts or select by network predictions. Verify
every added record with the existing full-path verifier; all path states must
remain TRAIN-owned. Recompute count/category/group metadata. Record indices,
source attempt and derivation kind. Retain signed-signature interval lower
bounds in the existing schema; store determinant-difference lower bounds as
separate metadata, never silently change the meaning of the old loader.
An eventual opt-in training sampler may use the stronger bound after explicitly
checking the invariant difference. Unknown intervals remain intervals.

Expected BEFORE deduplication:344 root anchors plus two records for each of the
181 two-CC witnesses, hence706 additions. This is not706 independent knots.
Output `artifacts/embedding-diverse-short-u-corpus-v1`. Immutable source/input
hashes, single writer, reproducible replay, no test data in TRAIN. These old
validation views remain development diagnostics; a future promotion requires
new prospective held-out data rather than recycling the completed paired test.
