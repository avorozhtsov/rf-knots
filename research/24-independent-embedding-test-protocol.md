# Prospective held-out embedding check

2026-09-20. Written before evaluating any model on the hash-test partition.
This is a bounded experiment, not a proof of arbitrary-size knot solving.

## Frozen hypotheses and models

Primary candidate: `embedding-short-u-augmented-joint-v1`, checkpoint SHA
`b3411589f8358350016cbfe49fe2231e48ab1521e282e31c34b7d5d0ebc41ff4`.
Controls: its same seeded, untrained architecture and length-only search.

Secondary, prospectively declared comparison: the currently running
`embedding-balanced-root-u-joint-v1` versus `embedding-balanced-dense-u-joint-v1`,
both seed 20260922, 4096 updates, batch64, same policy stream. Their final hashes
will be recorded after completion; no test-dependent checkpoint selection.
The question is whether full-path u supervision helps versus matched roots.

All models above are fixed independently of test scores. No training, parameter
tuning, root filtering, budget changes or model selection will use this panel's
results. A failure is reported, not repaired on the same panel and called a fresh
independent success. Later research may use a new held-out partition/protocol.

## Data

Generate 128 short-u queries with seed 20260930, at most 100000 attempts and
at most 16 representations per `(det, abs(signature))` group. Use the existing
generator configuration and immutable `owner` split: only `owner == test`.
All historical and current train paths exclude this partition. Record actual
group count; 128 representations are not 128 independent knot types.

Keep the original replayed upper bounds and signature lower bounds. Report an
additional, explicitly identified stronger lower bound
`max(abs(signature)/2, int(det != 1))`, certified because a knot of determinant
different from the unknot's cannot have u=0. Equal determinant does not certify
identity or u=0. Never turn a random-walk CC count into a lower bound.

Save zero-CC lifted copies with 9 strands and length >=60. They test stability
under those rewrites, not new knot types or arbitrary-complexity generalisation.
Partial generation is not silently relabelled as a completed 128-query panel.

## Measurements

Distance: interval violation and exact-u class errors on original and lifted
views; always disclose class counts. Compare also with constant predictions
0,1,2; do not hide a strong constant baseline behind a weak random network.
Report original and strengthened bounds separately, with no retraining in between.

Search: all generated original queries, not a score-selected subset. Use the
previously fixed anytime greedy + exact layout tie-break, not MCTS. Neural scores
only order the frontier; never use them for proof-producing pruning.
Resource caps must include each query's known replayed witness; all arms get the
same caps. Primary search comparison: length128 versus length64 + neural64.
Repeat for actual L10 and L1000 and keep every failure censored. Compare the
trained portfolio to the untrained portfolio, not only to length128.

Replay every returned witness independently. Show newly solved/lost cases,
lower/higher/equal costs on common solutions, per-group results, actual neural
state evaluations and recorded search time. Averages over solved subsets cannot
substitute for failure-aware paired results. Do not call these witnesses optimal.

No claim of general superiority follows from a small raw success-count change.
This panel is independent of development, but multiple model comparisons and
correlation within knot groups remain limitations. MCTS improvement and large RL
remain separate requirements, not consequences of success in this diagnostic.

## Mathematical provenance

Determinant is a knot invariant obtained from the Alexander polynomial at -1;
the unknot has Alexander polynomial 1. Therefore unequal determinants prove
different knot types and a nonzero Gordian distance (integer, hence at least 1).
See [KnotInfo determinant](https://knotinfo.org/linkinfo/descriptions/determinant.html)
and [KnotInfo Alexander-Conway normalisation](https://knotinfo.math.indiana.edu/descriptions/conway_polynomial.html).
This extra bound is mathematical, not a model-score-dependent relabelling.

## Prospective additional cohort (before any test scoring)

The random generator is filling the reserved partition slowly and produces few
u1 certificates. Keep its original 128/100000 contract unchanged. Separately
import **all** Artin-replayable witnesses with declared CC<=2 and owner=test from
UnknotDB's existing `outputs/rf-best-witnesses-20260815-v0.tsv`, SHA
`aa66aa7e1541b77c934053166fe25b74f7e21c514cec8d2d4a292dc66ec069d2`.
This source predates all current models. Source metadata has 108 roots, 45 with
CC<=2, and 11 test-partition candidates in three fingerprint groups before full
replay. Eligibility is based only on source/replay/split, never model scores.
Record every replay rejection. Do not merge these small historical cohorts into
one apparently large independent sample: historical successful-search selection
and within-type correlation are explicit limitations. Use the same distance,
constant-baseline and paired-search protocol and report this cohort separately.

## Completion record (not a protocol change)

Both secondary trainings completed their declared 4096 steps. Root checkpoint:
`7b03c5021a1c56cd65c0d48d0996b4735fd88ecd94897b74034763b2ac6714af`;
dense: `fec476b229c0eb269ac878e7f488d68f4fb3609083e90d35f50771b7cae2ea9b`.
Main-pair and policy RNG states match between these runs.

Random generation exhausted 100000 attempts at 55/128 queries, eight groups;
retained as partial and NOT scored. The historical cohort completed with 11/11
eligible roots replayed, no replay rejects, three groups. Panel SHA
`b8618f4005e6f889128c30aa3504b48ac55f07465c16955f2ca544775fe5636a`.
All declared models and controls scored on that cohort; reports and replay-audited
paired/grouped comparisons are retained in
`artifacts/embedding-external-independent-test-v1/`.
Findings and limitations are in research note 23. No follow-up tuning is authorised
by calling this test an unused holdout again.
