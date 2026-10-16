# Root-value calibration failure: stratified diagnosis and next gate

2026-10-04. Read together with protocols 264–266. This is a post-result
diagnostic, **not** a revision of the frozen root-b16 experiment or a new
positive MCTS result. No search or training worker was live when checked.
The step-512 checkpoint still has SHA256
`9870fa79f60e8585d37446a093d8ebd58b06485c28a69ffb1e428cabd90930c8`;
the DEV report still has SHA256
`308d6833ebbfc372ab30ee286ad2191402bd20c308aa631280754f1eda2fc5cb`.

## What the aggregate hid

The training set contains 2,560 **attempts** but only 128 distinct paired
root representations (ten seeds per lambda). It is not 2,560 independent
states. TRAIN word length ranges 13–208, median 112; long DEV ranges 74–137,
median 123. Therefore a simple claim that the long DEV roots are outside the
TRAIN length range is false. TRAIN has 1,400 attempts at length >=100,
898 successes. Long DEV has 104 such attempts, 62 successes.

At length >=100 and crossing-change cap 1, TRAIN success is 158/200 (L10)
and 134/200 (L1000); long DEV is 19/24 and 17/24. At cap 2, TRAIN is
222/390 and 198/390; long DEV is 14/28 and 12/28. Thus simple length-plus-cap
stratification does not explain the large probability overconfidence: the
empirical bounded success rates are broadly similar within those strata,
while the head places almost every long DEV prediction in the 0.8–1.0 bin.
One plausible, unproven mechanism is memorization/overfitting of the 128
TRAIN representations across repeated seeds. Check a **root-group holdout**
inside TRAIN before attributing the failure to architecture or optimizer.
The existing frozen head was not trained with that split.

On long DEV, all 67 successful attempts used exactly the permitted CC cap:
41/41 successful cap-1 attempts used CC=1 and 26/26 successful cap-2 attempts
used CC=2. The constant predictor `CC = remaining CC cap` therefore has
**zero** MAE on successful long DEV attempts, better than the head's roughly
0.01–0.02 MAE. The prior comparison against an unconditional TRAIN mean
understated this trivial conditional baseline; near-zero long CC error is
not evidence of learned path-cost information. The move output improves
modestly against the unconditional TRAIN mean but requires a cap-conditioned
baseline and root-group holdout too.

No direct evidence of a training/evaluation feature-schema mismatch emerged:
both paths load the same actor checkpoint, encoder and value-head class and
pass the same four context quantities. That check is not a proof of parity;
an explicit CPU-vs-MPS encoded-feature and head-output parity test remains
useful. The checkpoint's near-exact fit to TRAIN per-cap success fractions
(for example L10 cap1 predicted 0.909 vs observed 0.918, cap2 0.623 vs
0.622) together with poor unseen long-root calibration is consistent with
group-level overfit, not with a generic inability to learn the labels.

## Predeclared next gate before costly internal-node collection

Create a separately versioned, root-group-disjoint bootstrap experiment from
the existing TRAIN teacher outcomes. Partition the 128 paired roots once by
stable root identity, stratified by CC cap (1, 2, >=3), length (<100, >=100),
and both lambda results; keep all ten seeds and both objectives of each root
in the **same** side. Train only the value head from the same frozen actor
and optimizer budget, reserving at least 25% of paired roots for the heldout
gate. Score p_solve by root-weighted and attempt-weighted Brier/log loss,
with a cap/length-conditioned empirical constant estimated from TRAIN roots
only. On successes, compare paired CC/move MAE to cap-conditional constants,
including `CC = cap`. No checkpoint selection on the heldout split. Repeat
with a second predetermined split seed for uncertainty, not post-hoc cherry
picking. Record CPU inference separately.

If that gate fails, do not spend a large search budget on multi-budget
internal-node labels under the same head/data recipe. Investigate regularized
or lower-capacity head, larger *distinct* TRAIN-state coverage, and
group-aware sampling under a new prospective contract. If the gate passes,
predeclare a small stratified multi-budget internal-node collection: states
must be selected from TRAIN-owned tree prefixes before seeing their future
outcomes; keep root and exact-state group ownership, check collision with
heldout roots; use fresh independent continuations at fixed local b values
such as 4/16/64, never `B_root - used_simulations`; retain bounded failures
only for p_solve and fully replay successful paired CC/moves. Use new disjoint
DEV states for calibration before any AlphaZero-style leaf-search comparison.
The existing DEV panel has already influenced this diagnosis and is not a
pristine final test for a revised head. TEST150 remains excluded from choice.
