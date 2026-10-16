# Small head-only ablation after failed root-group gate

Prospective contract, 2026-10-04, frozen before any new ablation result. The
fixed-b16 actor, labels, root roster and target definition remain exactly as
in protocols 264 and 268. Existing split salts 20261005 and 20261006 are
**exploratory** because their results have been examined; they cannot be
presented as a fresh final evaluation. No DEV265 or TEST150 is used to select
models. No MCTS teacher collection occurs in this experiment.

Run three single-factor head-only variations, each on both existing splits:

| Variant | Width | AdamW weight decay | Batch draw |
| --- | ---: | ---: | --- |
| narrow | 32 | 0.01 | 64 attempts uniformly |
| decay | 128 | 0.10 | 64 attempts uniformly |
| grouped | 128 | 0.01 | 64 distinct fit root pairs, then one lambda and one of its ten seeds uniformly per pair |

Keep frozen actor/encoder, three output meanings, masked BCE plus paired
conditional cost loss, initialization seed 20261004, batch 64, AdamW LR
0.001, gradient clip 1, 512 optimizer steps, checkpoint interval 64 and
native MPS with no fallback. Fit only the 96 paired roots/1,920 attempts of
the same split. The group-aware draw changes batch composition, not the
expected root distribution. Source, model dimensions, parameter counts,
split roster, label and actor SHA, optimizer/RNG and actual training CPU and
wall time must be recorded. Equal updates are a training-budget control, not
automatically equal CPU: report actual CPU and parameter count, and do not
claim compute parity if they differ materially.

Score only the fixed step512 checkpoint by the same root/attempt-weighted
Brier and log loss, conditional fit-only success-frequency baseline, masked
paired CC/move errors, and `CC=cap` baseline as protocol268. Repeat the same
CPU inference timing. Compare exploratory results to the original width128
control on the same split, separately L10/L1000 and by cap/length stratum.
Do not promote a variant merely for beating the original control: it must
beat the conditional fit-only baseline in Brier **and** log loss on both
exploratory splits and both lambdas, without >0.10 positive calibration gap
in long cap1/cap2 strata. This strict screen selects at most one variant
by mean root-weighted Brier improvement. If none passes, no confirmation
run and no expensive multi-budget internal-node data collection under these
recipes.

If a variant passes that exploratory screen, confirm it and the unmodified
control on a third predeclared root-group split salt **20261007** using the
same 25% stratification and 512 updates. That is a new split for this
ablation, although all 128 roots have figured in earlier aggregate research;
do not call it an untouched external TEST. Require the same no-worse-than-
fit-conditional baseline gate there, then design a new disjoint DEV panel
before use as an AlphaZero-style leaf evaluator. Preserve every failed arm
and never edit a source or contract after its SHA is pinned.
