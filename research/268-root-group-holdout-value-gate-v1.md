# Root-group holdout gate for the fixed-b16 value head

Prospective contract, 2026-10-04, frozen before new split training or
holdout scores. This is a diagnostic of representation-level generalization
using the already collected TRAIN teacher outcomes, not an AlphaZero search
and not a new teacher collection. See 267 for the motivation. Do not use the
already examined DEV265 panel or TEST150 for checkpoint choice.

Use the immutable actor32768, existing 2,560 root-b16 labels and the same
three-output `BudgetedCostValueHead`/loss, optimizer, batch size, update count
and MPS-only device as protocol264. The 128 distinct paired root
representations, not 2,560 attempts, are the split units. For each root pair,
both lambda variants and all ten teacher seeds are kept together. Partition
independently for split salts 20261005 and 20261006. Define six strata by
CC cap (`1`, `2`, `>=3`) crossed with word length (`<100`, `>=100`). Sort
stable root-pair indices in each stratum by SHA256 of
`"{salt}:{base_root_index}"`, break ties by index; take
`round(0.25 * stratum_size)` as holdout. The six counts are 31, 20, 10, 39,
17, 11, so the holdout is exactly 32 paired roots per split and the fit set
96. Save the explicit roster before any training and its SHA in the manifest.

For each split, initialize a fresh head from the same seed 20261004 and train
512 AdamW steps, batch 64, LR 0.001, weight decay 0.01, clip 1; freeze the
actor encoder/policy. Sample only the 1,920 fit attempts. Checkpoint every 64
steps with AdamW and RNG state. Do not select a checkpoint on heldout data:
evaluate only the final step512. Keep the two splits in separate artifacts.

For each heldout root/lambda/seed, evaluate p_solve at b16 and its same-attempt
conditional paired CC/move predictions. Report attempt-weighted and
root-pair-weighted Brier and binary log loss, separately by lambda and cap/
length stratum. All groups have ten seeds, so the two global point estimates
should coincide; keep both to expose the unit of independence. Fit baseline
success frequencies on fit roots only for each lambda x cap-bin x length-bin,
with Beta(1,1) smoothing to avoid 0/1 log loss. Report also the global
fit-only frequency. For successful attempts only, compare CC and move MAE
against fit-only lambda/cap-bin/length-bin conditional means; also report the
trivial `CC = cap` baseline. Failed bounded searches have no cost target.
Time the frozen encoder + head inference separately on CPU with one thread,
including first-call and warmed passes, and report number of roots/queries.

The gate passes only if, in both splits and for each lambda, the head's
root-weighted Brier and log loss are no worse than the conditional fit-only
baseline and neither long cap1 nor long cap2 stratum has a large positive
calibration gap (>0.10 predicted minus observed). Treat tiny strata as
underpowered and report their counts; no conclusion based on a single seed.
Passing this gate only permits a separately preregistered small internal-node
multi-budget teacher pilot; it does not validate current DEV265 performance
or authorize using this head in value-guided MCTS. Failure points to
representation-level generalization, capacity/regularization, or insufficient
distinct states, not to mathematical impossibility of neural guidance.

Pin protocol, labels, roots, actor, source and tests by SHA in each run. The
script must fail closed on mismatched labels/splits/checkpoints, preserve all
prior artifacts and not commit/push/publish anything.
