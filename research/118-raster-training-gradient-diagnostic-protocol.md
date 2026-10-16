# TRAIN-only decomposition of protocol97 gradients

2026-09-21. Diagnostic only; no optimizer, new training, search, TEST scoring,
checkpoint selection, weight retuning or modification of the active evaluation.

## Question and scope

The three tasks accumulate gradients before a single global norm clip and AdamW
step. The metric loss is a weighted interval mean; positive-policy loss is a
multi-positive mean with weight0.1. Full cancelled relative comparisons occur
every16 updates at weight0.16. Their scalar losses alone cannot establish their
relative influence on the shared encoder. Test whether their gradient magnitudes
and directions differ, without treating a conflict as proof of poor search.

Use both initial and final checkpoints of BOTH protocol97 arms. Reconstruct each
arm's OWN initial reference log probabilities with the original microbatch1 and
require its hash to equal the training manifest. All four COMPLETE cancelled
relative groups (334,52,74,54 states) are included, never sampled or independently
trajectory-normalized. No cache in any neural forward.

Before observing these gradients, fix one batch: reproduce the first16 TRAIN
draws using protocol97 seed20261047 and select draw16 (the first scheduled
relative update). Use its32 metric rows and8 positive states at every checkpoint.
Compare every full relative group with this same batch, not only the group drawn
at step16. This is a bounded diagnostic cross-product, NOT a replay of the actual
step16 update and NOT an estimate over all training batches. Save the full draw
trace, group coefficients/contexts and reference scores.

## Measurements and gates

For each checkpoint separately accumulate weighted gradients for metric,
positive, relative0, relative1, relative2, relative3, zeroing gradients between
tasks. Save complete flattened CPU gradient vectors plus parameter names/shapes,
module partitions and missing-gradient flags. Report norms and pairwise dot
products/cosines globally and by module. An absent/zero vector has undefined
cosine (null), not cosine0. Check finite values and exhaustive disjoint module
coverage. Independently reconstruct ordinary and four scheduled gradient sums,
their pre-clip norms and hypothetical global clip factors at limit1.

Shared-encoder blocks: local raster trunk; project/message/context/pool. Separate
blocks: each of knot/layout/isotopy output heads; position; policy head; metric
objective. All parameter gradients use the actual training weights. Report the
helpers' unweighted losses distinctly from those weighted gradients.

CPU float32, one thread, model train mode for gradients and eval for fixed
references, deterministic current architecture. Pin completed manifests, initial
and final checkpoints, inherited inputs/sources, and diagnostic sources. Check
model/objective state bytes before and after every component; no optimizer is
constructed. Recheck file hashes before committing the final diagnostic report.
Use a NEW exclusive artifact directory, with one immutable completed result per
checkpoint so partial work is distinguishable from a completed four-case audit.

## Interpretation limits

Negative dot products diagnose local task disagreement for these fixed examples.
They do not establish causal generalization damage or actual AdamW updates:
momenta, per-coordinate preconditioning, weight decay and changing parameters
matter. Periodic weight0.16 is not equivalent to weight0.01 each step after
clipping/AdamW. A large norm does not automatically justify reweighting a loss.
Neither the relative loss nor a censored rollout labels a state unsolvable.
Metric coordinate called `chord` in code is1-cosine, proportional to squared
Euclidean chord distance for unit vectors, not sqrt(2-2*cosine).

Completion of this diagnostic is not scientific promotion. The frozen paired
MCTS/replay evaluation remains the outstanding search-utility evidence. Any next
training change needs an explicitly documented TRAIN/validation rationale and a
fresh evaluation contract; do not tune on already observed TEST70/74 scores.
