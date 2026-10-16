# Matched F-dominant R2/R4 short training contract

2026-10-07. This contract follows the independently audited joint TRAIN
bank (`artifacts/joint-f-bank-v2/manifest.json` SHA
`6a61a0ec0bf47c59cdb601b633c962dc4b2d4df3b46ba9a2ca429fab49f19a12`,
audit SHA `2a4a7a19f6d55da6d64533fba41a9594ce85ad89c808205b0984382ba297dc73`):
392 distinct F-usable roots, 690 resource/objective groups, 9,798
verified completed paths. The matched parent for both architectures is
curriculum1024 checkpoint SHA
`03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8`.
This contract does not assert training has started.

R2 and R4 use exactly the same parameter tensors, initial weights,
policy head, bank, eligible-root order, optimizer and pseudorandom seed.
The only architecture difference is two versus four applications of the
same shared-weight graph message block. Use native MPS with CPU fallback
disabled, one trainer at a time. One deterministic group is selected per
step by cycling sorted eligible roots and then cycling that root's groups.
Sample 16 signed F state/action coefficients per step, half from each
sign, using `sample_coefficients`; this is unbiased for the *linear F
gradient*, unlike a nonlinear loss on sampled terms. The primary loss is
`-sum(coefficient * log(policy(action | exact state, resources)))` over
the sampled terms. No bounded failure enters the loss.

An auxiliary two-output head sees the root knot embedding and normalized
lambda/remaining CC/move resources; normalized targets are the best
observed successful path's CC and moves for that objective. Use smooth-L1
and measure its shared-encoder gradient separately each step, scaling
it to at most 10% of the F encoder-gradient norm. A known-reference
action imitation loss plus the inherited root-to-unknot CC metric loss
form one rehearsal component, measured and scaled to at most 20% of the
F encoder-gradient norm. Log component norms/cosines and actual applied
scales. If F encoder gradient is zero/nonfinite, skip optimizer update
and record why; never silently train only auxiliary/rehearsal. Clip the
combined gradient at 1.0. AdamW LR 5e-5, weight decay .01. Initial
sampling seed 20261007. Both arms must save model, metric objective,
auxiliary head, optimizer, Python/Torch/MPS RNG states, counters and
measured CPU/active wall time in SHA-pinned checkpoints every 16 steps.
An operational one-step smoke checkpoint is allowed before the 16-step
cadence and must preserve exact resume state. No checkpoint is selected
using TRAIN loss.

First run one real optimizer-step smoke checkpoint per arm, audit SHA,
AdamW/RNG and native MPS, then alternate equal checkpointed blocks on
the single MPS. The paired architecture decision targets 12 *active*
training hours per arm, not elapsed wall-clock or MCTS collection. The
24-hour clock starts only after both first valid checkpoints. The
1,000+ distinct F-root target gates later scale-up, not this short
matched pilot. Freeze a new score-blind validation-owned DEV panel
disjoint from TRAIN and previously viewed DEV panels before examining
the R2/R4 search comparison; primary L1000 distinct solved roots with
the guardrails in research/273. Never use TEST150 for selection.

No cloud purchase, database publication, commit, merge or push.
