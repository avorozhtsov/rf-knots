# Matched R2/R4 repair: bounded exact-context F objective

2026-10-07. The frozen protocol303 pilot exposed a substantive optimization
failure. Its signed *linear* log-policy objective has no finite lower bound:
for a negative coefficient, sending that action's log probability to minus
infinity sends the reported loss to minus infinity. At checkpoints 1, 1024,
3072, 5888, R2 F loss was approximately -13, -1885, -76190, -2168897;
R4 was -8, -1946, -78071, -2145707. Shared-encoder gradient norms rose
from about 200 to above one million. Clipping the combined gradient did not
make the objective bounded. The v1 run is preserved as a failed experiment,
not selected by its TRAIN loss. No v1 weights initialize this repair.

Use the same independently audited TRAIN bank, parent checkpoint, eligible
root order, R2/R4 shared-weight graph rounds, MPS, AdamW, auxiliary labels,
known-solution/distance rehearsal, 10% and 20% shared-encoder gradient caps,
checkpoint/resume fields, and 12 active-hour target as protocol303. Change
only the F objective: for each exact state and remaining-resource context,
use all nonzero signed F coefficients (at most ten per bank group) and form
all positive-action/strictly-worse-action pairs at that context. Weight a
pair by the product of normalized positive and negative coefficient masses,
then weight contexts by their positive mass. Minimize
`softplus(0.5 + log_policy(negative) - log_policy(positive))` averaged over
those pairs. The loss is nonnegative and rewards a margin without an
unbounded negative-action term. Do not pair actions from different states or
turn capped no-solve into a negative. The exact finite group removes the
old stochastic coefficient imbalance. Require both signs at every included
context, finite positive F loss and nonzero finite F encoder gradient.

Run new R2/R4 smoke checkpoints from the unchanged curriculum1024 parent in
separate artifacts, independently verify SHA, AdamW/RNG and finite loss,
then interleave equal-step blocks. Pin the wrapper script SHA in the trainer
contract; never modify the old frozen trainer/protocol/artifacts. The fresh
DEV panel `artifacts/f-r2-r4-fresh-dev-panel-v1/panel.json` SHA
`2465a02e8d32a60c6d6769d92865e895b9663151122710f029562b978b726a30`
was score-blindly frozen before either repaired arm's search outcome; it
remains evaluation-only. Freeze the evaluator before scoring. CPU and neural
work must still be reported and the research/273 selection guardrails apply.
