# Matched R2/R4 repair: calibrated, non-saturating F pairs

2026-10-07. The frozen linear v1 pilot is unbounded below and invalid.
The separate bounded-softplus v2 pilot fixed that defect but stopped on
R2 attempt 289 after checkpoint 288: an already highly separated pair had
zero F encoder gradient. Its fail-closed guard prevented auxiliary-only
training. V2 R4 had checkpoint 257. Both histories are preserved. The v3
pilot starts anew from the shared curriculum1024 parent, not either failed
pilot's checkpoint.

Keep the same audited bank, root and exact F-context ordering, R2/R4
shared-weight graph rounds, one native MPS, optimizer, auxiliary and
rehearsal gradient caps, checkpoint/RNG/AdamW structure, and 12 active-hour
per-arm target from protocols303/305. Change only the exact-context F pair
loss to binary cross-entropy with a prospective 0.98 soft target:

`0.98 * softplus(z) + 0.02 * softplus(-z)`, where
`z = 0.5 + log_policy(negative) - log_policy(positive)`.

Pair/context weights remain those of protocol305. This function is
nonnegative, has finite optimum at `z = log(0.02/0.98)`, and its derivative
approaches -0.02 rather than zero for an overconfident positive action.
It also penalizes extreme separation instead of allowing infinitely large
logits. A failed capped search never becomes a negative. Confirm by tests
at equal and extreme logits, then one native-MPS checkpoint per arm and a
matched 128-step stability block. Only then start the long alternating
scheduler. Preserve old versions unmodified and pin source hashes in the
new contracts.

The prospective DEV panel remains the previously frozen
`artifacts/f-r2-r4-fresh-dev-panel-v1/panel.json` SHA
`2465a02e8d32a60c6d6769d92865e895b9663151122710f029562b978b726a30`.
Freeze evaluator and checkpoint rule before reading either arm's search
outcome. The 24-active-hour decision clock restarts for v3; prior active
seconds do not count. Research/273 selection and compute guardrails apply.
