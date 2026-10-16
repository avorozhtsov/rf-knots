# Protocol94 interrupted: numerical preference-objective failure, not an embedding result

2026-09-21. The q254 arm actually executed16 committed optimizer updates after
the matched initialization check:267657 parameters, identical upper layers,
21 differing local tensor keys between planned q254/random initializations.
The random arm never started. All1125 pinned inputs and61 sources were checked.

The first checkpoint exposed a failure of the exponential relative surrogate
when used with a new policy head. Only this new worker85096 was intentionally
interrupted with SIGINT; launcher85055 exited with child returncode-2. Do NOT
resume94 unchanged. This is not a pause or cancellation of the overall goal.
No old training worker was stopped and no artifacts were removed.

Saved checkpoint: `artifacts/pretrained-raster-training-pair-v1/q254/progress.pt`,
SHA256 `9f09da952816b5b2838caf87f8cdda6bc0dfe06dcdd40aec39a0e3e8925fe20f`.
All saved model/objective tensors are finite; every populated optimizer state
reports16 updates. Initial weights, RNG, optimizer, source snapshots and logs
are retained. The child manifest is its last pre-interruption snapshot, not
proof of a currently live worker. `intentional-stop.json` disambiguates this.

## Observed numbers and diagnosis

First16 updates took93.95 seconds after preparation. At update16: metric
loss0.73418, positive-set loss44.91878, relative loss4,487,777,924.16 and total
pre-clipping gradient norm4366.27. This is not held-out quality evidence.

Diagnostic: `artifacts/pretrained-raster-relative-loss-diagnostic-v1/report.json`,
SHA256 `6b6237ca3be232f4998e571f6e923161df6971ab4f27ee02960603eff9ca2744`.
All1137 diagnostic pins were rechecked. The script reconstructs the exact final
data RNG, including sampling indices, without replaying a different batch.

The last selected complete relative group has74 states. For one sampled loser
action, log probability changed from-100.48970 to-58.34276, a difference+42.14694.
The old formula uses exp(clamp(log_ratio,-20,20)); multiplying exp(20) by the
sample coefficient magnitude9.25 explains the roughly4.49e9 penalty. At the
saved post-update16 model, ALL EIGHT sampled derivatives with respect to logp
are zero: hard clamping and the per-action clipped minimum make a flat plateau.

This is NOT a NaN/overflow event and does NOT prove a4.49e9 gradient. The logged
norm is for the combined training objective; the diagnosed relative term has
zero sampled logp gradients at the saved state. The diagnostic uses the saved
post-update16 checkpoint, not an unavailable exact pre-update16 model. Its
relative loss is4,487,777,792, consistent with the identified saturated term.

The aggregate full-group log-preference change is+876.92 even though a particular
loser action's probability increased. Per-action exponential penalties and
whole-trajectory comparisons are not the same target. This matters for the
user's original same-root, shared-prefix cancellation proposal.

## Implemented replacement candidate, not yet a restarted experiment

New isolated module: `scripts/trajectory_log_preference.py`. For coefficients c
already aggregated across identical state/action contexts, define

    S = sum_i c_i * (log pi_theta(a_i | s_i) - log pi_initial(a_i | s_i))
    C = sum_i abs(c_i)
    L_relative = 2*C*softplus(-S/C), if C>0; otherwise0.

This compares the complete cancelled trajectory group in log space. A common
scale C measures the total surviving coefficient mass; it is NOT independent
normalization of each trajectory by its own length. Therefore a shared prefix
with coefficient0 stays exactly absent. Winner/loser coefficient ratios remain
the requested ones (e.g.+1 versus-1/9).

At the reference policy, dL/dlogpi=-c, matching the original local gradient
direction and scale. Everywhere |dL/dlogpi|<=2|c|. There is no exp(log_ratio)
importance weight or hard log-ratio clamp. Normalizing S by the one common C
also makes one nat of average change the logistic scale, without an additional
tuned temperature. This is a preference loss, not PPO or a calibrated value.

The implementation computes the COMPLETE group's log probabilities without
gradients, derives the bounded logp derivatives, and then recomputes small
microbatches for exact parameter-gradient accumulation. No nonlinear loss is
applied to an eight-row Monte Carlo estimate and mislabeled unbiased. Stateful
normalization/dropout modules are rejected; deterministic two-pass agreement is
checked. The current raster/graph model has neither such training mechanism.

Eleven tests passed: reference-gradient identity, finite extreme log ratios,
nonzero adverse-preference gradient beyond+20, zero shared-prefix influence,
fully cancelled groups and two-pass/full-gradient equivalence at three batch
sizes. These are mathematical/software tests, NOT a real-network retraining
or a demonstration of lower L10/L1000. The source94 files remain frozen.

Before restarting: real-network gradient/numerical gate; a new prospective
protocol and separate output lineage; explicit compute accounting for complete
groups and any chosen update cadence. Preserve identical treatment in q254 and
random arms. Recheck metric and positive-policy learning rather than declaring
the candidate fixed from unit tests alone. Do not quietly alter94 or relabel
its interrupted partial run as a completed comparison.

## Older independent experiment

Both original69 arms completed4096/4096 without interruption. Final checkpoint
SHA256 values were checked against actual files:

- graph: `dd3c8d8396da69b9f0ff76747b1b57019e5ea23a33b9bfbb1a0eb8668748da53`
- control: `e72b159d7ddb3db27b1cc491cde23203b91cfb6ec7f66749f1e25abc52607689`

The main70 evaluation had completed4/14 jobs at the last snapshot and remained
live; the74 general-distance queue had completed. No metric scores from those
new outputs were used to choose the proposed relative-objective replacement:
the diagnosis and candidate above use TRAIN observations and algebra only.
Their scientific results still need complete review and search/cost audits.
