# MCTS tournament policy sweep: eight idea-(3) variants plus control

2026-10-02. New experiment, requested after the `units/raw` results were judged
uninteresting. This is a prospective protocol, not a modification or early stop
of the active 248 continuation. Preparation, value-head bootstrap and a bounded
CPU pilot may use an already saved, SHA-pinned checkpoint while 248 continues;
the final 248 checkpoint is not a prerequisite. Do not run a resource-heavy
eight-arm sweep concurrently with healthy 248. No TEST150 outcome, old139/155 search result or
held-out trajectory may select a variant or become a training target.

User's subsequent value-guided-search requirement is specified separately in
`research/256-cc-move-leaf-value-protocol.md`: the new leaf evaluator must
use an AlphaZero-style multi-output policy/value head, including separate
estimates of remaining CC and moves from the same conditional solution
distribution alongside bounded success probability. Gate that search
architecture on DEV and matched compute before using its collected paths for
the eight idea-(3) variants plus control below.

## Why new search data are required

The existing completed, audited `diverse-mcts-supervision-diamonds-v1` has
19,189 set-positive states but only four conditions with cost-preference groups
(514 aggregate state/action terms), despite 19 conditions with alternative
success paths among 256 conditions. Repeating those four groups for eight
models would not test a general MCTS-learning method. Existing failures are
compute-censored, not negative action labels. Counts must be independently
rechecked with the frozen input hashes before use.

## Frozen collection before any variant training

Select a single already saved starting actor from TRAIN/DEV evidence only;
record its SHA, architecture, selection criterion and any limits of that
criterion. A checkpoint still being updated by 248 is not an input: pin an
immutable earlier saved file. The actor needs a separately trained and
calibrated multi-output value head before value-guided MCTS collection; its
existing policy/embedding weights are only initialization. A DEV metric gain
alone is not evidence of search benefit. Do not read TEST150 for this choice.

Freeze a TRAIN-owned root roster, cost objectives L10 and L1000, exact caps,
legal action generator, network/search version, per-search simulation/network
evaluation limit, internal horizon, seeds and wall-time counters before running.
Use the canonical pgx-mcts-bench integration for search comparisons, not
local-ablation. For each root/objective, launch S=10 independent searches in
the pilot, extending to S=20 only under the same predeclared collection rule.
Each found solution must pass complete state/action replay, CC/move/resource
checks and TRAIN ownership. Deduplicate exact paths; count distinct roots,
distinct winning/losing complete paths and divergence nodes separately from
the number of runs. Missing solution at the cap is censored and cannot be an
ordinary negative action label. Save full run provenance, objective, actor and
source hashes, search costs, timing, censoring, and resumable shards.

Collection adequacy gate before the nine-arm training sweep: at least 16
distinct TRAIN root/objective groups with two replayed successful paths of
different cost, and at least 64 distinct state/resource divergence contexts.
These are design minima, not a guarantee of statistical power. If absent,
report the yield and try a TRAIN-only easier-root curriculum or additional
independent seeds under a new pinned collection contract; do not fabricate
losses from failures. Keep DEV roots and any new final TEST disjoint by query
group and audited ancestry.

## Targets and eight matched variants plus control

For each group, compare only identical root, objective, caps, actor version and
search compute. Rank completed solutions by the predeclared objective; use
separate L10 and L1000 groups. Ties are not losers. The best known complete
path is not asserted optimal. In a comparison with K losers, positives share
total coefficient +1 and strictly worse complete paths share -1/K. Aggregate
identical state/resource/action coefficients exactly before computing policy
loss, so a common prefix cancels. Never normalize each path independently by
its length. At a divergence, compare actions using identical state AND
remaining-resource context. A high-CC completed path can supply an additional
lexicographic CC preference, but only as an explicit separate objective; do
not silently count it twice in L10/L1000 cost ranking.

Use the same initialization, frozen collected data, optimizer-step count,
policy candidate pools, CPU accounting and random seeds for all variants:

| Arm | MCTS tournament signal | Distance/demo mixture |
| --- | --- | --- |
| A | none (control) | current metric + set-positive demonstration + old relative loss |
| B | first divergence, one sampled strictly worse success | same as A |
| C | first divergence, all up to K worse successes, total negative weight -1 | same as A |
| D | full divergent suffix, K negatives, common-prefix cancellation | same as A |
| E | compare branching actions at every exact shared state-and-resource context of a best path and K worse paths | same as A |
| F | E with 1–3 near-best positive paths, fixed cost margin | same as A |
| G | E plus separately reported high-CC preference | same as A |
| H | E with staged policy-heavy mixing | retain metric/demo rehearsal; altered mix is the named intervention |
| I | E with cost-gap-weighted strictly worse successes; clipped positive gaps, weights renormalized to total -1 | same as A |

Arm I tests whether the magnitude of a verified solution-cost difference
matters beyond its sign. For each positive path and its K strictly worse
completed paths, define raw loser weight
`min(gap / gap_scale, gap_cap)`, where
`gap = (lambda*CC + moves)_loser - (lambda*CC + moves)_positive > 0`.
Normalize those K positive raw weights to total negative mass -1; positive
mass remains +1. Pin `gap_scale` and `gap_cap` on TRAIN before DEV. This
never converts a censored run into a loser.

Fix K, near-best cost margin, S extension rule, update count and candidate
loss schedules in the run contract before DEV scoring. Do not choose positives
with `mean +/- 2 sigma` from 10–20 censored runs. Initially compare complete
successes only; any success-versus-censored soft preference is a separately
versioned ninth variant with reduced confidence, never a certified negative.

The policy tournament loss can use the existing bounded full-group log-space
preference implementation, but first-fork/shared-context/full-suffix coefficient
builders need distinct new code and unit tests. A shared-context comparison
requires identical braid word, strand count and used CC/moves; a matching word
alone is not sufficient. The current 248 loss and checkpoints
remain immutable. For H, measure per-task encoder gradient norms on TRAIN-only
batches and pin the coefficient so tournament gradients are comparable to
metric/demo gradients; raw loss values do not set the ratio. Report gradient
cosines/conflicts and exact effective draw counts per source. No single
``policy %`` is meaningful for differently normalized objectives.

## Selection and investment gate

Stage 1: nine short matched CPU training runs (one control plus eight
idea-(3) variants) with identical work quotas,
followed by a fixed small DEV search matrix. Report distinct roots solved,
L10/L1000 separately, CC on common successes, moved/censored roots, network
evaluations, simulations and wall/CPU time. Do not rank by distance loss alone.
Advance at most two candidates on the predeclared DEV primary criterion of
solved roots at matched search compute, with CC and time as explicit tie-breakers.
If small-sample differences are within uncertainty or no candidate improves
the control, report inconclusive rather than promote a winner.

Stage 2: replicate the two finalists and control on disjoint DEV groups/seeds
and an expanded but still bounded CPU/search budget. Only a stable improvement
qualifies one variant for the larger CPU investment and iterative
search->training collection. Freeze the selected actor/checkpoint, search and
comparison contract before collecting a NEW independently excluded TEST panel.
An independent TEST result is needed for a utility claim; comparison with
fully non-neural algorithms on the same starts/resources is needed before any
superiority claim. Uniform tree priority with neural continuations is not that
control. The user's suggested >15% solved is a practical signal for when to
prioritize the large heuristic comparison, not a theorem or a training label.

No cloud purchase, database publication, commit, merge or push is authorized.
Never interrupt a healthy 248 or 139/155 worker just to start this sweep.
