# Group-balanced diverse-metric joint training

2026-09-20. Fixed before either training arm or fresh evaluation-panel generation.
This tests expanding metric support, not a residual adapter or removal of global
embedding. Notes39–41 motivate the change; no completed paired-test scores will
be reused as an independent promotion test.

## Inputs and controlled intervention

Parent remains `embedding-multisize-clean-joint-v1/checkpoint.pt`, SHA
`076a12a86ca83c03d736edcaea09b31234cae329c9d8d5c743e8c8bef2fdc723`.
Use the same fully replay-audited TRAIN rollout data
`embedding-clean-paired-rollouts-v1`, actor SHA
`3b7ee2f4a5e0dfa68382530ad89b256e1103f8c4955bd6d676007ff11d8e2b31`.
No evaluation-only rollout may enter training.

Metric control: old4096-record corpus SHA
`04ffeca1bebccc5e8af6a7f7553b2faa2895e098041320e73bf4a2cd8093600e`.
Metric candidate: new4802-record corpus SHA
`f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14`.
Its344 source candidates passed the full independent audit, SHA
`1b109228d19da4d35315320dabf73b1b8ac848e27cdac7dff256b600aea2e447`.
The706 additions are root anchors, first-CC general pairs, and suffix anchors,
not706 independent knot types. The old4096 TRAIN rows are unchanged.

Both arms:4096 steps, seed20261018, batch64, lr2.5e-5, value BCE weight1,
relative policy weight0.1, clip0.2, metric weight1, full encoder/policy/value
training. Same architecture and initialization; no new neural layers. Same
relative-policy gates32 total/eight original-length>=96, microbatch64, and the
same complete-episode coefficients with exact common-prefix cancellation.
The old parent behaviour value target remains distinct from MCTS success.

Both arms use opt-in `--metric-bounds determinant --metric-sampling group-balanced`.
Replay-verified fingerprints permit the lower bound to be raised to at least1
when endpoint determinants differ. The upper bound is unchanged. This does not
turn every two-CC witness into exact distance2. Original corpus bounds stay
intact, as do old validation diagnostics.

Choose metric samples uniformly by nonempty `(task,bound-class)` stratum, then
query fingerprint group, then record. Task=explicit-unknot/general;
class=exact0/exact1/exact2/interval. Groups are not knot identities. Exactly three
uniform draws per sampled record align the shared Python RNG across arms despite
different numbers of groups. Keep63 such draws plus the fixed exact mirror-pair
distance2 anchor each step. Thus4096 steps means262144 sampled metric examples
per arm, not that many unique roots or RL games. The uniform/signature defaults
remain unchanged for historical callers.

Run candidate first, then control, serially:
`embedding-diverse-metric-joint-v1` and `embedding-diverse-metric-control-v1`.
Use final4096-step weights only. No early stopping, checkpoint selection,
hyperparameter changes, or promotion from training-root diagnostics. Preserve
source snapshots, inputs, exact initial tensors, checkpoints, optimizer/RNG and
actual process/file progress; resume incomplete workers from their saved steps.

## Prospective independent evaluation

Before scoring, construct fresh panels with existing model-blind generators:

- General Gordian:4000 attempts, seed20261019, max seed strands7, max genus10,
  output `embedding-diverse-metric-gordian-test-v1`.
- Short-u:4000 attempts, seed20261020, max strands12, depth8..32,
  output `embedding-diverse-metric-short-u-test-v1`.

Use the existing selection caps and adequacy gates, exhaust both streams, no
replacement seeds or top-ups. In addition to the five base EXCLUSIONS, exclude
the fresh-group, general-Gordian, constructive-short-u, and both clean-paired
panels by both endpoint fingerprint groups. Independently replay every TRAIN
path (old plus new metric and eligible rollout paths) and every new panel
witness; check no query overlaps, including CC intermediate states, and second
determinants. Failure of panel adequacy is reported, not silently repaired.

Evaluate parent/candidate/control on general original bounds, short-u original
and explicitly determinant-strengthened bounds; retain constant0/1/2 controls,
equal-group errors and exact-class errors, and both original/lifted views.
These lifted views are stability controls, not genuinely harder long-knot
transfer: new training already reaches208 letters/17 strands.

Evaluate all original short-u roots with the existing anytime embedding cost
search,128 expansions, both λ10/1000, witness-feasible caps, layout tie-break.
Include the three models and length heuristic; replay all successful witnesses.
Compare candidate vs parent, control vs parent, candidate vs control and vs
length. Report coverage, new/lost/common solutions, common-only cost differences,
neural work and time. No infinite costs or fabricated costs for unsolved cases.

Do NOT repeat the zero-floor512-simulation MCTS matrix as if it were a new
informative RL test. This iteration measures embedding-driven anytime search;
a cost-aware/receding-horizon MCTS training contract remains a separate next
requirement. No MCTS-benefit or arbitrary-size-quality claim follows here.

Promotion requires a useful short metric beyond trivial constants and the
matched control, without hiding general-pair degradation, plus better replayed
L10/L1000 outcomes with coverage considered. Ambiguous/negative results are
retained. Fingerprint groups, cost conditions and views are correlated; do not
claim statistical certainty from repeated representations or a few lucky solves.
