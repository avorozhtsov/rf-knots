# Diverse-metric iteration: execution and limitations

2026-09-20. Living execution note; protocols40–42 remain unchanged.

## Completed data work

- The fixed1024-attempt supply completed:344 candidates,300 fingerprint groups,
  163 exact-u1,53 exact-u2,128 intervals[1,2]. These classifications explicitly
  use determinant-strengthened bounds. Fingerprints are not knot identities.
- Independent audit passed for ALL344 candidates: full witness replay, CC-path
  ownership, old validation/current paired-query exclusion, reference replay
  normalization, and a second exact determinant algorithm. Audit SHA:
  `1b109228d19da4d35315320dabf73b1b8ac848e27cdac7dff256b600aea2e447`.
- The first, read-only audit process45797 was deliberately stopped (exit143)
  after32 roots because it reconstructed whole Alexander polynomials. No data
  or training was interrupted. The exact Burau-cofactor implementation passed
  comparisons with the old polynomial implementation; the replacement audit
  completed with exit0. This is an arithmetic implementation change, not a
  change in accepted candidates or bounds. See note41 for the mathematical source.
- Conversion added all706 proposed ordered pairs, with zero duplicate raw
  endpoint pairs:344 root anchors and362 first-CC/suffix pairs. Final corpus:
  4802 rows,474 query fingerprint groups, raw length<=208 and strands<=17.
  SHA `f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14`.
  Direct JSON-object comparison confirmed the original4096 TRAIN rows and both
  old holdout lists unchanged. Old `size_shift` views reach only45 letters/8
  strands and are NOT larger than this training corpus.

## Training implementation and launch

`train_rollout_value.py` has explicit opt-in metric-bound and sampling options;
old defaults remain signature/uniform. `balanced_metric_sampling.py` changes
TRAIN rows only, preserves interval upper bounds, and uses exactly three random
draws per balanced metric sample. Tests check non-exact bounds, group rather than
record weighting, aligned RNG consumption/resume and rejection of heldout rows.
Sampler plus existing rollout gradient/relative-return tests:20 passed.
Independent determinant/supply/conversion tests also passed.

Candidate launched exactly as protocol42, PID50562, unified session34386,
output `artifacts/embedding-diverse-metric-joint-v1`. Observed real progress:
512 ->3584 ->4096 replayed episodes, then132 prepared relative-policy groups,
then metric-corpus preparation, followed by actual optimizer steps112 ->192
->288 ->320. At step320, independently checked all pinned source/input bytes,
initial SHA `b2cd2502dace3b8a0daddcc7424151c6ef87578f63e3394467aa437f1c6e64c3`,
and all337 model/objective/optimizer tensors finite. The sampled step320
checkpoint SHA was `d739e52660e62130c87df28e9ddecfec0191db59cdfd50e710baefe8449858db`;
it is an INTERMEDIATE checkpoint, not a selected/promoted final model. CPU and
durable checkpoints/manifests advance. No control training or final model
evaluation has started yet. Do not edit its
pinned training scripts/helper or any `src/rf_knots/*.py` until this run completes.

After the candidate, run the old-corpus control with the same4096 steps, seed,
metric modes and rollout data into `embedding-diverse-metric-control-v1`.
Check identical initial tensors/configuration except corpus/output and matched
Python sampling RNG progression. Do not substitute either previous value/control
checkpoint for this new matched control.

## Fresh evaluation generation: an explicit adequacy failure

Both fixed streams were launched without scoring a model, with all protocol42
exclusion panels. Sessions26578 (general) and92420 (short-u).

The general generator exhausted4000 attempts and exited normally, but its panel
is **INADEQUATE**:15 rows in11 query groups, exact0/1/2 counts6/4/1 and four
intervals. `panel.json.complete` is false. Construction completion is not passing
scientific adequacy. The one exact2 example and15 total rows do not satisfy the
declared gates. Do not score this as the predeclared independent general test,
top it up, choose another seed, or silently relax the gates.

The short-u stream exhausted4000 attempts with32 one-CC witnesses and32 two-CC
witnesses in56 query groups. Final panel writing completed; both `complete` and
`construction_complete` are true. The full metric-path audit passed for all4802
TRAIN rows,495 path groups, and56 fresh query groups; all queries are disjoint.
`metric-independence-audit.json` SHA:
`db252d9f10da6ce84bd2d1e923a266ccc376f7b4b2d6a5f83d0082a846fb312f`.
The rollout-path certificate extension also passed:56 query groups are disjoint
from82 optimizer-eligible rollout-path groups. It verifies every byte-pinned
input of the previous complete4096-episode replay audit and reuses its complete
path-group set; it does NOT claim to replay those unchanged episodes again.
`rollout-independence-audit.json` SHA:
`efd62d515e14634306313b0aba0280cbe00a094d2d3cbfa840366a5cfaaf1a5f`.
Both files are in `embedding-diverse-metric-short-u-test-v1`.
Witness classes remain upper bounds, not automatically exact-u classes.

The failed general panel remains an explicit missing evaluation requirement.
A new score-blind benchmark family would need a separately declared protocol
and a disclosed change of scope; it cannot retroactively make this failed
panel adequate. No automatic model promotion is allowed while the required
independent evidence is absent. Training may continue without claiming the
overall embedding/search objective has been achieved.

## Explicit supplementary general-pair design

Protocol44 separately declares a common-positive-block family, without modifying
protocol42 or the failed general panel. Both pair endpoints must have actual
signature lower bound u>=3 while prefix distance has upper0/1/2. This tests
general short distances between nontrivial higher-u knots, not another set of
unknot anchors. Full replay/label and adequacy tests:11 passed.

Generator PID55305/session18211, output
`embedding-common-block-gordian-supplement-v1`. Its fixed1024-attempt stream
completed:80 rows in59 query groups, classes exact0=28, exact1=19, exact2=5,
interval=28. These counts satisfy protocol44's minimums; final lifted-view
writing completed, with panel SHA
`c4ed9b27cc2e8ca04853b772882636f829864306e7f25db08c4eddd4e5f0748a`.
The full metric audit completed with exit0:59 query groups disjoint from495
TRAIN path groups, all4802 TRAIN records replayed. Audit SHA
`0e633ebc4c5a838119fb2ebd0ebf1e0388c8d073b742f6ca51a1781a61071ffd`.
The rollout certificate extension also passed, reusing the byte-verified
complete prior episode audit:59 queries disjoint from82 eligible rollout-path
groups. Extension SHA
`9a909068a2165c5b1f1ca46db759337a650dd9cea7c76302e5fc4c4db66fa5c1`.
No model has been scored on the supplement. Five exact2 observations remain a
small sample, and shared structure limits generalization claims.

## Verified serial continuation

Protocol45's controller passed29 tests together with the earlier execution
guards and was launched as PID61409/session54467 into
`embedding-diverse-metric-sequence-v1`. Its actual manifest adopted the live
candidate PID50562, without launching a second worker. Candidate CPU time
advanced from32:04 to33:19 and optimizer steps976 ->1024; the controller's
observed step advanced from976 to1008 at its separate polling cadence.
Status remains `waiting_candidate`, no error. Control and final evaluation
have NOT started. Both panels now have the required two passing certificates.

After candidate completion it will train the declared matched control, then
run17 final-only evaluation/comparison jobs. No intermediate model selection,
no new MCTS matrix and no automatic scientific promotion. All pinned sources
and protocols remain frozen while this sequence runs.

The user's arbitrary-L/N architectural goal is clarified in note46. A fresh
run of encoder, site-proposal and progressive-search tests passed30 tests.
These are implementation checks, not evidence of learned long-knot quality.

## Final-only report consumer prepared while training continues

`summarize_diverse_metric_sequence.py` is a separate read-only consumer; no
pinned training/controller/evaluator source was changed. It requires the exact
17-job completed sequence, matching output/checkpoint/source/audit hashes,
complete query coverage, original panel labels and comparison source reports.
It recomputes metric errors and constant0/1/2 controls with the SAME equal-group
weighting; search coverage/new/lost/common-only costs remain separate.
The protocol44 amendment, failed original general panel, lifted-view limitations
and absence of MCTS/arbitrary-size-quality evidence are explicit in the output.
No model is automatically selected.

Consumer plus existing summary tests:29 passed, including changed labels,
checkpoint/command/order/source mismatches, missing queries and censored costs.
An actual invocation on the still-running controller failed intentionally at
the final-only gate and created no `embedding-diverse-metric-summary-v1` output.
This is an expected guard test, not a training failure or completed result.
At this update candidate PID50562 advanced to1248/4096 optimizer steps; CPU time
advanced to38:35, while controller PID61409 remained live in `waiting_candidate`.

After the sequence completes, run:

```sh
.venv/bin/python scripts/summarize_diverse_metric_sequence.py \
  --manifest artifacts/embedding-diverse-metric-sequence-v1/manifest.json \
  --out artifacts/embedding-diverse-metric-summary-v1
```

Then interpret the paired evidence; summary completion is not goal completion.

## Cross-module verification while the candidate remains active

The combined embedding, strand-graph, rollout, MCTS, balanced-sampling,
diverse-data/summary/controller, common-block, determinant, clean-paired,
site-proposal, lazy-action and sparse-witness test selection passed278 tests
in6.58s. No pinned code was changed by this verification.

A read-only snapshot of the step1344 checkpoint contained337 finite
model/objective/optimizer tensors; all pinned training source/input hashes
matched. Snapshot SHA
`3b626977d05dd7915960a5dc371745097d194ab9ea3c27632a3534b66f96357b`.
This is an intermediate integrity check, not model evaluation or selection.
Subsequent live logs advanced through1680/4096. Control remains unstarted,
and the serial controller remains `waiting_candidate` without error.

2026-09-21 continuation: candidate passed halfway, reaching2064/4096 steps.
An atomic read of that checkpoint had SHA
`a05ea585211cc374eeaefb869c07453882b9010718b0ba9a564c1a8bb5e23446`;
all337 model/objective/optimizer tensors were finite and every pinned training
source/input hash still matched. PID50562 CPU advanced to56:35; controller
PID61409 remained live. The intermediate weights were not scored or selected,
and the regular checkpoint file continues to be updated by the trainer.

After three quarters of the same run, a step3104 snapshot had SHA
`5b166dfce5874a80ee821ecf2d95d7df61c40576208ad3c6f0aacd6620577466`.
All337 model/objective/optimizer tensors were finite; pinned sources and inputs
matched. PID50562 CPU advanced to79:24, and controller PID61409 observed3104
with no error. No final evaluation or control training had started.

## Candidate completed; matched control verified active

2026-09-21. Candidate PID50562/session34386 completed4096/4096 steps and
exited0. Its final manifest is `completed`, with checkpoint SHA
`964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e`.
Independently verified the declared final configuration, all source/input
hashes, common initial SHA, step count and337 finite model/objective/optimizer
tensors. The final diagnostic `after.json` SHA is
`24638466546894873a24f69c185eefafbf346099e5c8d205a448aff85a61519b`.
Optimizer-loop elapsed time was5876.036s, excluding preparation/final diagnostics.
This completion is NOT model promotion or the final independent comparison.

Controller PID61409/session54467 transitioned to `waiting_control` and launched
one child PID81643 with the exact protocol42 old-corpus control command. No
candidate remains live and no duplicate control was launched. The child's
CPU time advanced from0:15 to1:20, while durable preparation/log records advanced
from1536 to4096 replayed episodes and then `building_relative_groups`.
Control preparation is VERIFIED ACTIVE; optimizer steps and matching initial
tensors are not yet observed for this arm. Its durable log is
`artifacts/embedding-diverse-metric-sequence-v1/logs/train-control.log`.
No final evaluation jobs have started. Keep all sequence dependencies frozen.

Control preparation subsequently completed and optimizer checkpoints advanced
64 ->96 ->112. Its CPU time advanced to5:06. Independent read-only comparison
confirmed byte-identical initialization SHA
`b2cd2502dace3b8a0daddcc7424151c6ef87578f63e3394467aa437f1c6e64c3`,
identical configuration except corpus/output, shared source/input hashes,
architecture,130260 value rows,132 relative groups and23 long relative groups.
All337 tensors in the sampled step96 control checkpoint were finite. Thus
control optimization is VERIFIED ACTIVE, not merely queued/preparing. Final
RNG alignment and paired final model checks remain pending until4096 steps.

## Both training arms completed; independent evaluation started

2026-09-21. Control PID81643 finished4096 steps, wrote final diagnostics and
is no longer live. Its manifest is `completed`; optimizer-loop elapsed time
was2777.598s. Final checkpoint SHA:
`30a664be4d2bc85cbd519fa66adf5adb0c843c56ea59ec2ba8c85c7e5a56e426`.
Final `after.json` SHA:
`0ab755f90ef998a1776e2ef1eede67aad787f78e968205e29321b918e54b04ae`.

Independently verified BOTH final4096 checkpoints and337 finite tensors per
arm, their checkpoint/initial hashes, current pinned source/input bytes,
matching configurations except corpus/output, shared supervision/architecture,
and identical final Python and Torch RNG states. Candidate checkpoint SHA
remains `964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e`.

Controller PID61409 transitioned to `evaluating` without error and launched
the first prescribed job, `metric-parent-supplement-original`, as PID96521.
At this transition0/17 jobs were completed. No promotion or held-out benefit
is established by training completion. Preserve the failed original general
panel and supplemental-family limitations in the eventual summary, and keep
all pinned evaluation dependencies frozen until the sequence finishes.

## All nine distance evaluations completed; cost search active

2026-09-21. Controller completed9/17 jobs, with all nine report hashes
independently matched to its manifest. Equal-fingerprint-group mean interval
violation on the original views (lower is better):

| Panel / bounds | Parent | Candidate | Control | Constant1 |
| --- | ---: | ---: | ---: | ---: |
| Common-block supplement / original | 0.515422 | 0.388780 | 0.516355 | 0.343503 |
| Short-u / original | 0.349564 | 0.189558 | 0.387274 | 0.160714 |
| Short-u / determinant | 0.349564 | 0.201975 | 0.387274 | 0.160714 |

These are descriptive held-out results, not significance or promotion claims.
The candidate improves these averages versus both learned comparators but
does not beat constant1. Improvements are not uniform: short-u exact2 row MAE
is0.561103 for the candidate versus0.379783 parent and0.225520 control.
The common-block panel remains the explicit supplementary family, not a
replacement success for the inadequate original general panel. Lifted views
do not establish generalization to intrinsically harder knots.

Controller PID61409 launched `cost-parent` as PID3080; its live command uses
all64 short-u queries,128 expansions, witness caps and layout tiebreak under
the frozen protocol. The subsequent candidate/control/length runs and paired
comparisons are still pending. No L10/L1000 search benefit is established yet.

## Sequence and audited summary completed

2026-09-21. All17 jobs completed; controller and evaluation processes exited.
The final-only summary consumer succeeded after verifying the complete job
matrix, source/input/output/model hashes, certificates and report semantics.
Summary JSON SHA:
`3dea9f4da75a88d0c347a8a49147fd1fe07c4d42b78946d73d396916b84ec260`.
Report: `artifacts/embedding-diverse-metric-summary-v1/report.md`.
Interpretation and next direction: `research/47-diverse-metric-results-and-next-step.md`.

Replay-audited search success is34/64 parent,35/64 control,57/64 candidate,
2/64 length at each CC weight. Candidate adds23/22 successes over parent/control
without lost solutions, but all common learned-model costs are unchanged.
This establishes a bounded direct-search benefit, not a calibrated general
distance metric, MCTS/RL benefit, or arbitrary-size solving quality. The full
goal remains active. No weights, protocols or existing evaluation outputs were
changed to obtain this conclusion.

All quoted process IDs are launch records. Recheck actual command lines, CPU,
live sessions and durable file advancement before using them as current status.
