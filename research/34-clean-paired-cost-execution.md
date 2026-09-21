# Clean paired-cost iteration: execution record

2026-09-20. Frozen protocol is note33; do not edit its pinned file while any
associated builder/auditor is running. This note records execution, not a
revision of the scientific gates. The overall goal is still open.

## Completed behaviour collection

`artifacts/embedding-clean-paired-rollouts-v1/` completed all512 conditions,
4096 episodes and230044 transitions in614.64 recorded seconds after preparation.
There are1024 successes. Training eligibility holds for3209 episodes, including
950 successes;887 excluded episodes remain archived. This aggregate includes
diagnostic seeds6/7, so it is NOT the number of optimizer training examples.
Transitions are visits, not distinct representations or independent games.

The256 actual saved base roots were rechecked: their paired10/1000 conditions
have identical words, strands, resource caps and witnesses, and different UIDs.
All rollout shards completed normally and the source hash gate passed. Parent
actor weights are unchanged. Provenance hashes:

- Final manifest: `4c7bfe3ada74c3504584f0c4da7ea3b47c1556f571b920fe35ddcd5e84f55c19`.
- Actor: `3b7ee2f4a5e0dfa68382530ad89b256e1103f8c4955bd6d676007ff11d8e2b31`.
- Paired roots: `b9d8b8d84ec8a745e6bf73c5c0c520d77024384370a40b9dc3461051ff22e6df`.

The source collection process exited successfully. Historical PID81169 and
session86362 are NOT evidence of any subsequent live work.

## Fresh evaluation inputs, not model scores

Both fixed4000-attempt generators completed; neither stopped early or changed
its seed/gates. No checkpoint has been scored on these panels yet.

- General Gordian:66 pairs in24 new query groups,29 exact0,14 exact1,6 exact2,
  17 intervals. Panel SHA
  `5b772a235eccb7ecb5d21bd43f8fcdc144031aaea3985a1352c4780be2604a2a`.
  Independent audit passed:4096 metric-train paths,123 intermediate groups,
  no intersection with queries or the excluded historical panels, independent
  determinants checked. Audit SHA
  `3d6273be72c7023b42bde468e486c2f9ab23ac6c41e3362b9a585e8f5554b922`.
- Short-u:64 roots in55 new query groups,32 one-CC and32 two-CC witnesses.
  Panel SHA `c43912a65030e9734a27d5261fa07acb007fca54f6a2a822405176386ffad8ff`.
  Adequacy and independent audit passed:4096 metric-train records,123 train-path
  groups,55 independently checked query determinants, no query/train/excluded
  intersections. Audit SHA
  `c8d61fd3f133b774136efcef429950b9c1a4153a99828fcf79bdcff9d60e4e83`.
  The audit process exited normally; the exact polynomial checks at N12 were
  slow, not stuck, and were allowed to finish without a duplicate restart.

Both have the predeclared lifted stability views. Synthetic distribution,
fingerprint collisions and correlations across views/conditions remain caveats.
Test-owner groups cannot enter optimizer data through eligible rollout CC
endpoints. Do not confuse a generated adequate panel with a completed audit.

## Training implementation and launch

The training objective is unchanged by policy micro-batching: sum the already
prefix-cancelled relative terms across chunks of64, backpropagate each chunk,
then clip the combined gradient once and perform one optimizer step. There is
no per-route or per-chunk length normalization. This bounds activation memory
for long trajectory groups. Verify finite gradients before the optimizer step.

The implementation now rejects missing/duplicate episode seeds/groups, enforces
the declared total/long-root preference gates, stores exact initial tensors and
their SHA, and writes durable preparation progress. An inadequate dataset is
recorded as insufficient supervision, not silently trained or relabelled.
88 focused tests passed: root pairing/selection, complete-episode semantics,
prefix cancellation, gradient accumulation, constructor replay and scalable
encoder/search regressions. Chunked gradients match a double-precision control
to1e-14 absolute tolerance; real float32 batch-shape roundoff is tested with
1e-5 absolute tolerance, not claimed bitwise identical.

The sequential value/control launcher is live in session66118 (historical shell
PID86583, first child86584). The first arm is **VERIFIED ACTIVE**, observed at
192/2048 updates after all4096 episodes were replayed. CPU time advanced,
checkpoints advanced48 ->112 ->192, all saved model/objective tensors were finite
and shared encoder tensors actually changed. Preparation found130260 eligible
pre-action value rows,132 relative groups and23 length>=96 groups, passing both
fixed gates. These rows are correlated visits, not130260 distinct knot types.
The control is
**QUEUED**, not independently launched. One failed gate stops the sequence.
Both commands use2048 steps, seed20261013, policy micro-batch64, minimum32
relative groups and minimum8 length>=96 groups; only value weight1/0 and output
directory differ. No new head or residual adapter is introduced.

Initial checkpoint SHA is
`b2cd2502dace3b8a0daddcc7424151c6ef87578f63e3394467aa437f1c6e64c3`;
all model and objective tensors were checked equal to the pinned behaviour actor.
No actual validation-driven checkpoint selection occurred. The untrained parent
value head is poorly calibrated on same-root diagnostic seeds: Brier0.62064
versus0.19530 for the fixed training-root success rate0.29096. Raw and log resource
features include CC weight; initial value was never trained in note27. Preserve
the planned constant-leaf controls, rather than claiming progress merely by
beating this untrained head. High initial BCE is recorded, not grounds for
silently rescaling inputs or changing the frozen experiment.

The separate accepted-rollout partition audit completed successfully in
session72612 (retired PID91791):4096 episodes/230044 transitions replayed,
2423 optimizer-eligible episodes with705 successes,2790 eligible training CC
endpoints recomputed,82 encountered training groups. Their intersection with
the79 combined prospective query groups is empty. Report:
`embedding-clean-paired-rollouts-v1/training-partition-audit.json`, SHA
`397bcb324f348f4a197c4067a37cb251fb75fa84c2548cc8f492ab15fc820a95`.
All report input hashes were rechecked after completion.

Preserve source files while the training sequence runs. The first arm subsequently
advanced to208/2048; the control has not started. Next: finish both training arms,
verify final SHA/RNG/initial states and execute the fixed prospective evaluation.
No independent improvement or GPU promotion is claimed here.

## Evaluation controller launched

Subsequent live checks confirmed first-arm progress608 ->752 ->864/2048 and
advancing CPU time on worker86584. The second arm is still queued. These are
dated execution observations, not a permanent assertion that those PIDs live.

The42-job evaluation controller was launched in session13956, PID99732, output
`embedding-clean-paired-evaluation-v1/`. Its durable manifest reports
`waiting_for_training`, has no completed evaluations and no error, and records
the live first-arm step count. It waits for BOTH final checkpoints before any
new-panel model scoring; no duplicate training/evaluation was started. The
exact execution matrix and previously unspecified parent-portfolio seed are
frozen in note35. The controller and paired-comparison tests passed37/37;
all seven invoked evaluator/comparator command interfaces imported successfully.

Controller source/input hashes and completed output prefixes are guarded. A
resume cannot replace already pinned final checkpoints, silently restart a lost
partial evaluator or overwrite an unexpected existing result. Matching live
children are adopted instead. Training sources and parameters remain unchanged.

The next continuation verified worker86584 and controller99732 alive, CPU
time advancing and first-arm checkpoints reaching1072/2048. All85 saved
model/objective tensors were finite, and every declared training source and
input still matched its recorded hash. The controller was still waiting with
zero scored jobs; the control arm had not started. This is an active wait, not
a stalled-process inference or an independent performance result.

An independent read-only final consumer is now prepared:
`scripts/summarize_clean_paired_evaluation.py --manifest
artifacts/embedding-clean-paired-evaluation-v1/manifest.json --out
artifacts/embedding-clean-paired-summary-v1`. Run it only after the controller
completes. It verifies all42 ordered outputs and source/input hashes, checks
current model hashes against final pins, recomputes interval errors, keeps
original/lifted views and bounds separate, and reports new/lost/common solves
without imputing costs to censored failures. It does not select or promote a
checkpoint. Its ten tests plus37 controller/comparator tests passed. A live
check confirmed it rejects the currently incomplete evaluation without writing
a misleading partial scientific summary. The frozen controller/training
sources were not modified by this addition.

## First arm completed; control preparation verified active

The value-weight1 arm completed all2048 updates and exited normally. Its final
checkpoint SHA is
`e0d022b681577aa55d6dfdc2a5d974a86a35a771656f6d7402149d054502df4e`.
The exact initial SHA remains
`b2cd2502dace3b8a0daddcc7424151c6ef87578f63e3394467aa437f1c6e64c3`.
An independent read of the final checkpoint verified2048 steps, all337 combined
model/objective/optimizer tensors finite, and every recorded training input and
source hash unchanged. The final diagnostic `after.json` SHA is
`50a1f3fc2fec0925b0be905bd1a77dd1603dbcb22588b511197c67c52a425a33`.

Same-training-root diagnostic seeds6/7 give value Brier0.1414316947 versus the
fixed training-root-rate control0.1953020255 (1024 trials,271 successes).
This is better within-root probability prediction for the pinned parent
behaviour, NOT a knot-disjoint result, an MCTS-success estimate or evidence of
improved L10/L1000. No new-panel scoring or checkpoint selection has occurred.

The original serial launcher86583 started control worker7964 automatically.
Its command has the same data, seed20261013,2048-step budget and micro-batch64,
with value weight0 and the control output path. Repeated live checks observed
CPU time11.16 ->43.09 seconds and preparation replay512 ->1536/4096 episodes,
with a fresh `preparation.json`. Thus control PREPARATION is VERIFIED ACTIVE;
optimizer updates are not claimed yet. Evaluation controller99732 remains live
and `waiting_for_training`, with the first arm recorded completed. These PIDs
are historical discovery hints; recheck processes and files on continuation.

Control subsequently finished preparation and entered real optimizer updates,
with live checkpoints48 ->80/2048 and advancing CPU time on7964. A fresh check
verified that both arms have identical initial-file SHA, architecture, sources,
inputs, all remaining configuration fields,132 relative groups,23 long groups
and value-row counts. Their complete `before.json` diagnostics are identical.
Only the output directory and declared value BCE coefficient1/0 differ.
All337 model/objective/optimizer tensors in the control checkpoint at80 steps
were finite; every frozen source/input hash still matched. Controller99732
recorded first-arm completion and second-arm training, with no error. Large
logged BCE in control is a diagnostic for its unsupervised value head: its
coefficient is zero, not a reason to change the predeclared protocol.

## Both training arms completed; independent evaluation started

The control arm completed2048/2048 and the serial training session66118 exited
with code0. Final control checkpoint SHA:
`0dc5ec29b5ba40979ae0b04b509cb0e9cc8ff758421d02c5ea4069e88a2936d6`.
Control `after.json` SHA:
`321ced08455f08ea646bb0c928b3ffc1a559375a09ddce5a6d2006cae3186b98`.
First-arm checkpoint remains
`e0d022b681577aa55d6dfdc2a5d974a86a35a771656f6d7402149d054502df4e`.

An independent final audit (not merely trusting controller status) verified:
both saved step counts2048; all337 model/objective/optimizer tensors finite in
each arm; exact initial hashes identical; configurations match except output
and value coefficient; architecture/input/source hashes unchanged and equal;
Python sampling RNG states and Torch RNG tensors equal at the final step.
The deterministic common sampling implementation and pinned data were retained
throughout. Training workers have exited; their old PIDs are not live evidence.

Controller99732/session13956 passed its gates and moved to `evaluating`, pinning
the parent and both final checkpoints. Four metric jobs were observed completed
and hash-recorded; a real child18980 was evaluating value/short-u/original, with
CPU time advancing. No training was duplicated or restarted. These are
intermediate execution observations; all42 jobs and the final read-only summary
are still required. No independent search improvement or promotion is claimed.

The distance stage subsequently completed all9 jobs; descriptive results and
explicitly post-hoc error-direction diagnostics are in note36. The frozen
evaluation then completed `cost-parent` (job10/42), report SHA
`e4ef270e74475fc56213b02e812672cd834ee71e9aed6ae754c2ee08622f5ffa`.
An independent full replay audit verified exactly64 roots at each of CC weights
10/1000, successful witnesses, resource caps, recorded costs and censored failures.
Parent solved34/64 in each condition, using36179 neural-state evaluations per
condition, with recorded search times303.052722s and300.139072s respectively.
The two conditions are correlated, not128 independent knot tests. This baseline
alone establishes no improvement from the new training.

The controller started `cost-value` automatically (observed worker23059), with
advancing CPU time and durable rows7 ->13/128. No duplicate evaluator was
started and the protocol was not changed after inspecting distance scores.
