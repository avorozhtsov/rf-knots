# Protocol97 training advancement and complete evaluation queued

2026-09-21, local observation around12:58 Europe/London. This is execution
evidence, not a completed training or scientific-promotion report.

## Verified training progress

The same Q254 worker PID94033 (parent94005) advanced from16 to128/2048 saved
optimizer updates. The128-update checkpoint was reloaded and independently
hashed:
`7c2d4002f2502f06c4346a99f5216a7ef99891ba4c0952ad52c7311b59980c04`.

Every model/objective/optimizer tensor was finite; all populated Adam counters
were128. Relative-group update counts were[3,1,3,1], covering1330 full-group
states. All original training inputs and67 sources remained unchanged.
Optimizer-loop time was805.27 seconds, excluding preparation. Gradient norm
before clipping was3372.80 at that snapshot; this is a combined multi-task
gradient, not a measured relative-only gradient or proof of learning success.
The random-local arm remained queued for its own2048-update run.

## Complete evaluation controller is actually running

Directory: `artifacts/pretrained-raster-log-evaluation-v1/`.
Controller PID98832, observed live with a fresh durable manifest, status
`waiting_for_training`. Its13-job sequence has0 completed jobs at this snapshot.
It verified the actual canonical training launcher/process, rather than treating
an old PID or an incomplete manifest as a reason to restart training.

Controller contract SHA256:
`faa51a5c2568ab073bf477b6b5f4559cb19bb7a51a70d10b5b5b101d508bf58d`.
All1172 controller input/source pins were rechecked. Sources are now frozen;
do not edit them beneath the queued controller. Protocol99 describes the exact
metric/search/replay matrix and attribution limits.

New implementation files:

- `evaluate_raster_log_search.py`:128 conditions with resumable hashed shards.
- `audit_raster_log_search.py`:full replay and semantic-cost certificates.
- `run_raster_log_evaluation.py`:actual-process training gate, serial jobs,
  duplicate-worker checks, immutable completed-output verification.
- `summarize_raster_log_experiment.py`:final-only metric/search comparison,
  recomputed labels/statistics, explicit censoring and attribution limitations.

Together with earlier new97 modules and cost-certification regressions,80
focused tests passed. New files pass targeted Ruff. This does not claim a
whole-repository test/lint result or a completed128-condition new-model search.

## Parallel earlier test and next required evidence

The earlier70 MCTS initial-model worker PID90564 advanced to48/128 conditions
and continued using CPU. Its controller PID84595 is also live; no duplicate or
restart was introduced. The full earlier70 matrix remains incomplete.

Next: verify continued durable training, both completed2048-step checkpoints,
then let the queued13-job evaluator run and verify its final summary. The new
summary explicitly does not claim a causal embedding ablation: both arms use
embeddings, and the transfer contrast also changes the initial policy reference.
Distance quality, L10/L1000 gains, unseen-knot generalization and arbitrary-size
solution quality remain unproven until the corresponding evidence exists.
