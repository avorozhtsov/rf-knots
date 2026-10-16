# Training149 complete; corrected independent evaluation launched

2026-09-25, 23:44–23:49 UTC (26 September locally). This supersedes the pending
launch status in163. Preserve all original sources and experiment artifacts.

## Completed training

Both arms committed2048 optimizer updates. Long completed its final DEV reports
at23:00:21 UTC; parent completed23:00:23 with phase
`training_completed_not_promoted`. Session98193 returned exit0; no live training
process remained. This is4096 optimizer updates, not4096 MCTS games.

Final checkpoint SHA256:

- control: `89c109f06d19f21f6b8ccd06c2e0e619155e962f26e333446d6d094d97e23766`.
- long: `64fd57deb9ac195fc40db8adbb929df90fde67aa4c2ca3178b405d83aaf741d8`.

Read-only verification session89429 exited0: all1439 input pins, both completed
arm contracts, initial/final checkpoint identities,128 metric-history files per
arm and four DEV report hashes per arm passed. Both final model/objective/AdamW
states are finite; optimizer step counters equal2048. Full paired draw-stream
reconstruction is a separate audit now running, not yet claimed passed.

## Confirmed preflight failure and isolated repair

The original controller's `ready()` failed in read-only session93400, exit1:
`ValueError: 149 training contract differs`. No evaluation output was created.
Schema and config matched. The training contract hash was valid under its
producer's codec, but not under the controller's imported codec:

- training149 uses `raster_log_training.digest` / `raster_pair_training.digest`:
  sorted JSON with default separators;
- controller155 used `build_mcts_training_supervision.digest`:
  sorted JSON with compact separators.

This is an evaluator validation bug, not damaged training. Original controller,
tests, training sources and manifests remain byte-unchanged. Separate
`scripts/run_long_general_evaluation_v2.py` changes only the training-contract
comparison to the producer codec, and additionally pins its own source/tests.
Its own evaluation-contract codec and all18 scientific jobs/conditions remain
unchanged. Separate v2 tests retain lifecycle/duplicate/resume checks and add
readiness acceptance for the producer hash plus rejection of wrong hash,
changed config and incomplete training. Together with summary tests:30 passed,
including rerun after formatting; Ruff passed. The original tests mocked
`ready()` during dispatch, so they had not exercised this hash incompatibility.

## Verified active evaluation155-v2

New output: `artifacts/raster-long-general-evaluation-v2` (v1 remains absent).
Controller PID574/session17201; single training-pair auditor PID639. Initial
live verification: worker99.2%CPU,36.37s CPU in38s elapsed. Manifest is
`evaluating`,0/18 jobs complete, current `training-pair-audit`; source snapshot
and log exist. Log currently has only the known optional Plink tkinter GUI
warning, not an audit failure. No terminal result yet.

Evaluation contract SHA256:
`2b8ed18db84caba1220fde46a96bbdbc88119201b2f0fdf276c44f9e40362b6f`.

Actual launch:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/run_long_general_evaluation_v2.py \
  --run artifacts/raster-long-general-training-pair-v1 \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-long-general-evaluation-v2
```

Next heartbeat must inspect this v2 output and actual processes, not launch the
original v1 controller. Use explicit resume only after a confirmed stop and
inspection of any child process/output. The controller advances through full
training audit, cache parity, six distance scores, five searches and five full
trajectory replay audits automatically; no new scientific settings were chosen.

## Continuing search139

Old controller20100/worker21103 remain live, worker about99%CPU and303m35s CPU.
Committed progress:40→48 at21:59→56/128 at23:27:11 UTC. Seven shard hashes and
roots hash passed; latest `conditions-00048.json` SHA256
`acaa783ba0c056171c5e3102609fea4ffc5b09a5acd94c608d547cb6cbd85014`.
Total336 episodes,188504 policy evaluations,0 solutions. No logged runtime
error found. This is still only the first search configuration; no public-bound
improvement or superiority to fully non-neural heuristics is demonstrated.

Continue hourly observation of both healthy processes. Do not modify frozen
sources, promote from an incomplete comparison, publish databases or commit/push.
