# Independent149 distance evaluation: implemented, not scored

2026-09-24. Continues the active embedding goal without changing live139 search,
live147 acquisition, frozen historical experiments, or public UnknotDB. The
previous goal turn made progress: it completed training-audit implementation
and tests and verified actual139/147 advancement. This turn adds the scientific
distance consumer needed after149 training; it does not claim new model quality.

## Implemented input and checkpoint contract

`scripts/long_general_evaluation_inputs.py` requires a completely committed149
pair and its independent training audit, both full2048 endpoints and all129
committed checkpoints per arm, exact source/manifest/checkpoint hashes, and the
unchanged evaluation matrix. It checks both initial and final entries before
loading one model, so a successful single arm cannot hide an incomplete pair.

Both arms use the same raw resource-feature implementation, not135's units
intervention. Initial-state sharing requires exact architecture/model/objective
identity plus CPU encoder and head-position-aware policy forward equality on
the first147 TRAIN endpoint. This is an engineering spot-check backed by exact
architecture and tensor identity, not a proof of model quality on every shape.

General147 TEST consumes the complete passed mathematical audit and rechecks
the query ownership, all known TRAIN-path/DEV/historical exclusions, redundant
pair fields and canonical-pair uniqueness. Short-u150 consumes the complete
passed audit and all64 original roots plus64 corresponding lifts. No metadata
filter silently reduces either denominator. Query invariant buckets are not
identified knot types; Q254 pretraining independence remains unproved.

## Scoring

`scripts/evaluate_long_general_distances.py` uses CPUfloat32, one thread and
microbatch1. It reuses the unchanged97 inference/interval calculations, checks
unchanged weights, records exact inputs, checkpoints, Torch version and timing,
and never runs an optimizer. Only the registered shared initial/control-final/
long-final choices are scored, each on both panels. There is no intermediate
checkpoint selection.

Preserve raw predictions, certified intervals, exact-class and invariant-group
errors, constants0/1/2, and ranking among disjoint certified intervals. Additional
reports separate certified-positive pairs (lower>=1, not merely upper>0),
maximum endpoint length bands0..48/49..96/97..192/193..384/385+, and maximum
strand bands1..12/13..24/25..33/34+. Each nonempty stratum has its own constant1
comparison and ranking. Empty positive cohorts produce explicit empty/undefined
statistics, not a division error or a fabricated success. Pair comparisons are
dependent, not independent statistical trials.

Report observed size counters from actual149 optimization separately for both
arms. These counters do not describe every unused historical TRAIN candidate
or Q254 pretraining, so the385/34 lifts must not automatically be called beyond
all prior training. Better distance statistics alone are not MCTS utility.

## Verification completed this turn

-18 new tests passed, covering checkpoint stage/coverage/hash/path guards,
  incomplete-parent refusal, TRAIN/DEV/query exclusions, duplicate/field
  rejection, exact initial tensors and position-aware forwards, positive-only
  ranking, interval uncertainty, length/strand boundaries and complete counts.
-58 combined evaluation/training-data/auditor/runner/historical-distance/input
  tests passed. Ruff and formatting passed for the new files.
-A finite CPU smoke (session9887,exit0) used the real shared Q254-local raster
  initialization, not the toy network. It verified1148 input/source pins,
  copied the exact initialized state into the comparison model, and ran the
  initial-identity function on the TRAIN trefoil and a fixed already committed
  long TRAIN endpoint (147 attempt8, first state, length288,strands3).
-Embedding and policy outputs were bitwise identical for the two copies at
  heads0/1/2 and0/144/287 respectively. Model/objective state stayed unchanged:
  `c0ca354109be70f0973cff7a8b42a789fd0b3feaa0aeb99941d3e30fe2e78044`.
  The checked loader source SHA256 was
  `851eb5cd00af889f7036455004e734441e9626920868a8b60c3e58723fe552f3`.
  The immutable attempt shard SHA256 was
  `e34bcb4ecaccba78241a4df2e6c41e8f32e77f758913b2727000fc1f66f0f7a2`.

The smoke tested real architecture plumbing with equal weights. It did not
produce149 trained endpoints, a152 MPS gate pass, held-out quality scores or a
search result. No source or artifact bound by an active job was edited.

## Intended execution after prerequisites, not launched

After147 completion and its ready consumer,150 completion/consumer,152 real
MPS gate,149 training and its independent audit, run the registered six distance
jobs with new output files. Example for ONE of those jobs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/evaluate_long_general_distances.py \
  --run artifacts/raster-long-general-training-pair-v1 \
  --pair-audit artifacts/raster-long-general-training-pair-v1/training-pair-audit.json \
  --arm long --stage final --panel general \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-long-general-evaluation-v1/metric-long-final-general.json
```

The149 checkpoint-specific cache gate, five-search orchestration, complete
trajectory replays and final paired report remain to be implemented and run.
Existing139 search continues unchanged. The next routine combined139/147
process/file observation remains no earlier than06:32 UTC (hourly cadence).
