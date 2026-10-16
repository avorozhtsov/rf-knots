# Complete149 evaluation orchestration: PREPARED, not launched

2026-09-24. The previous goal turn completed cache/search/replay workers and
tests; this turn connects them to a full18-job dispatcher and an independently
checked paired summary. This is implementation progress, not new scientific
model/search evidence. No live139/147 process or frozen source was changed.

## Exact matrix and prerequisites

`scripts/run_long_general_evaluation.py` refuses an incomplete149 training pair
and verifies its unchanged frozen147/150 evaluation contract before creating
an output directory. It does not launch, restart, stop, or modify training.
The prescribed18 jobs are:

1. Independent complete149 training-pair audit.
2. Checkpoint-specific cache gate for shared-initial/control-final/long-final.
3. Six distance jobs: those three model states, each on general147 TEST and
   short-u150 original/lifted TEST views.
4. Five fixed151 MCTS configurations, each immediately followed by its full
   successful/censored trajectory and reference-witness replay audit.

Data consumers already passed before training and are hash-bound prerequisites,
not extra post-training acquisition jobs. The dispatcher cannot change seeds,
simulation counts, roots, checkpoints or resource envelopes at runtime.

Actual-command matching checks for an already live worker, including resumed
commands. A matching worker is adopted until it exits; a completed artifact is
verified and reused. Multiple matching workers, partial finite-job outputs,
nonzero child exits or changed evidence stop the controller with preserved
outputs. Search resumes only through its explicit checkpoint protocol. Child
completion checks are internal dispatch, not periodic user status polling.

Inputs, the exact job list and sorted source snapshots are frozen. Resume
rechecks the completed prefix and snapshots; completed re-entry cannot claim
success without all18 jobs and the bound summary. Neither a stale PID nor an
observation timeout is treated as grounds to launch a duplicate.

## Complete comparison, no automatic promotion

`scripts/summarize_long_general_experiment.py` requires all18 outputs in order.
It rechecks child source/input/checkpoint binding and independently recomputes
metric record identity, interval/group/class/constant statistics, positive-only
ranking, length/strand strata and actual149 size-counter reporting. It compares
both finals to the shared initial state and long-final to control-final, both
over all pairs and over certified-positive pairs. Record-weighted and equal-
invariant-group averages remain distinct; neither is a knot-type average.

Five paired search contrasts retain identical roots and caps: each final versus
initial, long-final versus control-final, and distance versus uniform tree
prior for each final. Report each L10/L1000 condition separately, new/lost
successes separately from common-success cost deltas, and all actual recorded
forward/action/time counters. Reference/solution costs are recertified; every
censored path remains bound to its completed replay audit. A budget failure is
not an impossible-state or bad-action label. A lower-bound gap is not a proof
of optimality. No automatic model promotion or public database theorem follows.

## Tests completed

The controller fixtures test completed-output reuse without dispatch, actual-
process adoption until exit, new versus explicit-resume commands, duplicate
rejection, nonzero exits, partial-output preservation and the exact18-job list.
A full dispatch-lifecycle fixture interrupts after three completed jobs, resumes
the remaining15, then proves completed re-entry leaves all files byte-identical;
a forged completed status with17 jobs is rejected. These fixture workers are
explicitly not scientific evaluations.

Metric child tests bind real report arithmetic to frozen fixture records and
reject wrong model/checkpoint/source metadata, positive-subset counts and
training-size claims. Summary tests reject partial matrices, reordered children,
tampered constants/groups/ranking/size fields, and distinguish record-weighted
from equal-group error deltas. The combined147/149/150/152/evaluation/search suite
passed145 tests. Ruff and formatting passed; the only warning was unavailable
optional Plink GUI support.

## Intended command after actual training, not executed

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/run_long_general_evaluation.py \
  --run artifacts/raster-long-general-training-pair-v1 \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-long-general-evaluation-v1
```

Use `--resume` only for that existing controller after inspecting its actual
processes/files. Full147 acquisition/audit,150 acquisition/audit,152 real MPS
readiness and149 training still precede this command. The scheduled combined
139/147 live check remains no earlier than06:32 UTC; no fresh process claim was
made during this implementation turn.
