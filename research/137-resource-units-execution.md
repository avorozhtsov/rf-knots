# Protocol135/136 execution handoff

2026-09-23. This is a live execution record, not a completed result.

## Implemented and checked

New resource-conditioning module, paired MPS training runner and unit tests.
Neither frozen97 sources nor old artifacts were edited. Resource units change
policy context only; initial encoder weights/embeddings remain identical and
all parameters remain trainable. Five new tests cover feature values, flags,
missingness, unbounded monotonicity, invalid resources, isolated exact tensor
copy, and checkpoint/feature contract rejection. Together with the existing
log-training, trajectory-preference and raster-policy tests:26 passed. Ruff
passes for the new files. PyTorch2.13.0 reports local MPS available.

## Launched, not yet optimizer-verified

Command:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u scripts/train_raster_resource_pair.py \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-resource-units-training-pair-v1
```

Unified execution session23131, PID53207. At the last check its elapsed time
was3m32s, accumulated CPU3m29.87s, CPU100%, stateR. Root manifest is
`preparing/verify_data`, with hashed input/source contract and source snapshot.
No arm directory or optimizer checkpoint existed yet. This stage replays the
4802 TRAIN records plus held-out records; do NOT call it trained or stalled merely
because the preparation manifest is unchanged. Inspect this exact process and
the manifest before considering resume; no duplicate runner is authorized.

Next stages are DEV query-partition verification, audited sampler loading,
initial policy references/DEV, then2048 updates per arm. Initial launch verification
is not finished until an actual optimizer checkpoint advances. Explicit resume
is supported; source/config changes must fail closed, not silently relaunch.

Subsequent live file advancement: the manifest now records256 DEV rows across25
query invariant groups, with zero overlap against the audited RF optimizer
lineage. Preparation continues into sampler loading; no optimizer step is yet
claimed. This verifies new data-partition evidence, not Q254 exposure absence.

## Fresh test generation is verified active

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/build_resource_short_u_test.py \
  --out artifacts/resource-units-fresh-short-u-test-v1
```

Session46692, wrapper PID53939, actual generator child PID54000. Wrapper has an
exclusive lock and is waiting for its child, not stuck. Child wasR at99.6% CPU,
with55.84s accumulated CPU. `generation.log` advanced from0 to100 attempts:
88 other partition,6 previous group,5 determinant1,1 accepted (witnessCC2).
The pinned registry excludes1838 TRAIN/historical/DEV query groups. Final adequacy
and full replay audit are pending; this is not yet an evaluation-ready panel.
The log subsequently advanced to200 attempts and4 accepted roots, two in each
witnessCC class, across4 groups.

Both processes only emitted the known missing-tkinter/Plink GUI warning so far;
there is no observed traceback. No prior process was killed or restarted.

## Still required

- Observe actual training advancement; complete and verify both paired arms.
- Complete136 generation/replay/exclusion audit and require the declared64 roots.
- Build the new general-distance panel using the already reserved seed20261073,
  with declared fixed generation budget and independently checked intervals.
- Evaluate fixed initial/final models on fresh distances and L10/L1000 search;
  retain failures and independent replay evidence. Neither DEV nor a successful
  test of serialization is a substitute for these scientific evaluations.
- The global research goal remains active. No checkpoint promotion, new public
  bound, database mutation, publication, commit or push is claimed here.

## September23 22:51 UTC: optimizer advancement verified

The previously live preparation advanced into real MPS optimization without a
restart. Both step32 and step48 checkpoint publications were observed. A separate
read-only consumer loaded the immutable step48 checkpoint and its initial state:

- SHA256 `6a2eced9715b4d5daffc6afc9fdb0fdc20c4db94c7d48f988de4c2a967e4047b`
  matches the manifest.
- All67 AdamW states report exactly48 updates; weights and optimizer tensors
  are finite.66 model tensors differ from initialization, including local
  raster encoder weights.
- Whole-group relative counts are `[0,2,1,0]`, totaling48/16 as prescribed.
- Paired data trace so far:
  `37ca19eca4007664457c237bed4fa84b8cf417a8d7665fe4a83f8aaf14cccb85`.

This proves actual training, not generalization or embedding utility. The raw
control is still queued behind units. Follow the requested hourly cadence for
ordinary training monitoring after this launch verification: next routine check
not before23:51 UTC, unless completion/error/new user input calls for action.

Short-u generation advanced to500 attempts,19 accepted roots (7 witnessedCC1,
12 witnessedCC2). No final adequacy claim.

## General-distance generation and final audit preparation

Protocol138 and `scripts/build_resource_general_test.py` now freeze the reserved
seed20261073 and3072 attempts, with RF-lineage/DEV/previous-query exclusions,
literal signed-CC/cyclic replay and independent integer determinant checks.
Nine new tests plus17 existing pure-braid panel tests passed; Ruff passed.

Launched:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/build_resource_general_test.py \
  --out artifacts/resource-units-fresh-general-test-v1
```

Unified session29824, PID55750. Initial file publication0 attempts was followed
by a fresh progress file at736 attempts,52 selected pairs across51 query groups:
exact0:26, exact1:11, exact2:6, interval:9. ProcessR at95.7% CPU and2m29.29s
accumulated CPU, with2m31s elapsed. Adequacy still pending (exact2 needs at least8).
No policy/checkpoint was loaded by this generator.

`scripts/audit_resource_training_pair.py` is prepared for the final pair. It
requires both2048-step arms, exact expected initialization, matching data/RNG
traces, complete optimizer moment coverage/counts, finite tensors, correct feature
contract and DEV/checkpoint binding. Two new audit tests plus five feature/state
tests passed. The final audit has NOT run on unfinished training and cannot be
reported as passed merely because its unit tests pass.

## September23 23:51 UTC: hourly live advancement verified

The same training PID53207 is alive (stateR,39% CPU,31m40s accumulated CPU,
elapsed1h07m39s), and original execution session23131 remains running. No restart
or duplicate was observed. Units advanced from the previously checked48 updates
to1168/2048. Its manifest was freshly published23:50:55 UTC. Root manifest remains
at the original arm transition, which is normal: current progress is in the arm
manifest, not the root's timestamp. Raw remains queued in this serial runner;
its absent directory is not a failed worker.

An independent read-only checkpoint load completed successfully:

- File `units/step-001168-b62e04bf8b013c4f.pt`.
- SHA256 `b62e04bf8b013c4f4cbaf9b4f123cd6adfc44ccf4c3812815156fe4adc4ecf1c`.
- All67 optimizer states report1168 updates;269 model/objective/optimizer tensors
  are finite. Whole relative-group counts `[24,19,15,15]` total73=1168/16.
- Draw trace `7609758fe46b9614a0a5c6dbe969afc4ebf04c332c17419b64a00908ab400e68`.
- Last logged batch metric loss0.07841025, positive loss0.16028083. These are
  sampled TRAIN losses, not held-out quality or search improvement.

Short-u wrapper PID53939 and generator child54000 remain alive, session46692
still running. Child isR at99.3% CPU, with64m22s CPU/65m03s elapsed. Fresh progress
file23:51:07 UTC records6800/10000 attempts and64 selected roots:32 per witnessed
CC class,62 query invariant groups. The panel is NOT yet finalized or independently
consumed: retain the fixed attempt budget and final replay/exclusion checks.
Do not stop early or claim64 new knot types. General125-pair panel stays completed.

There is77GiB free on the data volume. No evaluation controller was started, no
process stopped, and no training/source/input change was made. Next ordinary
process check is no earlier thanSeptember24 00:51 UTC (01:51 London), unless a
completion/error event or explicit user request requires earlier action.

The same inspection's input-integrity consumer then verified all1237 training
input pins,71 training source pins and18 generator pins unchanged. Generator
progress had advanced to6900 attempts with the same64 selected records. A bounded
word/error-pattern scan of its generation log found no traceback, exception or
nonfinite-number line; the known optional GUI warning is not a training failure.
