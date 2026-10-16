# Evaluation frozen before149 optimizer use

2026-09-24. This extends149/150 and must be source-hash-bound in149's manifest
before training. Evaluator implementation and real-weight resume/cache gates
remain required; this document is not an executed evaluation.

Use147's TEST general pairs and150's64 original short-u roots plus their64
385/34 lifted views. Bind both completed consumers and exact corpus/panel hashes
in the training contract. No use of DEV for selecting checkpoints; evaluate
only shared initialization and both2048 endpoints. Both initial networks must
be proven tensor- and function-identical before using one shared initial result.

Distance matrix: shared initial,control-final,long-final, each on general TEST
and short-u(original/lift separately). Report raw predictions, interval
violations, constant1, exact classes, positive-only and all-class rankings,
query-group averages, word-length/strand strata. Groups are invariant buckets,
not identified knot types; pairwise comparisons are dependent. Register train
size support before calling the lifted cohort extrapolation beyond all TRAIN.

The complete historical input-pool inventory in156 now establishes maxima254
letters/21 strands, including all positive and relative policy inputs. Its
report SHA256 is `1e0b3e58c0456ae0153a03d308350548949e9f914e30516b6ef53c3214146b82`.
Together with147's required384/33 source limits, the150 lifted minimum385/34
is beyond supplied149 optimizer-pool sizes on both axes. This does not audit
all earlier Q254 pretraining or establish useful size extrapolation; retain
those limitations and report actual original/lift sizes separately.

Search matrix(five configurations, each128 conditions):

- control-initial,distance tree prior(shared initial model);
- control-final,distance tree prior;
- long-final,distance tree prior;
- control-final,uniform tree prior;
- long-final,uniform tree prior.

Seed20261079; both L10/L1000 for every original root;8 simulations; c_puct1.5,
temperature1,widening_batch4,insertion_candidates4,batch_conditions8. Native
per-root caps, initial head positions, legal-action semantics and resource
updates match139's `build_roots` contract. CPUfloat32,threads1,local tile cache
capacity512 ONLY after checkpoint-specific full-network/action/search parity.
No scalar value, graph tail, external incumbent, unsupported truncation or
oracle action selection. No test trajectories added to optimization.

Replay every successful and budget-censored trajectory and the reference
witness. All costs include complete native paths. Report new/lost successes,
common-success costs, CC/witness improvements and all censored counts. Equal
simulations do not imply equal neural/hardware work: report forward/state/action
counts and elapsed time. Tree-prior removal alone is not an embedding-channel
ablation; neither training completion nor a lower distance error proves search
utility. This experiment is not a named public-knot campaign or database release.

The149 runner binds this entire matrix/config in its immutable training
contract. Downstream consumers must reject missing configurations, changed
roots, unbound checkpoint features or incomplete replays; no promotion from a
convenient subset. Existing139 runs and their frozen sources stay untouched.

## Runner implementation and verification

`scripts/train_long_general_pair.py` now implements the149 two-arm training
orchestration. It requires both completed consumers and binds this evaluation
matrix before optimization. It reuses135's unchanged metric/policy mathematical
helpers with a checked compatible loss contract, not its runner or mutable
configuration. All model parameters and the CC objective scale are optimized.

Checkpoints are immutable/content-addressed and include AdamW moments, both
Python RNG streams, Torch/MPS RNG states, shared/selected draw digests, full
relative-group counts, and actual metric/policy token/size counters. Checkpoint
and metrics history publish before their manifest; uncommitted content-addressed
files remain preserved. A completed resume only verifies and returns, without
mutating the completed manifests. Source snapshots and input pins must agree.

The new orchestration tests perform an uninterrupted32-step tiny CPU run and
a simulated interruption after the16-step checkpoint, then resume its last16
steps. Final parameters, objective, AdamW tensors, Python/Torch RNGs, draw traces,
relative counts and work counters agree exactly. Completed re-entry leaves all
files byte-identical. This is a tested orchestration mechanism, NOT a real-raster
MPS resume result: the CPU toy explicitly mocks MPS RNG functions. The real149
MPS gate remains necessary once audited147 TRAIN is available.

Additional tests cover checkpoint identity/path/feature contract, metrics
history gaps/hashes, missing final DEV, required evaluation input bindings and
refusal to create a training output when source data is unavailable. The
combined targeted suite passed88 tests; Ruff and tracked-diff whitespace checks
passed. Neither
production149 training nor150 acquisition has been launched yet. The commands
below are the intended next runs AFTER their prerequisite audits, not evidence
of execution:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/build_long_mixture_short_u_test.py \
  --long-data artifacts/long-general-cc-supply-v1 \
  --out artifacts/long-mixture-fresh-short-u-test-v1

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/build_long_mixture_short_u_test.py --audit-only \
  --long-data artifacts/long-general-cc-supply-v1 \
  --out artifacts/long-mixture-fresh-short-u-test-v1

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/train_long_general_pair.py \
  --long-data artifacts/long-general-cc-supply-v1 \
  --short-test artifacts/long-mixture-fresh-short-u-test-v1 \
  --execution-gate artifacts/long-general-mps-execution-gate-v1/report.json \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-long-general-training-pair-v1
```

Do not launch the last command merely because unit tests pass: the full data
audits and the real-weight MPS/resume check must precede it. Independent149
evaluation dispatch and the final paired report are implemented in155 but still
need actual execution; a training manifest's `completed` status explicitly
leaves independent evaluation pending.

The independent post-exit training auditor is now implemented in
`scripts/audit_long_general_training_pair.py`. It reconstructs both full draw
streams and their independently accumulated token/shape counters at every
committed checkpoint, checks initial Q254 weights, AdamW states and steps,
finite tensors, source/contract bindings, final parameter changes, and initial
and final DEV record/group/class arithmetic. It requires both Python sampling
streams to align; backend RNG alignment is reported separately because different
input shapes need not consume hardware randomness identically. Passing this
audit is training-integrity evidence only, not TEST accuracy or search utility.
Eleven new tests cover draw replay, work accounting and tampered DEV summaries,
including both real paired samplers with nonzero reference probabilities versus
the auditor's zero-placeholder references: their data traces must agree.
No actual149 training output exists yet for a full post-exit audit.

The independent checkpoint/panel loader and distance evaluator are now
implemented; see153 for tests and the real-weight engineering smoke. This does
not change the frozen matrix or count as a completed149 TEST evaluation.

The checkpoint-specific cache gate, resumable search worker and independent
full-path replay auditor are implemented as well; see154. They preserve the
same five configurations and native139 search semantics and have not run on
149 scientific outputs yet.
