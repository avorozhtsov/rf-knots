# v28: verified mechanical reductions and connected MCTS intake

Prospective preparation, 2026-10-23. Do not modify v27, its pinned sources,
teacher pointer, checkpoints, or the completed paired-learning-v2 study.
No new comparative evaluation or production/UnknotDB promotion is launched here.

## Why v27 supply stopped

The v2 generator used a finite existing256-root roster, fixed24-action witness
suffixes and two COMMUTE/BRAID moves. It stopped at batch52 supply_exhausted.
The horizon48 successor was explicitly a single feasibility pilot, not a loop.
Its1146 paths/223normalized views/15TRAIN invariant groups were admitted into
v27, but no later generator was commissioned. Main MCTS continued independently;
its exporter was fixed to the first356 full batches. v27 pinned both positive
banks and selected one objective-best path per view. Thus192363 newly audited
episodes at the A/B gate were NOT192363 new optimizer examples.

## Scope of this separate version

1. `verified_reduction_search_v28.py`: canonical cost tree and continuation,
   but REDUCE/DESTABILIZE requests use an independently legal, strictly shrinking
   elementary action without a neural forward. Every move increments used_moves;
   CC never changes. All paths replay through the unchanged reference engine.
   No normalization teleport, free macro, predicted lower bound or optimality
   claim. This changes search heuristics and may change move-optimal paths.
2. `learning_pipeline_v28.py intake-main`: bounded contiguous FULL audited
   main batches after an explicit cursor (initial356). Verify report/batch/actor/
   shard hashes and replay ALL episodes, successful AND censored. Only completed
   cycle-free actual MCTS incumbents become positive root labels. Unsolved/cycles
   are not negatives. Retain all1192 old main-positive rows byte-identically.
3. `collect`: new independently generated constructive TRAIN roots instead of
   repeated suffixes of the finite roster. Seed20261023460, index=batch*256+op,
   max256 candidates/30processCPU seconds, max_strands12, depth8..48, raw length
   <=192, proof moves<=192, normalized root length>=8. Four normalized-new views
   in four groups; all old v1/v2/horizon48 rosters, including unsolved partial
   supply, and all frozen panels remain excluded. Group is NOT a knot identity.
   Constructive proofs are supply validation ONLY, never MCTS training labels.
4. Per batch: freeze source/roster/actor, proof/owner audit, then4roots x2Lweights
   x2seeds20261023461/62 =16conditions; canonical16sim ceiling/20processCPU seconds
   per condition, <=320CPU seconds plus replay/setup. Preserve completed
   incumbents on CPU interruption. Full replay before cumulative bank/audit/READY.
   Continue only >=2 distinct newly solved normalized views and CPU/view<=300.
   Failure stops with evidence: no hidden retry/seed/curriculum/budget widening.
   `--batches` explicitly bounds a run1..32. No unbounded service or third worker
   is installed. Initial preparation may run one bounded feasibility batch.
5. `positive_sampler_v28.py`: positive SOURCE50:50 main+retained vs random;
   uniform group -> exact view -> unique semantic path round-robin. This replaces
   v27 best-only replay so newly admitted alternative incumbents are reachable.
   Observation frequency/seed multiplicity never weights paths. Achieved costs
   remain UPPER BOUND observations, not optimal critic targets. No negative
   comparative gradient or intermediate-state gradient added in this version.

## READY is not optimizer consumption

Bank snapshots stay immutable. Each READY carries bank/audit SHA and next main
batch cursor. The next snapshot consumes that cursor; the old bank must be its
exact prefix. Never silently restart at batch0/356 after a later snapshot.
No live mutation of a trainer's pinned banks. In `prepare-spec`, require the
current parent's explicit graceful user_stop_request/full checkpoint, full
model/objective/head/71AdamW/Python-Torch-MPS RNG, byte-identical parent retention,
both independently audited banks and a matching isolated CPU report. The helper
does NOT request the stop or update the teacher pointer itself. New spec/output
only; same absolute sampler origin351232, LR5e-5/clip1, positive encoder cap.05F,
oldaux.05F/totalaux.1F/rehearsal.2F, hourly full saves/minute metadata, native MPS
no fallback, unchanged12GiB disk guard. New native commissioning must prove
bitwise full restore, finite moments/counter deltas, and actual newly admitted
path consumption BEFORE switching the teacher pointer. No claim of ACTIVATED
until these checks pass. Preparation leaves v27 and main MCTS healthy.

The isolated two-update CPU test may select a future sampler slot by metadata
to exercise a new admitted path; it is NOT a skipped-step native continuation,
not saved/promoted, and adds zero time/steps to any learning-age counter.

## Short prospective utility check (proposal, NOT launched)

Keep the completed32-root A/B panel permanently closed for model selection.
First use a separate TRAIN-only diagnostic cohort to measure mechanical move
count/replay, actual new-label optimizer draws, success and neural forwards.
For a new user-approved comparison freeze a fresh score-blind24-root validation
panel,6each normalized-length band8..32/33..64/65..96/97..128, all old panels and
TRAIN groups/states excluded. Freeze roots BEFORE any arm scores. Three arms:
fixed parent+old search, SAME parent+auto reductions, fixed v28 candidate+auto
reductions. This separates search and learning changes; no checkpoint selection.
24 x3arms x2seeds xL1000 x10CPU seconds =1440searchCPU seconds upper plus replay/
load; optional L10 is a separately declared extension, not a posthoc rescue.
Report paired success/root-cluster intervals, CC/moves on joint successes,
actual CPU/wall/simulations/neural/automatic actions and censorship. Low counts
remain inconclusive, never proof of equivalence or benefit. No production change.

## Preparation commands

`PYTHONPATH=src:scripts:/Users/artemvorozhtsov/projects/pgx-mcts-bench/src`
and `PYTHONDONTWRITEBYTECODE=1` apply to every Python command below.

- `learning_pipeline_v28.py intake-main --out artifacts/v28-main-intake-000356 --parent-bank artifacts/mcts-root-positive-bank-v1 --parent-audit artifacts/mcts-root-positive-bank-audit-v1/report.json --first-batch356 --batches1`
- `learning_pipeline_v28.py collect --out artifacts/random-TRAIN-stream-v28 --parent-bank artifacts/random-horizon48-feasibility-v3/bank --parent-audit artifacts/random-horizon48-feasibility-v3/bank-audit/report.json --batches1`
- `smoke_connected_learning_cpu_v28.py --main-bank <main READY bank> --random-bank <random READY bank> --out <new isolated diagnostic directory>`
- Only after explicit graceful handoff and the native commissioning checks above:
  create a NEW spec via `prepare-spec`, then use the v28 learner. No command in
  preparation stops v27, deploys a model, or authorizes automatic promotion.

To grow towards1k/3k/10k views, increase only the number of successful audited
batches, not per-batch budget or retry count. Track separately generated raw
views, normalized-new views, groups, MCTS solves/replayed episodes, admitted
semantic paths, READY snapshots and ACTUAL distinct optimizer draws. Failed
supply/yield or a persistent READY-not-consumed backlog is a STOP requiring review,
not a reason to report healthy continuous generation. No deletion/DB/cloud/Git.
