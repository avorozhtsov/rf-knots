# Access restored; hourly monitor active;139 resume and149 launch

Subsequent initial-launch check: the139 search manifest now records new
workerPID21103 with next_condition=32 and status=evaluating. Resume verification
has therefore completed and search continues after the preserved prefix. No
additional condition batch or solution is yet committed.149 remains in
historical_data_and_initialization with no verified optimizer update yet.

2026-09-25,18:38–18:43 UTC. User asked for current status after previously
authorizing continued work, repairs and hourly reports. Permission mode now
allows reviewed, narrowly scoped execution outside the workspace sandbox.

## Actual state before action

Approved process inspection found no139 controller/search or149 training/evaluation
process. A second inspection after an interrupted conversational turn confirmed
the same absence. Torch2.13.0 reported mps_available=true in the approved host
execution. This differs from the prior sandbox-only false result; no backend
contract or fallback was changed.

139 files still recorded obsolete PIDs15420/21909 and the last committed prefix
of32/128 conditions at2026-09-24 13:34:33 UTC. Four shard hashes matched. Counts:
192 episodes,117152 policy evaluations,0 solutions. The controller had12/24
completed jobs. The prior processes were genuinely absent, not merely silent.
No terminal error explaining their disappearance was present in the inspected
search log, so the external termination cause is unknown. Do not claim work was
continuously running during this gap.

149 output directory was absent.147/150 consumers and152 gate remained passed;
there was no existing production training to resume or duplicate.

## Hourly reporting repaired through the supported scheduler

Updated the existing heartbeat `rf-knots-hourly-mcts-evaluation-check`, not a
new automation. Tool returned ACTIVE; persisted configuration was independently
read back: name `RF Knots — hourly training, search and reports`, ACTIVE,
hourly interval1, same target task01a04e1f-a865-7803-8e68-e71f1bd6409a.

The durable prompt now covers the continuing139/147/150/152/149/155 workflow,
requests a Russian report every hour even without changes, allows verified
recovery and in-scope fixes, and explicitly requires a fully non-neural heuristic
comparison before claiming superiority at minimum CC. It preserves frozen
contracts, dirty work and the no-cloud-spending/no-publication boundaries.
It no longer pauses merely because old97 completed. No other heartbeat was
activated. Local scheduled work still requires the host/app to remain available.

## Executions and initial verification

139 resumed exactly with its existing output and --resume:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
 scripts/run_resource_evaluation.py \
 --run artifacts/raster-resource-units-training-pair-v1 \
 --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
 --out artifacts/raster-resource-units-evaluation-v1 --resume
```

Session23235,controllerPID20100. It verified the prior completed prefix and
launched exactly one `evaluate_resource_search.py ... --resume` childPID21103.
Controller manifest updated18:41:28 UTC with that child. At initial observation
child100%CPU,elapsed49s,CPU45.38s. At18:43 the child search manifest still
recorded the old32-condition prefix: resume verification was not yet committed
as a new search state. This is a live resume process, NOT a new solved result.

149 launched once with its unchanged complete prerequisite gates:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
 .venv/bin/python -u scripts/train_long_general_pair.py \
 --long-data artifacts/long-general-cc-supply-v1 \
 --short-test artifacts/long-mixture-fresh-short-u-test-v1 \
 --execution-gate artifacts/long-general-mps-execution-gate-v1/report.json \
 --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
 --out artifacts/raster-long-general-training-pair-v1
```

Session98193,PID20138. Parent manifest created18:41:33 UTC with contract SHA
`f87d2b536e0eca77a599f843604f67aebb3318d20d7271ec16bd630698f9af51`.
Phase historical_data_and_initialization; actual CPU advanced10.71s→65.55s,
observed100%CPU. At18:43 no arm manifest/checkpoint yet existed. Thus the
production runner is launched and initializing, but zero optimizer steps are
verified so far. Do not describe it as a completed or improving neural model.
The optional Plink tkinter warning is not a mathematical/training failure.

The existing protocol remains two2048-update arms with identical Q254-local
initialization, raw features, global embeddings and head-relative positions.
Control uses the original mixture; long replaces8/32 metric-pair slots with
certified long pairs. Both encoder and policy train. This is not million-game
online RL and cannot by itself prove superiority over classical search.

Keep all pinned sources unchanged, preserve partial outputs, continue initial
launch verification and then rely on the hourly monitor. New errors must be
diagnosed, not hidden by relaxed gates. The broader scientific goal is active:
neither improved public upper bounds nor neural superiority is established.
