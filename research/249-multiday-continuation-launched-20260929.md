# Three-day paired continuation launched; three-hour monitor

2026-09-29,11:30 UTC launch;11:36 UTC initialization verified. User explicitly
authorized actual training and then requested three–four days, checking status
every three hours. Implemented separate protocol248, without modifying frozen
149/139/152 sources, checkpoints or running searches. No commit/merge/push.

27 targeted tests passed (new continuation, unchanged149 orchestration/data).
New CPU toy verifies uninterrupted versus interrupted/resumed128 additional
updates with identical full checkpoint trees. This is engineering evidence,
not real raster scientific quality. Ruff and whitespace checks passed.
Reviewed host check: Torch2.13.0, native MPS available, actual tensor operation
passed with fallback disabled.68GiB free disk before launch; checkpoints ~3MiB.

Detached process51397 launched once, exact command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
 .venv/bin/python -u scripts/continue_long_general_pair.py \
 --parent artifacts/raster-long-general-training-pair-v1 \
 --parent-audit artifacts/raster-long-general-evaluation-v2/training-pair-audit.json \
 --long-data artifacts/long-general-cc-supply-v1 \
 --execution-gate artifacts/long-general-mps-execution-gate-v1/report.json \
 --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
 --out artifacts/raster-long-general-training-continuation-v1
```

Log: artifacts/raster-long-general-training-continuation-v1-launch.log.
Contract SHA48ecf1c8a7c8df2b76803fd9961706b69fddf8042ff4965fdadd402d32dcea29:
1971 input bindings,214 source snapshots; original149 final checkpoint SHAs
89c109f06d19f21f6b8ccd06c2e0e619155e962f26e333446d6d094d97e23766(control),
64fd57deb9ac195fc40db8adbb929df90fde67aa4c2ca3178b405d83aaf741d8(long).
Parent completed state, all parent histories/DEV/checkpoints and passed audit
bindings rechecked. CPU advanced9.15s→114.98s→282.40s at~93–100%; current process
command matched. Initial loading is NOT optimizer advancement.

At11:36, parent phase changed to training/matched_round; control restored model,
objective, AdamW, both Python RNGs and Torch/MPS RNGs bitwise exactly. Original
Q254 anchor preferences matched parent. New baseline step2048 checkpoint SHA
d08d4b487c6b2d347d9e342ae104fa67e2767d052db3b396bb4f3cdab3ad62b7.
Control entered optimizer phase; next target3072, then long to3072. New committed
optimizer updates still require the first64-step checkpoint; do not call the
baseline copy new learning. Repeated1024-step paired rounds continue until72h
summed committed training work, with DEV every4096 and unchanged losses/data.

Existing heartbeat updated through supported automation API, same id/thread:
rf-knots-hourly-mcts-evaluation-check, ACTIVE, interval3hours, prompt explicitly
covers248 continuation, exact resume, separate DEV/new final TEST and deferred
pure heuristic comparison until positive signal. Persisted configuration read
back. This is local execution: host must stay available; report outages honestly.
No second scheduler or cloud resource created.

Existing searches23761 and31474 were active with~99%CPU before launch and left
untouched. Old150 TEST is already observed and cannot select new checkpoints.
Independent continuation audit and fresh replay/exclusion-certified final TEST
are pending. Training completion/DEV loss are not neural superiority.

## Verified new optimizer updates,11:39 UTC

Control committed cumulative2112 steps (64 NEW updates),219.479297s additional
training time. Checkpoint SHA
51d0eaa6a6f16f54e0506886c84a42d55a5bc1f60d01d2a707397482986c7806.
Independent readback verified content hash, config/arm/contract/counter bindings,
66 model tensors changed versus parent, AdamW states changed, ALL recorded Adam
step counters exactly2112. Metric pairs67584 and relative groups132 (2112/16).
Latest batch metric loss0.076028736, positive loss0.056496107; these are batch
losses, not evidence of DEV/search improvement. Process51397 still active,
CPU time440.50s,elapsed9m33s,~50%CPU and904032KiB RSS. This is VERIFIED ACTIVE
optimizer training, not merely a copied2048 checkpoint. Long arm awaits the
matched sequential first block; no duplicate worker needed.

Power assertion process54133 (`caffeinate -s -w51397`) is bound to this worker:
prevents automatic system sleep only on AC power until training exits; screen
and manual sleep remain available. Inspection now returns to three-hour cadence.
