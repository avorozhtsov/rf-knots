# Protocol97 execution and evaluation readiness

2026-09-21. This is a new output lineage; protocol94's interrupted16-step
checkpoint, logs and sources remain preserved. No paid GPU was started.

## Launch and fixed inputs

Run directory: `artifacts/pretrained-raster-log-training-pair-v1/`.
Launcher: `scripts/run_raster_log_pair.py`; canonical PGX checkout:
`/Users/artemvorozhtsov/projects/pgx-mcts-bench`.

The launcher serializes Q254-local and random-local training,2048 updates each.
The new criterion and16-update full-group schedule are fixed in protocol97.
The original initialization is reloaded, not the interrupted94 weights.
Preflight checked1143 input hashes and67 source hashes. Gate96 report SHA256:
`661fa3703c27cf2ed0f64887dfb42969e4ccab5a2f1537b6b17bb78834ef2158`.
All eight real-network gradient cases passed with zero optimizer updates.

At launch parent PID94005 spawned worker94033, with fresh manifest/initial.pt
and increasing CPU time. Preparation includes full replay/validation of TRAIN
data and takes several minutes before optimizer updates. PIDs are historical
observations, not a substitute for checking real current processes/files.

The first durable checkpoint reached16/2048 optimizer updates in130.68 seconds
of optimizer-loop time (excluding preparation). SHA256:
`ff013a56378283e451c9f855e591b6c816a3121eb86ffc6965c2eb6008dc5fe1`.
Reloaded model/objective/optimizer tensors were finite; all populated Adam step
counters were16. Full-group counters were[1,0,0,0], with334 relative states;
the four group sizes matched[334,52,74,54]. All1143 inputs and67 sources were
rechecked unchanged. This establishes real training progress, not completion.

At that update metric loss was0.668645, positive loss7.701512 and full-group
relative loss70.059250. These are TRAIN observations, not held-out distance or
search gains; the new and old relative loss values are not comparable objectives.
Random-local arm remains queued behind Q254-local, not concurrently training.

## Verification completed before scientific evaluation

- 49 focused tests passed across tile transfer, scalable architecture, training
 accumulation, policy/search interface, log preference, new training schedule,
 recovery counters, final training audit, metric scoring and search-batch replay.
- New and changed protocol97 implementation/test files pass targeted Ruff;
 no whole-repository lint claim.
- A real Q254-initialized network self-pair (trefoil versus itself) scored0.0
 with the new microbatch1 scorer. This is an interface/numerical smoke check,
 not independent scientific evidence or a trained success.

## New evaluation consumers, not completed evaluations

`scripts/audit_raster_log_pair.py` requires both arms completed, verifies exact
initial tensors, final/progress hashes, finite parameters/optimizer state,
training counters, full relative-group sizes/counts, and matched RNG/data order.
It refuses incomplete training and tampered sample accounting.

`scripts/evaluate_raster_log_distances.py` requires that final paired audit.
It scores each arm's saved initial/final model on the existing frozen70 short-u
original/lifted64+64 and74 general-distance126 panels. It verifies source/data
pins, loads saved objective calibration without TEST fitting, and reports:

- Per-record interval violation and equal-fingerprint-group summaries.
- Exact distance0/1/2 errors and constant0/1/2 controls.
- Strict ordering only when two certified intervals are disjoint; ties receive
  half credit. Pairwise comparisons are descriptive, not independent samples.

Inference uses raw diagrams and clamps cosine to[-1,1] for floating-point safety,
matching the new raw-distance search interface. These are reused frozen TEST
panels, not fresh independent replications; Q254's historical exposure remains
incompletely audited. No quality scores from this new experiment exist yet.

The final-model full64-root L10/L1000 search, replay, semantic cost-floor audit
and combined scientific interpretation remain required. A new search runner
must use the prior-only raster interfaces, not the old SitePolicyValueNet loader
or normalized-diagram heuristic. Do not call training integrity alone promotion.

`scripts/raster_cost_search.py` supplies the new batch core: raw-diagram distance
priors, raster policy continuations, no scalar value or oracle incumbent, actual
encoding/forward/action counters, and replay of every continuation and success.
Two tests cover a solved one-crossing fixture and a censored trefoil under zero
CC allowance, unchanged parameters, exact neural-state accounting, and rejection
of empty batches. These are small synthetic-interface tests, not TEST results.
The full resumable search controller and final scientific summary are still to
be connected; no complete64-root evaluation of the new model has been launched.

## Parallel earlier experiment

The earlier protocol69 graph/control training completed4096 updates per arm.
Its independent70 controller remains active: six preparatory/metric jobs are
complete and the initial-model MCTS worker persisted24/128 conditions at this
observation. CPU advancement and newly written condition shards were checked.
Protocol74's supplemental evaluation completed earlier; the combined final
report still requires the entire70 search/replay matrix. No worker was restarted.
