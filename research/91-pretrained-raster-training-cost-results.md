# Pretrained raster / graph: actual metric-training cost

2026-09-21. Protocol90 completed all six cases, exit0. Report:
`artifacts/pretrained-raster-training-cost-v1/report.json`, SHA256
`8a93ddafc83b77b6fe5d0462a4d8ee158522ec50cccf95a43d2d6b2d831109d6`.
All231 source/input pins and all six case hashes rechecked after completion.

## Measurements

CPU, one thread, float32, PyTorch2.13.0, shared16GiB Mac. Actual audited TRAIN
pairs and the exact protocol88 initialization containing Q254 spatial weights.
Each case: one warmup and two measured AdamW steps; timings below average the
two measured full steps. Metric loss only; no policy loss, search, data-loading
time or proof replay included in the timed region.

| Selection by total pair length | Length of each endpoint / strands | Pairs per step | Pairs per microbatch | Seconds per step | Peak process GB |
| --- | --- | --- | --- | --- | --- |
| Median | 11 /4 | 1 | 1 | 0.0283 | 0.866 |
| Median | 11 /4 | 2 | 2 | 0.0525 | 0.764 |
| Median | 11 /4 | 4 | 4 | 0.0901 | 0.902 |
| q90 | 21 /6 | 2 | 2 | 0.1694 | 0.958 |
| Maximum | 254 /21 | 1 | 1 | 1.1009 | 1.119 |
| Upper four rows | 254 /21 | 4 | 1 | 3.8873 | 0.967 |

GB means decimal10^9 bytes. Peaks include process imports, the source corpus,
model, optimizer and activations. They do not isolate tensor memory. The
nonmonotonic peaks are NOT evidence that more examples inherently use less
memory. Cases were separate processes, with concurrent control training and
changing system pressure. Two observations per case are not timing confidence
intervals. No GPU training was measured or launched; MPS availability alone is
not a throughput result.

Quantiles describe50615 deduplicated graph TRAIN rows after excluding three
exact-zero rows, NOT the group/class-balanced sampling distribution of69.
Median selected bounds are exact1; q90 has one[0,1] and one exact1; maximum
single has[1,2]; four upper rows contain three exact1 and that[1,2]. Do not turn
these selected cases into an overall training ETA or unique-knot count.

## What this establishes

The unified network supports real interval-loss backward/optimizer updates,
including finite nonzero gradients into the copied Q254 raster trunk, at the
longest graph-training sizes. Pairwise gradient accumulation makes larger
effective batches feasible without retaining all raster activations at once.
The weighted-mean accumulation is unit-tested against full-batch loss and
gradients, including unequal witness weights and a non-divisible microbatch.
This equality is restricted to CC-only, rank-free loss; a future ranking loss
would need a separate cross-microbatch contract.

Five new tests passed. The combined focused suite across transfer, architecture,
profiling and common-domain diagnostics passed23 tests; targeted Ruff passed.
No historical source was changed. An initial test-only missing import was
fixed before launch. All saved scientific/init checkpoints remained unchanged.
The18 disposable updates (six cases times three) were not saved or evaluated;
they are not a trained candidate and falling probe loss is not generalization.

## Evidence-driven next actions

1. Finish the existing graph/control4096-step pair and the queued independent
   distance/search evaluations. At the last check during this probe control
   advanced3072→3136, with live CPU and all60 frozen source hashes unchanged.
   This is a historical snapshot, not a claim of its future/current state.
2. Implement and test a dedicated policy/search interface for the new dict-return
   network. Do not pass it to the old tuple-return evaluator or invent a value
   prediction. Preserve set-valued positives and same-root relative coefficients;
   budget failures are not unsolvability labels. Account for policy/search cost.
3. Fix the next prospective matched pretrained-versus-random training experiment:
   same newly initialized upper layers, same data order, optimizer, compute
   accounting and independent evaluation. The sole initialization contrast is
   the local Q254 trunk. Use bounded microbatches and identical effective loss.
4. Use distance and L10/L1000 outcomes, not this engineering gate or training
   loss, to decide whether to scale. CPU measurements neither establish nor
   refute a10M-example Nebius GPU plan; a genuine accelerator benchmark and
   cost limit are needed before such a commitment.
