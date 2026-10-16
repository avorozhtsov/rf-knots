# Full-network cache probe passed on synthetic fixtures

2026-09-21. Protocol112 completed separately from the ongoing uncached protocol97
search. No training, TEST evaluation or MCTS was performed by this gate. No
running worker, frozen source or checkpoint was changed or polled off schedule.

## Reproducible artifact

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/probe_raster_cache_full_network.py \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-cache-full-network-gate-v1
```

The command exited0. The output directory must be new; reproduction must use a
different directory. Report:
`artifacts/raster-cache-full-network-gate-v1/report.json`.
SHA256: `4c2323439bd19d90da9d8d31935ed904fca9dc4e1606cfbc7950c9010838f3bd`.
All132 recorded input/source hashes were independently checked after completion.

Both completed protocol97 checkpoint identities in112 were enforced before
loading. Model and objective tensors were identical between reference/private
cached copies and unchanged after the gate. The runtime was Torch2.13.0, CPU
float32, one Torch thread, eval/no-grad; no optimizer or global layer was changed.

## Coverage and results

Each arm has102 comparisons:17 synthetic states × two cost weights10/1000 ×
three head modes (absent, zero, last). The complete Cartesian coverage and tile
counters were independently recomputed from the report. Candidate order and
padding were matched. Short fixtures cover all nine action kinds and their
independent reference-verified one-move successors. Long fixtures reach L256/N65.
These are candidate subsets, not exhaustive action spaces or a knot test set.

PASS needs its own resource-constrained fixture: it is legal only when no other
move exists. An initial fixture preflight failed on an attempted all-nine-kinds
single state before any comparison report was produced. The fixture was corrected
to eight non-PASS kinds plus a separate PASS-only state; tolerance and model
identities were not changed. Eleven focused tests then passed in2.14s (no skips),
including the nine111 cache tests and two legality/coverage tests. Ruff passed.

Maximum absolute discrepancies:

| Output | Q254-local | Random-local |
| --- | --- | --- |
| Encoded site tokens | 2.8610e-6 | 3.3975e-6 |
| Knot embedding | 1.7881e-7 | 1.3411e-7 |
| Layout embedding | 1.6764e-7 | 1.5469e-7 |
| Isotopy embedding | 1.9372e-7 | 1.8626e-7 |
| Legal-action logits | 3.4332e-5 | 1.9073e-5 |
| Action probabilities | 3.8743e-6 | 7.7486e-7 |
| Learned CC distance to empty anchor | 3.2783e-7 | 7.4506e-8 |

All comparisons passed the predeclared combined relative/absolute tolerance
`rtol=1e-5, atol=1e-5`; this is not an absolute-error-only threshold. Effective
action masks matched exactly, every selected legal candidate stayed enabled,
and no argmax action changed in either arm. All compared values were finite.

Each arm requested2394 logical tiles, computed119, and reused2275, with zero
cache invalidations during unchanged-weight inference. This reuse fraction is
dominated by deliberately repeated synthetic states/head/budget variants and
is not a forecast for live search.

## Interpretation and remaining work

The cache now has bounded numerical-equivalence evidence through the complete
trained network and learned distance objective, not only through the original
local trunk. It has no demonstrated speedup, trajectory equivalence or scientific
quality benefit. No claims about unseen knot quality, higher search success,
lower L10/L1000, or million-game throughput follow from this engineering gate.

Even unchanged argmaxes do not prove equal sampled actions: small probability
differences can cross sampling thresholds. A separate bounded paired-search
engineering gate, replay checks, and end-to-end timing under a controlled setup
are still needed. Do not retrofit the active uncached evaluation or mix cached
results into its matrix. Training/autograd reuse remains explicitly unsupported.
