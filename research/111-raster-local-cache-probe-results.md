# Local raster cache probe: numerical parity, not full search equivalence

2026-09-21. Follow-up to110, executed separately from the unchanged protocol97
evaluation. No evaluation worker was polled outside the requested hourly schedule.
No frozen source or existing checkpoint was edited. No training or new MCTS ran.

## Executed artifact

Command from the RF Knots root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/probe_raster_tile_cache.py \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-tile-cache-local-probe-v1
```

The output directory must not exist; the script refuses overwrite. To reproduce,
use a new output directory. The command exited0. Torch version2.13.0, CPU float32,
one Torch thread. The source is the original transferred Q254 local checkpoint,
not either final protocol97 model:
`397ca781bb38cee215b53cc068219aed80325c1f84a177d829b879bc80a20771`.

Report: `artifacts/raster-tile-cache-local-probe-v1/report.json`.
SHA256: `386875011d39d017fb3cc977064b124e44a638e7271d1aa0a1834366d312561e`.
All four recorded source/checkpoint hashes were rechecked after completion.
Logical tile counts and computed-tile counts were independently summed from
the22 case records and matched report totals.

## What passed

-22 comparisons cover empty/one-letter/short words and synthetic words up to
  length256 with65 strands, repeated calls, a crossing change and cyclic shifts.
- All local128-dimensional outputs passed `rtol=1e-5, atol=1e-5` against uncached
  inference. Maximum absolute error was4.76837158203125e-6. Only17/22 cases were
  bitwise identical: **this is not a bitwise-equivalent replacement**.
-1316 logical tile requests required73 computed tiles, with1243 reuse hits and
 73 retained entries below the512-entry limit. All six immediately repeated
  fixtures required zero additional tile computations. These are deliberately
  repetitive synthetic fixtures, not expected hit rates on search trajectories.
- The real local model's weights/configuration were unchanged. No optimizer
  updates occurred. Current protocol97 search does not import this probe.

Nine focused tests passed in1.84s, with no skips. They use a smaller native
masked-axial trunk and cover repeat/CC/cyclic output parity; in-place updates,
`.data` writes, parameter replacement, checkpoint loading and normalization
epsilon changes; bounded eviction; non-aliasing returned outputs; rejection of
training/gradient-enabled calls, unsupported modules/dtypes, invalid inputs and
nonfinite weights. Targeted Ruff checks passed.

## Deliberately conservative implementation

`TileCacheProbe` is a standalone callable, not a new production model class.
It accepts only the existing tile encoder and audited native masked axial
modules, in eval mode with gradients disabled, on CPU float32. It hashes all
model tensor bytes and relevant configuration at each call, clearing its cache
on changes. This detects even writes that bypass ordinary parameter-version
counters, at a cost that must be included in future throughput measurements.
Binary tile content plus central extraction row identifies a local output;
returned stacked tensors do not expose cached tensor storage.

## Still required before use

No full global embedding, policy probability, all-move or trajectory equivalence
has been demonstrated. The final trained arms have not been tested with this
cache. Runtime/backend changes and other devices/dtypes are outside this gate.
Small local floating-point differences can affect sampled actions, so do not
retrofit or combine these results with the frozen uncached search matrix.

The recorded timings are single samples during concurrent computation, include
warmup/order effects and use highly repetitive fixtures. No speedup estimate or
general throughput claim is justified. Follow the separate full-network and
paired-search gates in110 before proposing production integration. Training
reuse needs a distinct gradient-equivalence design; detached persistent cache
entries cannot be used as a substitute for trainable local features.

Source SHA256:

- `scripts/probe_raster_tile_cache.py`:
  `21c6d3e4394238b302b7be520691da2e9133f29018300a3890635b19229725b2`
- `tests/test_probe_raster_tile_cache.py`:
  `a6bead55dc6844d4caa5ad75f77f63093645742b6078917f07fa3280627ac613`
