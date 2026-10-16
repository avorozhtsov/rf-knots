# Local raster reuse: geometry diagnostic, not a search result

2026-09-21. Read-only investigation while protocol97 search continues. No
running source, checkpoint, search budget, device, or protocol was changed.
This diagnostic uses synthetic words only, not TRAIN/TEST selection or current
MCTS trajectories. It does not establish an implemented cache or a speedup.

## Why investigate this

The current local tokenizer compiles a crossing-centered 12x7 four-plane tile
for every crossing, applies the shared raster trunk, and extracts two central
row features. `RasterPolicyEvaluator` currently recomputes those features for
each requested state. Two short process samples during the first search batch
were dominated by Torch convolution calls; these samples do not establish
whole-run percentages or within-batch progress.

For a single crossing change, word length, strand count, generator magnitudes,
crop origins and central extraction rows remain unchanged. Only a sign changes.
Consequently only centers whose seven-column cyclic window contains that
crossing can have different compiled input: at most seven centers (at most
min(L,7) distinct centers for short words). Some of those tiles can be unchanged
because that crossing lies outside their strand crop.

This is a local-input identity argument, not a claim that the knot embedding,
graph features, policy logits or value of the position remain unchanged.

## Measured geometry

Fixtures start with generators 1 through N-1, then append inverse pairs
`(g,-g)`, where `g = 1 + (37*i + 11) % (N-1)`, until length L. These deliberately
structured synthetic fixtures are not representative estimates of cache hit
rates on actual search trajectories.

For each fixture, independently flip the sign at positions 0, L/2, and L-1.
Compile tiles in batches of at most64. Compare exact keys consisting of all
compiled binary tile bytes and the central upper-row extraction index
`abs(word[center])-1-origin`. Every changed key was checked to lie within the
seven affected cyclic centers.

| L | N | Flipped position | Changed tile keys | Unchanged keys at same position |
| --- | --- | --- | --- | --- |
| 64 | 33 | 0 | 4 | 60 |
| 64 | 33 | 32 | 4 | 60 |
| 64 | 33 | 63 | 4 | 60 |
| 256 | 65 | 0 | 6 | 250 |
| 256 | 65 | 128 | 2 | 254 |
| 256 | 65 | 255 | 5 | 251 |
| 4096 | 1025 | 0 | 4 | 4092 |
| 4096 | 1025 | 2048 | 2 | 4094 |
| 4096 | 1025 | 4095 | 2 | 4094 |

The original fixtures had25,35,33 distinct keys respectively. This extreme
within-word repetition is fixture-specific and must not be extrapolated to
ordinary knots. No neural forward pass, training update or MCTS was executed
by this geometry diagnostic.

## Candidate optimization and required gates

An inference-only bounded cache of local128-dimensional tokens could avoid
repeating identical local CNN computations for a fixed model. A key must include
the exact tile content and extraction rows, plus a model/configuration/dtype/
device namespace; do not key it merely by absolute crossing position, braid
fingerprint or knot identity. Absolute position and geometry are added later
and must still be computed for the current word.

The transferred axial trunk uses normalization independently within each tile,
not across examples in a batch. This supports reuse of identical local inputs
at the mathematical-function level. It does not prove bitwise equality when
different batch sizes select different numerical kernels.

Before any integration:

1. In a separate versioned engineering gate, compare uncached and cached local
   features, global embeddings, legal-action masks, logits and probabilities.
   Include short cyclic words, crop boundaries, long words, different heads,
   all supported move types, model replacement and cache eviction.
2. Check repeated calls and changed weights; cache entries must never survive
   an optimizer update without invalidation. Do not use detached persistent
   inference entries during training. Differentiable within-forward deduplication
   would need its own gradient-equivalence tests.
3. Measure end-to-end time and memory on prespecified non-TEST inputs, including
   key construction, misses, eviction and global graph computation. Token reuse
   counts alone are not throughput evidence.
4. Replay a separate paired search gate at unchanged budgets/seeds. Floating-point
   differences can change sampling; do not assume identical trajectories from
   small feature errors. Record actual CNN cache misses separately from logical
   policy-state evaluations.
5. Do not retrofit the running protocol97 evaluation, which explicitly excludes
   persistent caching. Finish and retain its uncached reference results first.

After a local edit, graph message passing, multiscale context, pooling and policy
scoring can change beyond the edited neighborhood and still require recomputation.
Changing strand count can also change masks/origins broadly. No constant-time
full-state inference, training acceleration, or million-game feasibility is
established by this diagnostic.

## Inspected source identities

- `scripts/pretrained_raster_tiles.py`:
  `c50e6c553eaeb003335db015d3e5ad05053bcffd52cb3c74eb43b64904cf5542`
- `scripts/pretrained_raster_graph.py`:
  `98c3a2423ea8c9760214176bb8750d0c49a2a9080f67d2f3d17a184df516b506`
- `scripts/pretrained_raster_policy.py`:
  `d7823565574f28298f7d224517e4aa70cafa2d895262a1ba21433a9294eaba27`

The normalization implementation was inspected in canonical PGX
`src/pgx_mcts_bench/networks.py`, classes `MaskedGroupNorm` and
`MaskedAxialCylinderResidualBlock`, SHA256
`b084ec44c6c1a7e24eba588f7bcffa8599ebcc9415789be7590e071239ceb463`.
No local-ablation checkout was used.
