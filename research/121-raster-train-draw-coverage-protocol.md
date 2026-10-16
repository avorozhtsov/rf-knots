# Replay the actual protocol97 TRAIN draw stream

2026-09-21. Read-only data audit, no network forwards, optimizer or new search.
Do not infer missing invariance supervision from the graph sampler alone: the
original corpus also contributes zero-CC pairs. Quantify the complete mixture.

Reconstruct all2048 draws from seed20261047 with unchanged protocol97 samplers,
using dummy reference vectors because reference values do not affect sampling.
Load and validate the original frozen training inputs with their existing
loaders. Only TRAIN statistics are reported; loading the corpus may also verify
other partitions for integrity, but no held-out scores/statistics are used.

Require the complete draw digest, component counts and relative-group scheduling
to match BOTH completed arm manifests. No replacement sampler, RNG stream,
retry, deduplication or fresh augmentation enters the replay. Emit progress and
save a new exclusive artifact lineage. Preserve all existing files.

Report separately for original/suffix/graph/sentinel and their combined mixture:
draw counts by certified interval; distinct exact raw representations and
unordered endpoint pairs (not distinct knots); exact-zero pair counts, identical
versus nonidentical endpoints, nonidentical non-anchor pairs, and changes of
length/strand count. Report maximum input dimensions. Count the actual fraction
of each batch's witness-priority weight carried by each source and zero-CC rows,
averaged across batches. Do not mistake row fractions for loss-weight fractions.

Also report each sampler's complete support, actual multi-positive visits and
distinct exact resource/action contexts, full relative-group visit counts and
L10/L1000 coverage. Canonicalization under mathematical equivalence is NOT used
for these raw-representation diversity counts; equal fingerprints are not called
identical knots. A zero-CC interval proves equivalence only through its already
audited witness, never through observed embedding similarity.

This audit diagnoses supervision coverage, not learned invariance, generalization
or search utility. Source diversity and sample counts alone cannot prove that a
million more draws will help. The frozen paired MCTS evaluation continues, with
process checks no more often than hourly.
