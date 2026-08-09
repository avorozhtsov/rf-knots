# Braid raster representation

This note specifies the active raster input. The exploratory probes and old plots
are archived in
[`../archive/pre-semantic-moves-v1/research/18-raster-representation-full.md`](../archive/pre-semantic-moves-v1/research/18-raster-representation-full.md).

For `k` strands and `n` word positions, construct a masked tensor with one row per
strand and one column per position. A minimal cell encoding distinguishes empty,
straight, over-left, under-left, over-right, and under-right states. Additional
planes identify real versus padded cells, the active cursor, and legal semantic
action sites.

The closed braid is cyclic in the word direction. It is not silently cyclic in the
strand direction. If affine/cyclic strand generators are enabled, they are named,
verified semantic actions and are available consistently to every comparison arm.

All normalization is mask-aware. Identity-row and empty-column augmentation is
used during pretraining to force padding invariance. Cyclic word shifts and paired
inverse encodings are also augmentations; source lineage is retained so equivalent
augmentations cannot leak into held-out evaluation.

The first current comparison is `window-local` versus `raster-axial`, trained from
scratch on the same source-disjoint curriculum and evaluated at matched parameter
count and search compute. The raster family advances only if it reaches at least
70% solve rate and its strand-count transfer is not explained by extra simulations.
