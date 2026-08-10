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

All normalization is mask-aware. Inactive row-capacity and all-straight identity-
column augmentation are used during pretraining to force padding invariance.
An active extra strand is introduced only by a verified Markov stabilization;
plain all-straight active rows are not knot-preserving padding. Cyclic word shifts
and paired inverse encodings are also augmentations; source lineage is retained so
equivalent augmentations cannot leak into held-out evaluation.

The current development comparison is three-way: `window-local`, the frozen local
`raster-axial`, and the full `raster-routed` candidate. They train from scratch on
the same source-disjoint curriculum with identical MCTS simulation counts; model
evaluations, parameter counts, and wall time are reported separately because the
full raster is more expensive per evaluation. `raster-routed` achieved 100% first-
rung solve rate but failed the known crossing-change optimum before and after its
adaptive `F=8` continuation. It is therefore not the fourth scientist; the main
roster retains `raster-axial` while scalable variants continue separately.
