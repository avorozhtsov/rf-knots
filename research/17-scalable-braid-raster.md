# Scalable braid scientists

The active goal is a solver whose parameters do not depend on a fixed maximum
strand count or braid length. The detailed exploratory history is archived in
[`../archive/pre-semantic-moves-v1/research/17-scalable-braid-raster-full.md`](../archive/pre-semantic-moves-v1/research/17-scalable-braid-raster-full.md).

## Representation

Render a braid as a `strands × word-position × channels` tensor. Channels encode
empty/straight cells, over/under crossings, legality, masks, and the location of
the action cursor. Word position is cyclic for a closed braid. Strand position is
masked and is not automatically wrapped: a wrap generator is a separately defined
semantic extension, not ordinary `B_k` hidden in preprocessing.

Padding may add inactive masked strand-capacity rows or all-straight identity word
columns before search. These do not change the represented braid and are free but
recorded. Activating a new straight strand is not padding: under closure it adds a
link component. A taller equivalent braid therefore requires a verified Markov
stabilization, including its crossing. Learned insertion/removal during an episode
is a semantic environment operation and is recorded explicitly.

## Architecture family

`raster-axial` is the frozen seven-column baseline. It applies shared residual
blocks along word and strand axes but inherits the serial fixed-capacity policy
head and cannot supply a trustworthy global task critic.

`raster-routed` is the first unadmitted scalable candidate. It receives the full
canvas, reuses one masked residual operator at word dilations `1, 2, 4, 8`, and
uses one row-pair scorer for every ordinary generator and the dynamic active-
strand seam. Its parameter shapes are independent of strand capacity. Separate
masks distinguish active strand workspace from real Artin-word columns, and
objective/internal budgets condition every recurrent application.

Hierarchical `2×2` and `4×4` summaries remain a later ablation, not a property of
the current candidate. They should be added only if dilation is the measured
receptive-field bottleneck.

Its first gate found 10/10 feasibility but mean crossing-change cost 0.3 instead
of the known optimum 0. An exact-state `F=8` continuation worsened that cost to
0.6 and was also much slower than the local raster. It therefore remains outside
the main roster. This is evidence about the current learning path, not a rejection
of the scalable representation family.

`strand-graph` is the complementary family: strand tokens exchange messages at
crossings and the policy routes back to a crossing/strand pair. It tests whether
explicit connectivity transfers better than image-like locality.

## Required gates

1. Beat or complement `raster-axial` and remain competitive with `window-local`
   on easy held-out knots, with both network evaluations and wall time reported.
2. Preserve accuracy when identity rows/columns and cyclic word shifts are added.
3. Transfer to a strand count absent from training.
4. Show that a global-information task improves with hierarchical/recurrent blocks.
5. Pass the same native-learning, budget-calibration, retention, and sharing gates
   as every other scientist.

Passing a small supervised raster probe is only an admission signal. It does not
establish RL learning, long-horizon routing, or hard-knot performance.
