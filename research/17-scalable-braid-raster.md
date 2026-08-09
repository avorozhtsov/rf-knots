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

Identity padding may add straight strand rows or empty word columns before search.
Because it changes shape without changing the represented knot, preprocessing
padding is free and recorded. Learned insertion/removal during an episode is a
semantic environment operation and is recorded explicitly.

## Architecture family

`raster-axial` applies shared residual blocks along word and strand axes, masked
normalization, and relative/cyclic word positions. A routed policy head selects a
local block, then an action within the block. This is the short-term scientist.

The scalable extension adds hierarchical `2×2` and `4×4` block summaries and
recurrent message passing. All levels share block weights, so increasing the
canvas reuses the learned operator rather than introducing a new alphabet.

`strand-graph` is the complementary family: strand tokens exchange messages at
crossings and the policy routes back to a crossing/strand pair. It tests whether
explicit connectivity transfers better than image-like locality.

## Required gates

1. Match `window-local` on easy held-out knots at matched parameters and compute.
2. Preserve accuracy when identity rows/columns and cyclic word shifts are added.
3. Transfer to a strand count absent from training.
4. Show that a global-information task improves with hierarchical/recurrent blocks.
5. Pass the same native-learning, budget-calibration, retention, and sharing gates
   as every other scientist.

Passing a small supervised raster probe is only an admission signal. It does not
establish RL learning, long-horizon routing, or hard-knot performance.
