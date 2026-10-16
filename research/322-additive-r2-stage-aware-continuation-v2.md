# Additive R2 stage-aware continuation v2

2026-10-11. The first additive-bank R2 contract is preserved at step133635
after one smoke step and a stable 128-step block. Its end checkpoint SHA is
`bbc34cd22cca7b0403be82a3c96910c0b2971a8ef9b564cc448ae131166c77fd`.
It used 80% old / 20% new root draws and kept the old 392 roots byte-for-byte,
but sequential old-root ordering gave no stage-3 draw in this short block.
That is a sampling limitation, not a numerical failure. Do not edit or
resume the v1 contract for longer training.

V2 starts from this exact checkpoint without resetting model, cost objective,
auxiliary head, AdamW, Python/torch/MPS RNG or global step. It keeps the
independently audited additive bank SHA
`3a65df6530d8fe62d9e297b204072db4c1a7ec437f131338c640dc28c23a182e`
and audit report SHA
`47487fb089336eb424a34e79603fb510d61a0285868cea6a18d54b31ac7a2b57`.
Old-root draw fraction remains 80%; new roots get 20%. Within the old draws,
fixed 20-slot stage pattern is 5 stage0, 7 stage1, 7 stage2, 1 stage3.
Each stage cycles through every old F-usable root in that stage. This limits
stage-3 oversampling to 4% of all draws while ensuring that the hardest old
roots remain present. New draws round-robin through all seven new roots.
The schedule is deterministic in global step, not wall time or checkpoint.

Use the same smoothed exact-context pairwise F objective, 10% auxiliary and
20% known-solution/distance encoder-gradient ceilings; fail on zero/nonfinite
F gradient. Native MPS, no CPU fallback, full model/AdamW/RNG checkpoint at
every 128-step block. First perform a one-step smoke and bounded stability
block with independent SHA/optimizer/gradient audit. If healthy, continue
to 7,200 **new v2 active seconds** in serial blocks, preserving all
checkpoints and original v1 run. This is exploratory continuation on only
399 distinct F roots, not a 1k/30k-scale accomplishment. A fresh disjoint
DEV comparison must be frozen before promoting an endpoint. No TEST150 or
viewed DEV selection, no cloud purchase, unknotdb mutation/publication,
commit, merge or push.
