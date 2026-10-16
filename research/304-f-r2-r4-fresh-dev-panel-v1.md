# Prospective R2/R4 architecture-choice DEV panel

2026-10-07. Freeze this selection before viewing any R2/R4 search outcome.
Generate 16 short (start-word length 20..64) and 16 long (129..192)
distinct validation-owned query groups with seed 20261009, SHA-ranked over
indices 0..15999. Require fully replayed constructive references, paired
L10/L1000 resource caps, and exact nonterminal braid-state disjointness at
word length >=20 from all 1000 TRAIN roster witnesses. Exclude every saved
TRAIN query group and all roots in the five existing viewed DEV panels.
Shorter common tails and intermediate knot-type ownership remain limitations.

The panel is score-blind: no model inference or search outcome enters
selection. Save panel, source, roster and exclusion-panel hashes. If supply
fails, retain the failed attempt and revise in a new version; do not relax
criteria silently. Independently audit reference replay, ownership, exclusions,
paired caps and SHA before search evaluation.

At the matched 24-active-hour R2/R4 checkpoint, use canonical pgx-mcts-bench,
fixed seeds and 128 simulations for both arms on the same panel. Primary
criterion and guardrails are in research/273. TEST150 is not eligible.
