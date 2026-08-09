# Semantic reporting / native reward split bug

Detected on 2026-08-09 during the from-scratch scientist preflight.

The serial wrapper correctly counted and reported portable semantic moves, and its
remaining-L feature used that count. However, terminal self-play rewards were
inherited from the underlying environment, whose `used` counter includes internal
head shifts and memory operations. Training therefore optimized
`A*cc + native_plies` while evaluation reported `A*cc + semantic_moves`.

This invalidates the preflight as learning evidence. It nevertheless exposed the
split because repeated training produced stable solve rates but inconsistent
objective behavior: `strand-graph` reached zero crossing changes, `window-local`
stayed at 0.406, and `raster-axial` improved L10 while worsening L1000.

The repair recomputes every serial terminal reward from crossing changes and the
wrapper's semantic-move counter. Internal operations still consume the native
episode clock. Regression tests compare identical terminal witnesses with and
without extra full-cycle head shifts and check the exact ratio-weighted payoff.

The generated preflight checkpoints were removed rather than admitted or retained
as starting weights. All semantic-v1 pretraining must restart from scratch after
this repair.
