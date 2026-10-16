# Validation-panel supply correction after strict ownership failure

2026-10-04. The prospective v1 builder exited 1 after all 16,000 ranked
candidate indices with **0/16 short and 0/16 long** roots. Rejections:
length/filled 9,626; duplicate/excluded 4,110; ownership 2,264. The
requirement that every crossing-change successor's invariant-owner be
validation made the panel empty. Preserve the v1 source/protocol and this
failure; do not claim a v1 DEV result or silently relax its contract.

For a separate v2 panel, freeze the same seed 20261204, ranked indices
0..15999, 16 roots length 20..64 and 16 roots length 129..192, and all
previous root-query exclusions. Require each **root** to be validation-owned.
Also exclude any candidate whose exact nonterminal braid state `(word,
strands)` appears along any of the 1,000 complete TRAIN reference witnesses;
exclude the universal solved empty state from this comparison. Replay all
reference witnesses and verify distinct roots and identical L10/L1000 caps.
This tests unseen roots and exact states, but does **not** claim that every
intermediate knot-type invariant is validation-owned: crossing changes can
move to another ownership bucket. Report this limitation explicitly.

Do not use TEST150 for selection. If the fixed v2 supply fails, preserve
that failure and redesign the test; do not weaken criteria in place. The
subsequent matched search is still the predeclared three-arm, 64-simulation
comparison from protocol281, now on this separately fixed v2 panel.
