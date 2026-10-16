# Idea-(3) tournament pilot readiness and next decision

2026-10-02, after the first full 16-simulation seed and while the remaining
nine independent seeds are collecting. This is not a model-training result.

## Verified foundation

The immutable v3 teacher contract covers all 128 TRAIN representations at
L10 and L1000. The independent through-256 audit replayed every episode,
rechecked TRAIN ownership and shard SHA, and found 101 root/lambda groups
with different admitted solution costs and 132 exact state/resource
divergence contexts. The design minima of protocol254 (16 groups, 64
contexts) are met, but the agreed S=10 collection is not yet finished.

The eight idea-(3) signal forms B-I plus control A are implemented in
`scripts/mcts_tournament_signals_v1.py`. The implementation takes only
complete replay-verified witnesses, cancels identical state/resource/action
terms and never calls a censored attempt a losing action. Six unit tests and
one real saved-group construction smoke pass. This is target construction,
not eight trained candidates. Their comparison still needs a pinned
TRAIN-only sampling/mixing contract, matched training and DEV search.

## Data-efficiency warning to revisit after S=10

For the first five complete seeds (1,280 committed attempts), the saved
ownership flags give 124 of 256 root/lambda groups with different costs.
This later slice has **not yet received the independent full replay audit**.
At a provisional near-best margin of two objective units, only 64 groups
would supply F with both multiple positives and a strictly worse loser.
Only 11 groups have distinct CC counts, limiting the extra CC objective in
G. These are TRAIN-only adequacy diagnostics, not variant rankings or
evidence of downstream utility. Do not let raw episode multiplicity make
the effective number of distinct root groups look larger.

## Next decision, not an automatic promotion

1. Let the healthy v3 S=10 worker finish without a duplicate. Verify all
   shard hashes, exact replay, TRAIN ownership, distinct roots/paths/costs
   and branch contexts separately for L10 and L1000.
2. Pin K, near-best margin, cost-gap scale/cap, shared root-group draw
   schedule, distance/demo/tournament gradient mix, steps/seeds, reference
   policy and compute accounting from TRAIN only. Run an equal short
   nine-arm pilot (A control, B-I eight variants); do not select on TEST150.
3. Compare on a fixed excluded DEV panel at matched simulations, neural
   evaluations and reported CPU/wall time. Report solved starts, CC/moves,
   L10/L1000 and replay. At most two candidates advance to replication; if
   uncertainty is large, report inconclusive.
4. Current v3 paths come from terminal/rollout MCTS using an old policy and
   no calibrated multi-output value head. Any pilot trained from these
   paths is a **rollout-tournament screening**, not a completed test of the
   requested AlphaZero-style value-guided architecture. Separately train
   and calibrate that head before making an architecture claim or collecting
   final value-guided tournament data.

Do not mutate the frozen v3 teacher/search source, 248 checkpoints or
completed earlier artifacts. No cloud, publication, commit, merge or push.
