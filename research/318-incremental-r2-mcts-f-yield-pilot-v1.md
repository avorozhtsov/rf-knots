# New-root R2 MCTS F-yield pilot v1

Prospective TRAIN-only measurement for the 30k F-usable target in protocol317.
The additive roster-v2 SHA is
`93bbaa282ec79bd017b5c7adfbd4f55041adddebfe5505650d258d1f17ce58f8`.
Independent audit verified all 256 constructive witnesses, unique query
groups, TRAIN ownership at root/CC successors, heldout exclusions, input
SHA, and paired L10/L1000 caps. Candidate roots are not F labels.

Select 32 roots score-blind: all first eight stage-0 roots, then the first
four CC1 and first four CC2 roots in each of stages1–3 in SHA order.
For each root and both L10/L1000 run seeds20271317 and20271318,
64 **real** simulations each, 128 independent conditions total. Use the
frozen selected R2 v4 step114561 checkpoint SHA
`1ffe7346e0725fcb98125ebf03ab1faaaf7048985d69eeb00300e2828cca6946`
as policy plus its objective, canonical pgx-mcts-bench, distance prior,
four insertion candidates, widening batch4, c_puct1.5, temperature1,
original paired caps and tile cache. This avoids checkpoint choice using
viewed DEV. Pin all source/input SHA before writing results.

Write one resumable SHA-tracked shard per condition with full episodes,
ownership flags, CPU/wall and neural-call counts. Independently replay all
found solutions and verify start/caps/CC/moves/L10/L1000. Derive exact
F-preference groups only from fully replayed TRAIN-owned completed paths plus
the constructive reference, using the fixed margin2, at most3 near-best
positives and K4 strictly worse paths at identical state and remaining
resources. Report newly F-usable distinct roots and CPU per root separately
by stage, CC and objective. Bounded no-solve is censored, never a negative
action. Do not bulk-scale from 32 roots alone if hard yield is sparse.

This is search collection, not network training and not an unknotdb update.
The ongoing R2 optimizer contract remains unchanged. A future immutable
bank must retain all 392 old verified roots and add only audited new F roots.
