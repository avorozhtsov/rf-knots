# F policy: independent long DEV confirmation against A

Frozen before either actor is evaluated on this panel, 2026-10-04.

The prior 16-root long DEV screen (protocol262) suggested an L1000 advantage
for F over control A, but its five F-only successes came from only three
distinct roots; F was slightly behind at L10. F is a rollout-tournament
**policy** intervention, not the failed bounded-success value head. This is a
bounded confirmation expenditure, not a large new training run and not a
claim of heuristic superiority.

Generate 32 new validation-owned, distinct invariant query groups using
`conjugate_witness` indices 0..2047, seed 20261119, max_strands=20,
min_depth=20, max_depth=60, ordered by SHA256 of `20261119:index`.
Require start word length 60..150, fully replayed constructive witness,
and at least eight CC=2 references. Exclude the long-general corpus groups,
the short DEV panel, the protocol262 long DEV panel and TEST150 groups.
If the fixed rule cannot supply 32 eligible groups or eight CC=2 references,
stop without altering selection. The constructive witness fixes identical
caps for A/F; it is never supplied to search.

Use the saved SHA-checked A and F checkpoints at 144 optimizer steps.
For each root, separately evaluate L10 and L1000 with seeds 20271119 and
20271120, 128 simulations, insertion_candidates=4, widening_batch=4,
c_puct=1.5, temperature=1.0, distance tree prior, cached CPU float32
inference, identical horizon/caps. This gives 256 paired conditions.
No training or checkpoint selection occurs during this evaluation. Preserve
every result shard, timing, neural-evaluation count and SHA; full witness
replay is mandatory. Missing solutions are budget-censored.

Primary descriptive comparison is the **number of distinct roots** solved
at least once by each actor at each objective, with paired root-level
only-F/only-A counts; seed-level counts are secondary because two seeds on
one root are correlated. Report objective cost, CC and moves on common
successes, as well as total CPU and neural evaluations. F qualifies for a
larger CPU training/collection budget only if on this new panel it solves at
least four more distinct L1000 roots than A, solves no more than two fewer
L10 roots, has no higher CC on common successes, and uses at most 1.10x
the control's total CPU and neural evaluations. Otherwise the result is
inconclusive or negative and the next priority is data diversity and the
separate multi-output value architecture, not prolonged F training.

This panel is DEV, not TEST. TEST150 cannot select a model. A later new
independent TEST and a fully non-neural canonical pgx-mcts-bench control at
matched resources are required before a superiority claim. Do not change
the prior SHA-pinned experiments. No cloud purchase, publication, commit,
merge or push.
