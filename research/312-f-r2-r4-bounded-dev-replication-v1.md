# Prospective bounded R2/R4 DEV replication after the first-panel tie

2026-10-10. The first matched DEV panel has been fully audited and is a tie
under protocol 307: each arm solved 14/32 distinct L1000 roots. This is a
separate, score-blind replication contract, frozen before constructing its
panel or running either endpoint on it. Preserve the first panel and all
failed and completed sources unchanged.

Use the same frozen R2 and R4 v4 endpoints at optimizer step 114561, with
their existing SHA-pinned checkpoint, AdamW/RNG and training manifests. Do
not train further or select a different checkpoint for this replication.

Construct exactly 16 short validation-owned roots of length 20..64 and 16
long validation-owned roots of length 129..192 from a new SHA-ranked
conjugate-witness candidate stream with seed 20261010 and indices 0..31999.
Require distinct query groups, full reference replay, paired L10/L1000
instances with the same resource caps, and exact nonterminal braid states
of length at least 20 disjoint from all 1000 TRAIN roster trajectories.
Exclude every query group already used in TRAIN, TEST, or any saved viewed
DEV panel, including `f-r2-r4-fresh-dev-panel-v1`. As in the earlier panel,
short shared tails and intermediate knot-type ownership are limitations;
do not claim stronger disjointness. If supply is insufficient, save the
failure and define a new contract before changing the stream or quotas.

Pin the new panel SHA, all endpoints and evaluator source before search.
Use canonical pgx-mcts-bench with 128 simulations, insertion candidates 4,
widening batch 4, c_puct 1.5, temperature 1, distance prior and cache on.
Run both arms on the same 32 roots, both L10/L1000 caps, and seeds
20271010 and 20271011: 256 tasks. Save one SHA-pinned shard per task with
full witness, CPU/wall seconds and neural evaluations. Independently audit
all shard SHA, starts, caps, costs and full replay; capped no-solve is
censored, not a negative action label.

Evaluate the replication panel on its own: the primary criterion is distinct
L1000 roots solved under either seed. A difference below three roots is a
tie. The candidate may solve at most two fewer distinct L10 roots, and may
not have higher CC on jointly solved roots. Report short/long, each seed,
paired wins/losses, CC, moves, objective cost and compute separately. If CPU
or neural evaluations differ by more than 10%, do not choose until a
separately pinned compute-matched comparison. Stop after this bounded
replication; if still tied, report no architecture winner rather than
repeatedly sampling panels. This remains an internal architecture test, not
evidence of superiority over non-neural heuristics. Do not use TEST150.
