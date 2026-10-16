# R2/R4 bounded DEV replication: no architecture winner

2026-10-10. Protocol `312-f-r2-r4-bounded-dev-replication-v1.md` was
frozen before the new panel and search. The panel SHA is
`dd7c2d517514b6efbfd77e7f578fcf5b9088be286271e7cecd0f46045beda2de`:
16 short and 16 long validation-owned distinct roots. The independent panel
audit replayed all references and checked paired caps, ownership, old-panel
and TRAIN exclusions, and substantial-state disjointness as specified.

The fixed step-114561 R2 and R4 endpoints each ran both L10/L1000 instances
with two matched seeds and 128 canonical pgx-mcts-bench simulations: 256/256
conditions. Evaluation manifest SHA:
`002e92d78d7a90cfc7456638f88d6250687a2fc4ac84dcb9adf2e53fb91cf17f`.
Independent report SHA:
`e04a44d45ba7f602b3cbac4d1f79534caabfe774896ceecee2bed04088c151dd`.
All 256 shard and pinned-input SHA passed, and all 104 found solutions were
fully replayed with start, caps, CC, moves and both costs checked. The
evaluator terminated normally; no resumed duplicate remains.

Distinct roots solved by at least one seed:

| Arm | L10 short | L10 long | L1000 short | L1000 long |
| --- | ---: | ---: | ---: | ---: |
| R2 | 15/16 | 0/16 | 16/16 | 0/16 |
| R4 | 15/16 | 0/16 | 14/16 | 0/16 |

Thus primary L1000 is R2 16/32 versus R4 14/32. The two-root difference
is below the preregistered three-root threshold: **tie**, not R2 promotion.
L10 is 15/32 for each. No R4 crossing-change regression occurred on
same-seed joint successes. Across the 128 conditions per arm, R2 used
8487.71 CPU seconds and 831391 policy neural evaluations; R4 used 8529.86
CPU seconds (+0.50%) and 825892 neural evaluations (-0.66%). The compute
gap is below the 10% rerun threshold. Full per-seed and paired details are
in `artifacts/f-r2-r4-bounded-replication-audit-v1/report.json`.

The first independent panel was also a tie (L1000 14/32 each). Neither
panel establishes useful long-root generalization: this replication solved
zero long roots for either arm, and the first panel only one long L1000 root
for R2. Stop further DEV-panel resampling under protocol312. Do not select a
winner or give either arm a larger CPU budget based on these results.
Unsolved capped searches remain censored, not action negatives. The TRAIN F
bank has only 392 distinct F-usable roots, including eight in the hardest
stage; the 1000-root scale-up requirement is not met. A new TRAIN-only
hard-stage data/signal experiment needs its own prospective contract and
measured CPU yield. TEST150 remains unavailable for selection. Claims of
superiority over heuristics require a separately preregistered fully
non-neural matched control with full replay.
