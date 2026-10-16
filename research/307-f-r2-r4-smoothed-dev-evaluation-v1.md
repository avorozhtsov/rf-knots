# Prospective matched DEV evaluation for the smoothed R2/R4 pilot

2026-10-07. This protocol is frozen before examining any R2/R4 v3 search
result. Use only the score-blind, independently audited 32-root panel
`artifacts/f-r2-r4-fresh-dev-panel-v1/panel.json` SHA
`2465a02e8d32a60c6d6769d92865e895b9663151122710f029562b978b726a30`.
The fixed evaluator is `scripts/evaluate_joint_f_r2_r4_v3.py` SHA
`be9d412f8f22a73c2ed97796dd768ffefff28ba27b130676ce553b2bce0bba19`.
Evaluate the fixed R2 and R4 v3 endpoints at the same optimizer step after
each has at least 43,200 seconds of checkpointed active native-MPS training.
The common step is the first completed paired scheduler step that meets both
active-time thresholds; no selection by TRAIN loss, prior DEV or TEST150.
Pin both checkpoint SHA and training manifests in the evaluation contract.

Use canonical pgx-mcts-bench, 128 simulations, insertion candidates 4,
widening batch 4, c_puct 1.5, temperature 1, distance prior and cache on.
Run each arm on the same 32 roots, both L10/L1000 resource instances and
the same two fixed seeds 20271008 and 20271009: 256 tasks total. Save one
SHA-pinned shard per task, full found witness and resource counters, CPU/wall
seconds and neural evaluations. Independently replay all successes and audit
caps, CC, moves, cost and paired starts. Capped no-solve is censored, not a
negative action label.

Primary decision: number of distinct L1000 roots solved by at least one
seed. L10 distinct solved may be at most two fewer than the other arm; on
jointly solved roots there must be no CC regression. Report each seed,
short/long band, CC/moves/cost, CPU, wall and neural evaluations separately.
If CPU/neural work differs by >10%, require a separately preregistered
compute-matched rerun before selection. Fewer than three distinct-root
L1000 gain is a tie. This is an internal architecture-choice DEV check, not
a claim against fully non-neural heuristics or general superiority.
