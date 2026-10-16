# Protocol 70: initial-model search completed and replay-audited

2026-09-21. This is a completed **single-arm** result, not the final
69/70/74 comparison and not evidence of an embedding improvement.
The ongoing protocol 97 training/evaluation contract is unchanged.

## Evidence and coverage

Directory: `artifacts/graph-mixture-evaluation-v1/mcts/initial/`.

- Final manifest SHA256:
  `3abbef1378bb46186ea69f92bd99af0b18a94dd6bedfd7fde5a25eabe8b238e4`.
- Passed replay-audit SHA256:
  `1a13ef32da46bc4f62ac5c76720d49228ab14513d9c82b6b8fb9cf5ee14a0714`.
- All 128 conditions: 64 fixed query roots at each of L10 and L1000.
- All 766 episodes replayed, including unsuccessful budget-limited episodes:
  375,975 full-path moves and 375,332 continuation moves.
- Both completed jobs were reverified against the controller's committed
  records using `run_graph_mixture_evaluation.verify_job`.
- Saved shard hashes/ordered coverage were rechecked with `committed_rows`.
  Reference and solution costs were independently reconstructed using
  `summarize_graph_mixture_experiment.certify_costs` for this complete arm.

The model is the trained RF protocol-57 starting checkpoint of the paired
graph-mixture experiment. It is neither a random model nor a Q-Run solver.
The search uses the previously fixed protocol-70 panel, resources and seeds;
no learned value head or oracle incumbent is used.

## Confirmed outcomes

| Objective | Solved conditions | Continuations | Stopped at move cap | Solved continuations |
| --- | ---: | ---: | ---: | ---: |
| L10 | 3 / 64 | 382 | 379 | 3 |
| L1000 | 2 / 64 | 384 | 382 | 2 |

The five conditions involve five distinct panel roots. Their witnesses all
use one crossing change. Here `L_lambda = moves + lambda * CC`.

| Panel index | Objective | CC | Moves | Cost | Certified optimum | Excess cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | L10 | 1 | 92 | 102 | 100 | 2 |
| 34 | L1000 | 1 | 109 | 1109 | 1107 | 2 |
| 58 | L10 | 1 | 60 | 70 | 68 | 2 |
| 61 | L1000 | 1 | 85 | 1085 | 1085 | 0 |
| 63 | L10 | 1 | 61 | 71 | 69 | 2 |

For these five roots the saved reference attains the certified semantic
lower bound. Consequently the excess column is a certified regret, not just
a gap to an arbitrary witness. Index 61 reaches the lower bound exactly.
Each other solution has one insertion or stabilization accounting for the
two extra semantic moves. These optimality claims are within the declared
action system and cost, not arbitrary knot-diagram operations.

## Interpretation and next step

This is still a low success rate under the fixed search budget. The many
move-cap stops do not prove that the states or individual actions are bad,
nor isolate whether policy quality, exploration or budget caused failure.
The old 1/64 test had different roots/resources; these counts are not a
controlled improvement over that older result.

The controller accepted the completed search and its replay audit, then
launched the `graph` arm (PID 21872 at observation, with its own new manifest).
The `control` and `uniform` arms and their audits remain required. Do not
promote a model or generate the final combined summary from this single arm.
The separately running protocol-97 raster transfer experiment is also still
incomplete. Its test panels are shared evaluations, not fresh replications;
this observation must not be used to alter its frozen training schedule.
