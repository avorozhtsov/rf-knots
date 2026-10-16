# Protocol 70: graph-mixture search completed and replay-audited

2026-09-21. This records the completed `graph` arm, not the final
69/70/74 comparison. Protocol 97 remains frozen and is not retuned on these
TEST outcomes.

## Evidence

Directory: `artifacts/graph-mixture-evaluation-v1/mcts/graph/`.

- Final manifest SHA256:
  `b9b99489d9bed9a3ae1525667661a2d6ac9f2bd4d428e0b3fc7c00678014e4d6`.
- Passed replay-audit SHA256:
  `586f694d750278cfe512e63ccb8a617bdfb495aa39056fea42850d164d6b6db6`.
- All 128 conditions, 766 episodes, 376,409 continuation moves and 377,049
  full-path moves replayed. The recorded action-score count is 66,541,071.
- Both committed jobs were independently reverified with
  `run_graph_mixture_evaluation.verify_job`; all 16 shard hashes and ordered
  coverage were checked with `committed_rows`.
- References and successful witnesses were independently reconstructed with
  `summarize_graph_mixture_experiment.certify_costs` for this complete arm.
  This did not bypass the incomplete-matrix gate or create a final summary.

The model is the 4096-update graph-mixture descendant of the RF protocol-57
initial checkpoint, not the Q254 raster-transfer model. No learned value head
is used in this search. The audit checks recorded resources, RNG context and
costs; timings and tree-work counters are not independently rerun.

## Confirmed outcomes

| Objective | Graph-mixture successes | Initial-model successes (report 104) |
| --- | ---: | ---: |
| L10 | 2 / 64 | 3 / 64 |
| L1000 | 1 / 64 | 2 / 64 |

These arms share the fixed panel, search settings and seeds. The graph arm
does not improve the aggregate success count in this evaluation. Three
successes are too few for a broad generalization claim; the matched trained
`control` arm and `uniform` search are still required.

Here `L_lambda = moves + lambda * CC`. All three graph-arm solutions use one
crossing change. Each corresponding reference attains the certified semantic
cost floor, so the last column is certified regret within the declared action
system, not merely a gap to an arbitrary reference.

| Panel index | Objective | CC | Moves | Cost | Certified optimum | Excess cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | L10 | 1 | 273 | 283 | 101 | 182 |
| 32 | L1000 | 1 | 37 | 1037 | 1035 | 2 |
| 55 | L10 | 1 | 427 | 437 | 117 | 320 |

None reaches the certified optimum. The two long L10 solutions contain many
insertions/stabilizations: their excesses decompose as `2*88 + 6 = 182` and
`2*157 + 6 = 320`, respectively; the final six moves in each expression count
commutations, not a chronological suffix. The L1000 solution has one
stabilization, accounting for two excess moves. This describes successful
paths, not the cause of the many censored search failures.

## Interpretation and continuation

Together with the weak distance results in report 103, this supplies no
evidence to promote the graph-mixture model. It does not prove that embeddings
are useless, that individual failed actions are bad, or that larger search
budgets cannot help. No database discovery or production replacement is claimed.

The controller accepted the replay audit and started `mcts-control` (PID 57352
at observation). Its full search/audit and the uniform search/audit remain.
Only after the complete prescribed matrix may the combined summary be emitted.

Separately, protocol-97 Q254-local training completed 2048 updates. Its final
checkpoint SHA256 is
`c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f`.
The single-arm integrity check verified exact initialization, final/progress
hash agreement, finite weights/optimizer state, 2048 optimizer steps, 65,536
metric draws, 16,384 positive-state draws and 128 full relative-group updates
covering 16,766 states. All 1143 input pins and 67 source pins matched.
This is engineering integrity, not independent quality evidence. The paired
random-local training is active; paired audit and independent evaluations
remain incomplete. Neither training nor the TEST contract was changed here.
