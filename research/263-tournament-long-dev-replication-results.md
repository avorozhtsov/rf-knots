# Long DEV replication of tournament policy arms A/F/G

2026-10-03. The predeclared protocol is `research/262-tournament-long-dev-replication-v1.md`.
The result manifest is `artifacts/mcts-tournament-long-dev-v1/manifest.json`,
SHA256 `ea0de4013c669f908d7d53b81bb02d3f68609b50459fbb199bed7df0019d850b`.
Its frozen contract SHA256 is
`ea4f17c9c30046cbacc119f0b3d69d215802a52ffb1f664c490d1d481a8d6ebb`.
The independently generated panel SHA256 is
`1c186fa81da4e755a1a46906971ed1067d38efbd44560984f6be7ba7b0764ced`.

All 192/192 shards have matching SHA256 and contiguous task identity; all
five contract-input SHA256 values matched. Every one of the 115 found
solutions passed independent complete witness replay, start/resource checks,
and exact L10/L1000 recomputation. The finished worker is no longer live.
The 16 validation-owned constructive representations have word lengths
74–137; seven reference solutions use two CC. The two search seeds on a
representation are *correlated repeated attempts*, not 32 independent roots.

| Arm | L10 solved/32 | L1000 solved/32 | Total/64 | CPU seconds | Neural evaluations |
| --- | ---: | ---: | ---: | ---: | ---: |
| A control | 24 | 15 | 39 | 3634.1 | 454313 |
| F near-best positives | 23 | 20 | 43 | 3628.3 | 449370 |
| G extra high-CC | 20 | 13 | 33 | 3695.9 | 455696 |

F versus A on identical representation/objective/seed conditions:

| Objective | Both solve | Only F solves | Only A solves | Neither | F lower/equal/higher cost when both solve |
| --- | ---: | ---: | ---: | ---: | ---: |
| L10 | 22 | 1 | 2 | 7 | 2/18/2 |
| L1000 | 15 | 5 | 0 | 12 | 1/14/0 |

At L10 the number of distinct representations solved at least once is A=13,
F=12, G=11 out of 16; at L1000 it is A=8, F=10, G=7. F's five L1000-only
successful attempts involve three representations, two of which A missed
with both seeds. All five F-only L1000 paths used one CC; this is improved
bounded-search success, **not** demonstrated lower CC on common successes.
Every F/A common success had equal CC. Its only L1000 cost improvement on a
common success saved ten moves. F's measured CPU and neural work were close
to A, but equal simulations do not imply exactly equal neural evaluations.

F beat G on success fraction and on common-success cost under both objectives;
G's extra CC signal had only 17 contributing TRAIN groups. The long panel
does not support advancing G. F is the best *candidate* for L1000 from this
stage, not a validated winner across both objectives: at L10 it is slightly
behind A, its L1000 gain comes from just three representations, and the two
seeds do not make those independent. Do not invest a large CPU budget in F
on this evidence alone. A new predeclared, disjoint DEV replication with
more independent roots or a different search budget would be needed to
confirm the L1000 signal. No TEST150 outcome was used for selection.

This experiment tests rollout-tournament **policy** training, not the user's
AlphaZero-style multi-output value-guided MCTS. The separate value-head
bootstrap and calibration gate in protocol256 is the next architectural
priority. Neither this result nor a uniform tree prior is a fully non-neural
heuristic comparison; superiority needs the prospective canonical
`pgx-mcts-bench` control, matched resources and replay.
