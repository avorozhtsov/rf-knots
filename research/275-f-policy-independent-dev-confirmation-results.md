# F versus A: independent long DEV confirmation is inconclusive

2026-10-04. Protocol272 completed 256/256 search shards. Manifest SHA256
`fe56e8e9a513aff1357b4a47178e5f3457a7d1d698fe640f4939646fad47de0a`;
independent auditor `scripts/audit_f_policy_confirmation_v1.py` SHA256
`aac00c7ac60bbc075542e822ce908e0cb25204479fc95c1e24d771d466a166ad`
exited 0. It checked all shard/input SHA values, 32 distinct validation-owned
and excluded-from-earlier-panel constructive witnesses, their replay, and
every found solution's full replay, start, resources and objective cost.
The fixed panel had 10 CC2 reference witnesses. Neither worker remains live.

There are 32 independent root representations, two search seeds per root,
and L10/L1000 separately. Repeated seeds are correlated attempts, not new
roots. At least one seed solved the root as follows:

| Objective | A roots solved | F roots solved | only A | only F | neither |
| --- | ---: | ---: | ---: | ---: | ---: |
| L10 | 20/32 | 19/32 | 1 | 0 | 12 |
| L1000 | 16/32 | 18/32 | 0 | 2 | 14 |

Seed-level attempts: L10 A=35/64, F=35/64; L1000 A=31/64, F=33/64.
On matched seeds solved by both, F had lower/equal/higher objective cost
3/23/8 at L10 and 1/22/6 at L1000. Every common success used the same
number of crossing changes, so these cost differences are moves. Total
CPU seconds A=6248.1, F=6214.8; neural evaluations A=861546, F=868943.
Search compute was close but not identical.

The **predeclared F promotion criterion fails**: L1000 gained only two
different roots, not the required four. F also lost one L10 root and had
more expensive common solutions more often than cheaper ones. The result
does not support a claim that F is better than control A, or a large CPU
investment *on the strength of this DEV comparison*. The user's newer
instruction to explore a long F-dominant curriculum is a separate research
priority; pursuing it is exploratory method development, not promotion or
validation of the 144-step F checkpoint. Preserve both conclusions.

The bounded failures are censored. TEST150 was not used for selection.
No fully non-neural matched heuristic control has been run here.
