# Q-run-like curriculum rung-1 independent DEV results

2026-10-05. The fixed new DEV panel SHA
`2e4ed78a89769117cac3e1fba8e4583c2079603ad0a749387725bf4378f60a11`
contains 16 short and 16 long validation-owned roots, excludes TRAIN and
the viewed v3 panel, and passed replay, paired-resource and exact-state
disjointness checks for word length >=20. Common shorter tails remain.

The corrected matched-search contract SHA
`c53e57748caa75e3003fc1697e63d451a93da955ade7263cb0a6df9f3e0c9991`
completed 384/384 conditions: F144, mixed256 and curriculum1024, both
objectives, two seeds, 64 simulations. Manifest SHA
`4b2beef647962769906c495d96994c3e020dda60f9b9e3110ead51a3465f436e`.
All shard SHA matched; an independent audit replayed all 236 solutions and
checked starts, caps, CC/moves and objective costs.

Distinct roots solved in at least one of the two seeds:

| Model | Short L10 | Short L1000 | Long L10 | Long L1000 |
| --- | ---: | ---: | ---: | ---: |
| F144 | 13/16 | 13/16 | 6/16 | 6/16 |
| mixed256 | 16/16 | 15/16 | 7/16 | 7/16 |
| curriculum1024 | 15/16 | 15/16 | 9/16 | 7/16 |

Curriculum1024 versus F144 by seed: short L10 solved 15 vs12 and 14 vs13;
short L1000 14 vs13 and 13 vs13; long L10 4 vs6 and 9 vs6; long L1000
5 vs6 and 6 vs5. Thus the short improvement is fairly consistent, while
the apparent long-L10 benefit is seed-dependent. Against mixed256, the
curriculum endpoint loses one distinct short-L10 root, ties short-L1000
and long-L1000, and gains two distinct long-L10 roots. This is a useful
but modest signal, not a decisive winner and not evidence against fully
non-neural search. Bounded failures remain censored.

The next Q-run-like step should add verified on-policy solutions rather
than repeat the same 1,000 reference witnesses indefinitely. Freeze a
separate TRAIN-only teacher collection from curriculum1024, then distil
only fully replayed solutions while retaining reference-witness rehearsal.
Do not use this viewed DEV panel for checkpoint selection. TEST150 remains
excluded.
