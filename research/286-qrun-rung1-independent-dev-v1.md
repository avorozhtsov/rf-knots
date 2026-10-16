# Independent DEV check for the fixed 1,024-step Q-run-like curriculum

2026-10-05. Freeze this before measuring rung-1 search. The rung-1
endpoint is the completed checkpoint SHA
`03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8`;
the mixed256 parent is SHA
`54ecbf5c7b3bb211d4dd42d0e8e254342e1748575000701d5600e5d81c2d6d43`.
No checkpoint selection on the already viewed v3 DEV panel.

Build a new, score-blind panel using seed 20261205 and SHA-ranked generator
indices 0..15999. Select 16 short roots (word length 20..64) and 16 long
roots (129..192), root validation ownership, distinct query groups, all
previous TRAIN/DEV/TEST exclusions, and explicit exclusion of all 32 v3
DEV roots. Require exact `(word,strands)` disjointness from the 1,000 TRAIN
reference paths for states of word length >=20. Common shorter tails and
crossing-change intermediate owner differences remain limitations. Replay
all constructive references, and pair L10/L1000 instances with identical
caps. If supply fails, preserve the failed contract, not a reduced quota.

On the audited panel, compare frozen F144, mixed256, and rung1-final with
canonical pgx-mcts-bench, exactly the same fixed continuation/tree settings
as the v3 comparison, 64 simulations per condition and seeds 20271205 and
20271206. This is 3 models ×32 roots ×2 objectives ×2 seeds =384 search
conditions. Preserve all shards and full replay. Report solved roots per
band/objective/seed, paired gains/losses and CC/moves/cost only on jointly
solved cases, CPU/wall and neural evaluations. Unsolved bounded searches
are censored, not negative training actions. Do not select another checkpoint
using this panel; any later replication needs a separately frozen panel.
TEST150 is excluded. No claim against non-neural heuristics without the
separate fair canonical control.
