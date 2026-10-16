# Independent DEV panel for the first Q-run-like policy bootstrap

2026-10-04. Freeze before inspecting the fixed 256-step outcomes. Compare
the unchanged F144 parent with two distinct 256-step training recipes:
short-only v1 and 75/25 short/length-stratified v2. Both are exploratory
verified-solution imitation, not F-dominant R2/R4 or AlphaZero value MCTS.

Select 32 distinct validation-owned constructive representations by
SHA-ranked seed 20261204, generator indices 0..15999: 16 short with
word length 20..64 and 16 long with length 129..192. Every root and
crossing-change successor on the reference witness must be validation-owned;
exclude all saved TRAIN, DEV and TEST query groups, including the new
1000-candidate TRAIN roster and earlier DEV panels. Replay every witness.
Each root has paired L10/L1000 instances with identical resource caps.
The builder may fail closed if the prospective supply is inadequate; do not
lower quotas after seeing scores. This panel is never training data.

Evaluate all three checkpoints after SHA/replay/optimizer audits on exactly
the same panel, one fixed search seed 20271204 and 64 simulations per
root/objective, canonical pgx-mcts-bench, fixed tree/continuation settings.
Report per length band and objective: distinct solved roots, paired
solutions, crossing changes, moves, objective costs, full replay,
CPU/wall/neural evaluations. No winner from imitation loss alone, and no
claim of superiority over fully non-neural heuristics. If this small panel
is inconclusive, replicate under a separately frozen contract instead of
selecting by an old viewed TEST. An unsolved capped search is censored.
