# Fixed Q-run equal-cost diversity pilot

2026-10-06. The full TRAIN teacher audit in research/291 found zero cheaper
solutions in 2,000 searches, but 813 verified, TRAIN-admitted paths that
are distinct from the constructive reference and have exactly equal
objective cost. Test whether these paths supply useful policy diversity;
do not treat them as better cost labels.

Before any optimizer step, build a TRAIN-only bank from the frozen 2,000
teacher shards and independently audited report SHA
`ee1049c12bdaca9142d13c2a8759e6bba91fe3c10d34815e1ddff3e58e1bbb86`.
For each root and objective, retain at most one alternative: among admitted,
fully replayed, equal-cost paths whose semantic action sequence differs
from the reference, choose the smallest SHA256 of the canonical witness
JSON. Keep the original reference for every root/objective. Re-verify the
selected witness, start, caps, equal objective cost and TRAIN ownership;
pin the resulting bank SHA. No DEV or TEST data may enter this bank.

Train two independent arms from the same frozen curriculum1024 checkpoint
SHA `03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8`:

- A: reference-only verified-solution imitation.
- B: the same sampling schedule, but choose the equal-cost alternative with
  probability 1/2 when the selected root/objective has one; otherwise use
  the reference. Always retain reference rehearsal. Both arms consume the
  same RNG draw for this choice, so later root/objective sampling remains
  aligned. An unchosen action is not labelled bad.

Use the existing raster policy, frozen metric objective, policy-only AdamW,
native MPS without fallback, 320 steps per arm, batch four distinct stages
(one root from each stage per step), LR 5e-5, decay .01, clip 1, seed
20261006, checkpoints every 32 steps with optimizer and all RNG states.
Root selection is deterministic stage-wise round robin. Sample objective
and witness position from separate saved RNG streams. The candidate pool is
the same four proposal actions plus the verified target; loss is mean
negative log probability of the target. Equal steps/batches and measured
CPU/MPS time are required; source, bank, checkpoint, and contract SHA are
frozen before the first step. Complete A before B on one MPS device to
avoid GPU contention; no DEV-based checkpoint choice inside either arm.

Before inspecting either arm's search result, construct a new score-blind
DEV panel using seed 20261209 and SHA-ranked candidate generation: 16 short
validation-owned roots of length 20–64 and 16 long of length 129–192,
paired L10/L1000 resource caps, reference replay, distinct root groups,
excluding TRAIN and both prior viewed Q-run DEV panels. Require exact
nonterminal trajectory-state disjointness from TRAIN for word length >=20;
common shorter tails remain an explicit limitation. The panel is fixed
before evaluation. Search curriculum1024 parent, A320 and B320 on the
same 32 roots, both objectives, two seeds 20271209/20271210, 64
simulations and identical canonical pgx-mcts-bench resources: 384 tasks.
Audit all shard SHA and replay; report solved attempts and distinct roots,
paired gains/losses, CC and moves only on joint successes, L10/L1000,
short/long, CPU/wall and neural evaluations. Capped no-solve is censored.

This is an exploratory pilot, not evidence of superiority. A larger CPU
investment requires a reproducible B-over-A solve gain on both seeds,
without a long-root regression or worse paired objective cost, followed
by a fresh independent DEV replication. The already viewed DEV panels and
TEST150 cannot select checkpoints. Fully non-neural matched search is still
required before any superiority claim. This pilot is not F-dominant R2/R4
and does not start its 24-hour timer.
