# R2-only F continuation v1

Decision: use R2 for the next exploratory branch. The two completed matched DEV
panels did not establish a statistically or preregisteredly superior architecture;
the second panel's L1000 16/32 versus 14/32 is a practical tie-break requested by
the user, not a claim of superiority. Frozen R4 is retained, not trained further.

This is a separate optimizer run from the R2 v4 endpoint at global step 114561.
It preserves model, cost objective, auxiliary head, AdamW, Python/torch/MPS RNG,
global step, and stage draws. The endpoint SHA is
`1ffe7346e0725fcb98125ebf03ab1faaaf7048985d69eeb00300e2828cca6946`.
No v4 manifest or checkpoint is edited. The TRAIN-only audited joint F bank is
unchanged: 392 distinct F-usable roots / 690 exact groups, stage coverage
111/194/79/8. This continuation does **not** constitute training on 1000 F-usable
roots, and cannot establish hard-stage generalization.

The v3 smoothed exact-context pairwise loss is the primary encoder gradient.
CC/moves auxiliary and known-solution/distance rehearsal retain the v4 gradient
norm ceilings of 10% and 20% of the F encoder norm. A zero/nonfinite F gradient
halts; no auxiliary-only update is allowed. Root/group order, optimizer
hyperparameters, and global-step sampling continue unchanged. Every training
block is 128 optimizer steps and ends in a full SHA-tracked checkpoint with
optimizer and RNG. New wall/CPU seconds are tracked separately from the parent.

Operational plan: verify one-step migration smoke, then run serial 128-step
blocks on the single R2 MPS worker until 7200 new active seconds, or stop sooner
for a documented numerical, storage, or data-quality failure. This two-hour
bounded continuation is the first rung of ongoing training, not the promised
1000-root scale-up. Do not choose a checkpoint using the already viewed DEV
panels or TEST150. Before any later promotion, freeze a fresh independent DEV
comparison and a new TRAIN data contract with substantially more hard F groups.

Parallel scientific data plan: use canonical pgx-mcts-bench to collect independent
TRAIN searches on hard roots with real local budgets; replay all completed paths,
compare separate CC/moves and L10/L1000 cost to constructive references, and
admit only verified completed solutions. Capped no-solve is censored, never a
negative action. New MCTS-derived F groups may enter gradients only in a new
versioned bank and optimizer contract, not by silently changing this run.
Likewise, candidate unknotdb additions must be staged with ownership,
deduplication, replay, and strict cost checks before modifying or publishing a
database. No cloud purchases, publication, commit, merge, or push.
