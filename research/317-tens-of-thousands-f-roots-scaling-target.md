# Scaling target: tens of thousands of distinct F-usable TRAIN roots

2026-10-10. The user clarified that 1,000 F-usable roots is only an early
milestone. The research target is **30,000 distinct TRAIN-owned root query
groups with a verified nonzero exact-context F signal**, with 10,000 as an
intermediate scale gate. Milestones: 1,000, 3,000, 10,000, 30,000. Neither
candidate roots, repeated search seeds, L10/L1000 copies, cyclic views nor
duplicate invariant groups count more than once. Do not call a completed
search, candidate roster, or optimizer draw an additional F-usable root.

Starting evidence is 392 F-usable roots out of 1,000 roster-v3 candidates,
with stage counts [111,194,79,8] out of [140,286,287,287]. Overall yield
39.2% cannot safely be extrapolated to hard roots: stage-3 yield is 2.8%.
At the unchanged aggregate yield, 10,000 F roots would require about 25,510
candidates and 30,000 about 76,531; these are illustrative lower-confidence
planning ratios, **not** approved bulk budgets. Hard-stage scaling requires
better sources and/or verified search yield, not just more copies of the
current 16,000-index generator.

For each milestone use an append-only pipeline:

1. Generate new, distinct TRAIN-owned roots across length/CC strata from
   versioned source families. Keep old sources and root identities. Audit
   constructive reference replay, ownership at start and CC successors,
   exclusions, caps, and query-group deduplication.
2. On a small fixed balanced sample, measure canonical pgx-mcts-bench
   independent-search yield and CPU/neural work at real local budgets.
   Preserve full paths and replay each admitted solution. Bounded no-solve
   is censored and never a negative action. Require strictly different
   completed-path objective costs and exact state/resource divergence to
   admit an F preference.
3. Estimate candidate and CPU demand by stage from measured *new* F-usable
   roots per candidate and per CPU-hour, with uncertainty. Freeze each bulk
   tranche's sources, budgets, seeds, caps, and audit before launch. If hard
   yield remains near zero, redesign the TRAIN source/search before scaling.
4. Publish an immutable versioned TRAIN bank that retains every previous
   admitted root/group and adds only new verified groups. A new optimizer
   contract pins one bank version; never hot-swap an active run. Keep a
   prospective old-root rehearsal fraction (initially at least 25% of
   root-balanced draws) and stage-balanced sampling, then check retention
   on fresh disjoint DEV. Do not silently discard old examples merely
   because new data dominates in count.

The current R2 continuation uses the earlier 392-root bank until its frozen
7,200-second target; it is not the 10k/30k-scale training run. The new
incremental roster-v2 has 256 **candidate** roots pending independent audit
and MCTS F-yield measurement. Next immediate experiment: preregister a
balanced small teacher pilot on these new roots, not a blind 30k-run. A
larger source-family design must be validated before a multi-day bulk CPU
commitment. TEST150 and previously viewed DEV must not guide source or
checkpoint selection. Later unknotdb candidates need separate ownership,
deduplication, complete replay and strict CC/move cost checks before any
database mutation or publication. No cloud purchase, commit, merge or push.
