# Additive F curriculum supply v1

The user requires continual expansion with new TRAIN-owned nodes and
representations, while retaining old verified examples. The current R2-only
optimizer contract is immutable and continues on the audited 392-root F bank.
This separate CPU supply run creates **candidates**, not F labels or optimizer
steps. It must not alter the live bank or the R2 continuation.

Use `conjugate_witness(20261011, index, max_strands=20, min_depth=8,
max_depth=60)` for SHA-ranked indices 0..15999. Fixed new-candidate quotas by
the existing four length/CC stages are [32,64,80,80]. Exclude all 1000
roster-v3 TRAIN query groups, legacy heldouts, TEST150, and every listed
viewed DEV panel. Require TRAIN owner for the start and each crossing-change
successor, a fully replayed constructive witness and paired L10/L1000
instances with the same original caps. Deduplicate by invariant query group.
Save source/input SHA and rejection counts. A quota shortage is a preserved
supply failure, not permission to silently relax ownership.

After independent roster audit, a separately frozen canonical pgx-mcts-bench
teacher must search a balanced sample across stages with independent seeds and
real budgets; replay all completed paths and compute exact F contexts/cost
diversity. Only verified F-usable new roots can be appended to an immutable
bank-v3 that retains the old 392 roots. A later optimizer version must pin
that bank and mix old/new root draws; no data hot-swap into the active run.
The same provenance and replay gates apply before staging any candidate for
unknotdb. No publication or database mutation at this stage.
