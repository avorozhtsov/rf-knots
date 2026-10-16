# Additive F roster v2: distinguish TRAIN reuse from heldout leakage

The v1 incremental roster source and failed run are preserved. It found
0/256 candidates because it placed all old TRAIN roster root query groups in
the exclusion set for *crossing-change successors*. Reusing an old TRAIN
invariant at an intermediate state is allowed; it is not a DEV/TEST leak.
Inspection of generated examples confirmed new start groups with TRAIN-owned
CC successors, but many such successors shared an old TRAIN root group.

V2 uses two sets: `seen_roots` for uniqueness of new starts, containing all
old 1000 TRAIN roots, legacy heldouts, viewed DEV roots and newly selected
roots; and `forbidden_states` for CC successors, containing only legacy
heldouts and viewed DEV roots. Start and each CC successor must be TRAIN-owned.
No DEV/TEST condition is weakened. Supply for the easy stage is nearly
exhausted by the old roster; fixed new quotas are [8,48,96,104], total256,
with seed20261011, SHA-ranked indices0..15999 and unchanged generator/caps.
On shortage preserve the failure and make another separate contract.

These are only constructive candidates. An independent audit and canonical
pgx search/replay/F-yield measurement are required before adding any gradient
groups to a new immutable bank that retains all 392 old F roots.
