# Storage-safe continuation of matched R2/R4 smoothed F training

2026-10-08. The user approved reducing checkpoint retention and deleting
intermediate checkpoint files after a verified continuation. V3 remains
immutable. V4 changes only the checkpoint interval from 16 to 128 optimizer
steps and starts both arms from the same completed v3 paired boundary. The
training data, order, F loss, architecture, optimizer hyperparameters,
gradient caps, and 12 active-hour per-arm target remain unchanged. Copy the
entire model, objective, auxiliary head, AdamW, Python/Torch/MPS RNG, stage
draws, and elapsed active/CPU times; never initialize these afresh.

Before starting v4, verify v3 source and checkpoint SHA, equal steps,
checkpoint payload identity, and absence of a running v3 worker. Record
source/destination hashes and the exact parent steps in a migration report.
Resume each v4 arm for one 128-step block and verify checkpoint SHA, restored
optimizer/RNG and finite positive F loss before launching its scheduler.
Do not delete v3 files until both v4 arms have advanced from the imported
boundary and have passed the checks. Retain the original v3 manifest and
an explicit retention ledger, plus v3 checkpoint step 1, every 1024th
available step, and both final migration-parent checkpoints. A pruned v3
manifest is archival evidence, not a resumable v3 artifact; mark it so.
Check free space, exact targets, and process state immediately before deletion.
