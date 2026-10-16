# Prospective v4 endpoint adapter for the frozen R2/R4 DEV comparison

2026-10-08. This adapter is frozen before either arm reaches 43,200
checkpointed active MPS seconds or any R2/R4 search result is inspected.
It changes protocol307 only where v3 checkpoint provenance was replaced by
the verified storage-safe v4 continuation. The score-blind DEV panel SHA,
32 roots, both L10/L1000 instances, two seeds 20271008/20271009,
canonical pgx-mcts-bench search parameters (128 simulations, candidates 4,
widening 4, c_puct 1.5, temperature 1, distance prior, cache on), 256 tasks,
replay audit and architecture decision rule are exactly protocol307.

The eligible endpoint is the first completed paired v4 scheduler step at
which both arms have at least 43,200 cumulative checkpointed active seconds,
including the verified v3 prefix. Both must have the same optimizer step.
The evaluation contract pins the v4 trainer, scheduler, training manifests,
endpoint checkpoint hashes, migration report, retention ledger, DEV panel,
this adapter and original protocol307 before any search shard is produced.
No checkpoint selection by TRAIN loss, viewed DEV, or TEST150. The original
v3 evaluator and protocol307 remain unchanged. This is an internal DEV
architecture comparison, not a superiority claim.
