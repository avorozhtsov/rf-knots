# Storage gate after approved retention

2026-10-15 11:21 UTC heartbeat. Fresh host ps shows no continuous trainer/scheduler/MCTS worker. Free space is 9.6 GiB, below the unchanged 12 GiB guard. Do not restart either worker or lower the guard; no further deletion is authorized. Previous approved retention is complete and must not be repeated.

R2 v14 safely saved step356353, new active1673.036478 seconds: +2048 optimizer steps and +793.356054 active seconds since the last restart snapshot. Latest checkpoint SHA a8107a01f43b0b33f8e035d21fd6b11b0bee70df333b3e2ce099c066cfa186d8. All11 checkpoint SHA and contract input SHA passed. Full AdamW and Python/Torch/MPS RNG fields present. Latest F loss0.0984462723, F encoder norm0.06698170085; this new-root-only update has zero auxiliary/rehearsal gradients. Bank remains441 roots/755 contexts, no new optimizer admission or utility claim.

Main MCTS v2 safely stopped with requested_or_storage_stop, 1785 committed conditions (+6), 19428.194543 search CPU seconds (+142.607572), 55 complete audits. All1785 shard SHA independently rechecked. Partial batch is not an admitted F bank addition. Search is not optimizer training. No STOP_REQUEST or numerical failure is inferred solely from status; low disk independently prevents restart.

Observed directory sizes: v14 35 MiB, main MCTS181 MiB; VM volume4 GiB. These observations do not explain the system-wide fall from17 GiB to9.6 GiB. No unrelated processes or files changed. Required next action: restore comfortably above12 GiB through separately approved scoped storage cleanup or additional storage, then freshly verify source/checkpoints/AdamW-RNG/shards and absence of duplicates before same-contract resume. Preserve all remaining snapshots and production/database state.
