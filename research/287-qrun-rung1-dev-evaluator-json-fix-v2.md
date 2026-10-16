# Rung-1 matched DEV evaluator: JSON contract repair

2026-10-05. The first evaluator entrypoint failed before its first shard
or manifest: its in-memory seed tuple did not equal the JSON-normalized
contract list during the new-run identity check. Preserve the empty v1
artifact and v1 source. This is a serialization error, not a search result.

Use a separate v2 entrypoint with the same panel, checkpoints, two seeds,
64-simulation search settings, task order and intended 384 conditions.
Normalize the seed list before computing the contract; pin this wrapper
and protocol SHA in the contract. First run one condition, inspect its SHA,
then resume the same contract. No outcome-dependent change is authorized.
