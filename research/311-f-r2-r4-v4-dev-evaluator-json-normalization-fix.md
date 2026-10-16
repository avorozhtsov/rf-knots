# V4 DEV evaluator JSON-normalization repair

2026-10-09. The frozen v4 evaluator source SHA
330c5e78ff81b67adb92bf26961b68d2cad3a511e63bf0430d56114447fe2ffc
failed before writing a manifest or condition shard: its fresh in-memory
contract retained Python tuples for arms and seeds, then compared that value
to its JSON-normalized form (lists). Preserve that failed source and empty
artifact. The corrected v5 evaluator changes only this contract construction:
normalize the contract to JSON-compatible lists before computing its digest
and comparing it with saved state. Pin v5 source and this note in the new
evaluation contract; all panel, endpoint, seeds, 128-simulation search,
canonical pgx, audit, and decision rules remain protocols 307/310. First
run one condition into a new artifact, check SHA and full replay, then resume
the same source and contract for the remaining 255 conditions.
