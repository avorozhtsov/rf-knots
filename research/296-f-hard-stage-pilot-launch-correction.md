# Hard-stage pilot launch correction

2026-10-07. The first launch of protocol295 source
`scripts/collect_f_hard_stage_budget_pilot_v1.py` SHA
`14e644d34ef59a077ece8c3d65bc1fac89547f7b4781233cfffb3c26f13301f4`
failed before manifest, search or shard creation. Its Python schedule used
tuples, whereas the contract check compared it to JSON-normalized lists.
The separate empty artifact directory (writer lock only) is retained as
evidence. No v1 results exist.

The corrected separate source
`scripts/collect_f_hard_stage_budget_pilot_v2.py` SHA
`4a7bb0a30863dc85999f80a4d0db76107962ef37bd6b66d98c5a6a0d686ae074`
changes only the schedule representation to JSON-stable lists. A 64-task
schedule and JSON round-trip check passed. Run the unchanged scientific
protocol295 in a separate v2 artifact. The first shard must be independently
checked before resuming; no claimed collection or training progress yet.
