# F curriculum roster: stage-1 supply failure and new quota

2026-10-04. The first frozen roster builder
`scripts/build_f_curriculum_roster_v1.py` (SHA256
`91358278f7af31d8e2725b1afbf4eea70e7a4251d8a66fefaa5c0331bb197bfe`)
exited with code 1. Under seed 20261125, SHA-ranked indices 0..15999,
TRAIN ownership and all exclusion checks, it found only **151** qualifying
CC=1 roots of length at most 64 versus the prospective quota of 250.
The other three 250-root quotas filled. The exact error was:

`fixed roster insufficient: [151, 250, 250, 250], rejects={'duplicate_or_excluded': 4117, 'ownership': 963, 'size_or_filled': 10019}`

This is a supply limitation, not a failed neural-training result. Preserve
the failed v1 source and this record; no v1 roster artifact was published.
Do not inflate stage 1 with repeated seeds, lambda copies, or duplicate
invariant groups.

Use a separate v2 roster contract with the same generator, seed, ranked
indices, ownership and heldout exclusions, but fixed quotas **150, 283,
283, 284** for the four protocol273 stages (total 1,000 candidate roots).
The first stage keeps nearly all 151 available easy roots. Stages 2–4
remain length-ordered; short examples are retained as rehearsal once
training begins. The 1,000 candidate roots are still not 1,000 F-usable
training roots: the latter requires independently verified diverse-cost
completed searches. A separate balanced pilot must measure that yield and
CPU before bulk collection.
