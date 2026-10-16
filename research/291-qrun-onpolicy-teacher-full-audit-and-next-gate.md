# Q-run rung-1 on-policy teacher: complete independent audit

2026-10-06. The fixed TRAIN-only collection in
`artifacts/qrun-rung1-onpolicy-teacher-v1` completed all 2,000 root/objective
searches (1,000 distinct TRAIN roots, L10 and L1000, one seed and 64
simulations). Its manifest SHA is
`658c128c11fed26a639994c5d54af8ad31550f7f189be52c144cbb76c192f6a2`.
The collector source SHA and contract SHA remain respectively
`d45b6c20a4a4b956eeaf490093754fb2e6460e4ba1c4812c69d14abcfcc965bd`
and `7445b1708a5ea91811b115ed781773c7420b714f05642fba61a081b84bee6931`.
All 2,000 shard SHA match the manifest. The collector exited normally.

The independent audit at
`artifacts/qrun-rung1-onpolicy-teacher-audit-v2/report.json` (SHA
`ee1049c12bdaca9142d13c2a8759e6bba91fe3c10d34815e1ddff3e58e1bbb86`)
completed 20 contiguous 100-shard chunks. It checked source/contract/input
SHA, task order, TRAIN ownership at every CC state, full replay of each
admitted solved episode, starts, caps, CC/moves and objective cost. All chunk
identity fields and the sum of their counters agree with the final report.
The audit source SHA is
`2c3449ec6dce8ec0a85859add79335a4d9f37c711b80647dc914c922047b2ebb`.
It invokes the unchanged independent v1 audit, adding only a bounded cache
for the pure invariant calculation; v1 and cached v2 gave identical output
on the first 10 shards. No collector, search contract, checkpoint or old
artifact was changed.

| Stage/objective | Searches | Admitted solved searches | Cheaper than reference | Equal | Dearer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0/L10 | 140 | 131 | 0 | 93 | 38 |
| 0/L1000 | 140 | 128 | 0 | 84 | 44 |
| 1/L10 | 286 | 251 | 0 | 149 | 102 |
| 1/L1000 | 286 | 240 | 0 | 148 | 92 |
| 2/L10 | 287 | 201 | 0 | 133 | 68 |
| 2/L1000 | 287 | 187 | 0 | 127 | 60 |
| 3/L10 | 287 | 72 | 0 | 72 | 0 |
| 3/L1000 | 287 | 63 | 0 | 63 | 0 |
| **All** | **2000** | **1273** | **0** | **869** | **404** |

The search used 73,967.669 CPU seconds and 7,897,748 neural evaluations.
It produced 10,741 admitted solved episodes and 10,738 distinct admitted
root/path pairs, but **zero** improved-cost root/objective labels. CPU per
improved path is therefore undefined, not zero. A capped no-solve is
censored, not a negative policy action. Many found solutions cost more than
the already verified constructive reference; replacing the reference by
them would be counterproductive for the CC/move objective.

A separate post-audit comparison of semantic action sequences found 1,131
equal-cost admitted episodes. Of these, 318 repeat the reference sequence
and 813 are distinct equal-cost alternatives, spread across 609
stage/objective/root groups (a root can contribute under both objectives).
This is potential *imitation diversity*, not cost improvement and not
evidence of better downstream search. It is a post-audit diagnostic, so do
not retroactively select a checkpoint using it.

Decision: do **not** automatically train a new rung by replacing the
constructive reference with this teacher. First freeze a small, separate,
equal-CPU pilot comparing reference-only rehearsal against a controlled
mixture of verified equal-cost alternative paths, holding initial weights,
TRAIN roots, steps, gradient scale and resources fixed. Pre-register a new
validation-owned DEV search panel not used by the prior checkpoint choices;
compare L10/L1000 solve rate, CC/moves on paired successes and CPU/neural
evaluations. Promote only a reproducible gain. In parallel, reconsider a
teacher with greater search budget or different search policy, but measure
cost-improvement yield per CPU before bulk collection. This completed
teacher is search data, not new optimizer training or the F-dominant R2/R4
experiment. TEST150 remains excluded from selection.
