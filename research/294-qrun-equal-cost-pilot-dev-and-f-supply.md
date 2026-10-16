# Equal-cost diversity pilot: complete DEV audit and F-supply implication

2026-10-07. The fixed matched DEV search from research/292 and 293
completed all 384 conditions. Manifest SHA
`a6acc49ca3e75fbe114fb8449794583f3342cca0e09d53f1298648a17c2e6641`;
contract SHA
`3467236e542f2867eb211fcc927332c84616644dea900500d8a8be6bb8263773`.
The independent report at
`artifacts/qrun-equal-cost-matched-dev-audit-v1/report.json` has SHA
`8d29865711e1a404a3d2f555b85d12e281e526578c4016b194f9db73cbd245fe`.
Every shard SHA and pinned input SHA matched; all 229 found witnesses were
fully replayed with start, caps, CC/moves and L10/L1000 cost checked. The
collection process exited normally. This is matched neural-policy search,
not training and not a fully non-neural heuristic comparison.

Distinct DEV roots solved in either of two seeds, out of 16 per length band:

| Policy | Short L10 | Short L1000 | Long L10 | Long L1000 |
| --- | ---: | ---: | ---: | ---: |
| curriculum1024 parent | 16 | 15 | 4 | 5 |
| reference-only A320 | 16 | 16 | 7 | 5 |
| equal-cost diversity B320 | 16 | 15 | 6 | 8 |

B versus A on the *same* roots and search seeds: long L1000 gained 2/0
and 2/1 (gained/lost) on seeds 20271209/20271210. However short L1000
lost 1 and 2 solved roots respectively, and long L10 tied on the first
seed but lost one on the second. On jointly solved short L10 cases B was
cheaper/dearer/equal 2/3/11 then 1/1/14; on short L1000 it was 2/3/10
then 0/2/12. Jointly solved long cases had equal objective cost. Thus the
observed long-L1000 gain is real on this panel but the predeclared
no-regression and paired-cost conditions are not met. Do not promote B or
invest a large CPU budget in this mixture from this result. Capped failures
remain censored, not negative actions, and this small DEV panel is now
viewed and cannot select further checkpoints.

For all 128 matched search attempts per policy, solved attempts were
70 parent, 80 A, 79 B. CPU seconds were respectively 4494.15, 4354.40,
4344.77; policy neural evaluations 502157, 493608, 494055. Training was
matched by 320 optimizer steps and 1,280 root draws per arm, but measured
active training CPU/wall differed; this search comparison reports actual
resource cost rather than treating equal steps as equal compute.

The audited 2,000-condition TRAIN on-policy teacher also deserves a
supply correction before any F-dominant multiday run. A read-only count of
admitted completed paths found 539 distinct roots with at least one path
more expensive than the constructive reference (cost separation); none of
the 287 hardest stage-3 roots had such a path. Applying the planned F
near-best objective margin 2 and maximum three positives at the cost
level leaves at most 388 distinct roots, 689 root/objective groups, with
a strictly dearer eligible path: stage/objective counts are stage0
103/96, stage1 176/168, stage2 78/68, stage3 0/0 for L10/L1000.
This is an **upper bound**, not an audited F-signal count: shared
state-and-resource contexts and nonzero gradient still need checking.
Most minimum positive cost gaps were two moves. Consequently the teacher
cannot satisfy the required 1,000 distinct F-usable roots, and it provides
no hard-stage F comparisons. Neither the prior Q-run imitation nor this
equal-cost pilot starts the research/273 24-hour R2/R4 timer.

The follow-up exact TRAIN-only signal audit in
`artifacts/qrun-f-supply-audit-v1/report.json` (SHA
`b83731bd246f5ad1132bc5aaea83220d7e238b174e8c1bcdd8aea880605cd26d`)
replayed admitted traces through the actual F grouping rule, including
shared state/resource contexts, near-best margin 2, at most three
positives and K=4 negatives. It found **384 distinct F-usable roots**
and 676 root/objective groups. Stage L10/L1000 group counts were
103/95, 174/167, 74/63 and 0/0. This resolves the cost-level upper
bound of 388 roots and confirms that the current bank is far short of
1,000 F-usable roots, especially at the hardest stage. It is a signal
supply audit, not optimizer training or a search-quality result.

Decision: retain A as a reference-only exploratory baseline, not a proven
winner. Next freeze a *small* TRAIN-only targeted hard-stage supply pilot
with independent restarts and larger real search budgets, measuring
completed-path cost diversity, exact F contexts and CPU per usable hard
root before any bulk collection or multiday F training. Do not synthesize
negative actions from capped no-solves or use TEST150. A fresh independent
DEV panel is required for any later promotion, and a predeclared fully
non-neural canonical pgx-mcts-bench control remains mandatory before a
superiority claim.
