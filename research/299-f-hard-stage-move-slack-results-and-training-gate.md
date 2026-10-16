# Hard-stage move slack: audited yield and immediate training gate

2026-10-07. Protocol297 completed 64/64 TRAIN search conditions. The
manifest SHA is
`e4423cd705cc5c351e83b886fe941d63b652dd24666cd69feff8c0e92a6be926`;
all 64 search-shard SHA matched. The independent audit at
`artifacts/f-hard-stage-move-slack-pilot-audit-v1/report.json` has SHA
`8aecc3eaac6698b6f31645d1bb02e82612800d12cdf4075cf7f39e97bf70c8bc`.
It replayed every found solution, checked caps, original starts and all
TRAIN ownership flags, and applied the exact F grouping rule to the
reference plus admitted distinct completed paths across the two fixed
seeds at each root/objective/slack.

Across the 64 searches: 59 raw solved attempts, 58 with an admitted
completed path, 1,175 admitted solved episodes and 5,672.0 measured CPU
seconds. At move slack +16, all eight distinct hard roots produced an
exact nonzero F signal; +32 gave seven of eight. Per objective/CC cap,
the number of F-usable groups out of four roots was:

| Slack | L10 CC1 | L10 CC2 | L1000 CC1 | L1000 CC2 | CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| +16 moves | 4 | 2 | 4 | 4 | 2,766.1 |
| +32 moves | 4 | 3 | 4 | 2 | 2,905.9 |

Thus the +16 diagnostic cost about 346 CPU seconds per distinct F-usable
root in this selected eight-root sample. This is not a reliable population
yield estimate: the roots are the first four of each CC stratum from the
fixed roster, not a broad random sample; there are only two seeds and one
simulation budget. More slack is not automatically better. Do not
bulk-collect 600+ roots from an 8/8 statistic or use relaxed-cap outcomes
as if they were the original capped DEV search.

Training status: **no new F-dominant R2/R4 optimizer step has run**.
However, the old independent audit already provides 384 distinct F-usable
TRAIN roots, exceeding the research/273 minimum of 200 for a short
matched architecture pilot. The new eight stage-3 roots are disjoint from
that old F-usable set (old stage-3 count was zero), giving an upper
combined count of 392 once one frozen, replay-audited bank with consistent
resource contexts is built. Do not call the bank complete merely by
adding the two report counts. The 1,000+ distinct-root target is still
unmet; it gates the later multiday scale-up, not the short paired start.

Next priority: build the versioned joint F bank from the old audited
teacher and the +16 pilot (reference, distinct admitted successful paths,
same root/objective/resource cap), recheck exact F groups and exclusion
boundaries, freeze start checkpoint, R2/R4 source, equal samples/steps,
gradient-norm ceilings for auxiliary CC/moves and independent DEV panel.
Run a short checkpointed MPS smoke test and then start the matched R2/R4
pilot; the 24-active-hour clock begins only after both trainers have real
optimizer checkpoints. In parallel, design a broader score-blind
TRAIN-only +16 yield replication before promising a 1,000-root bank.
No network-training, DEV promotion or superiority claim follows from the
search pilot alone.
