# Separate TRAIN stage-3 budget-64 yield experiment

2026-10-04. The frozen 16-simulation pilot completed 640/640 shards,
manifest SHA256 `1f17e91bef18b0dccc719ced35a4718eff2d8f4412220fd885ec4ef03bdcf24f`.
Independent source/SHAs, every successful solution replay, caps and
TRAIN ownership passed. It found 14/32 distinct roots with nonzero F
signals: stages 0/1/2/3 contributed 6/5/3/0 of eight. There were 64
distinct exact F contexts; measured search CPU was 4051.777 seconds.
Stage 3 had admissible successful attempts (44/80 L10, 42/80 L1000),
but no usable F contrast. Thus simply generating 1000 candidates cannot
be reported as 1000 trainable roots. These are TRAIN screening results,
not DEV selection or optimizer steps.

Before expanding collection to all candidates, test whether more search
budget creates stage-3 path-cost contrasts. Use the same eight fixed stage-3
roots from protocol277, both lambda values, the same ten independent seeds
20270000..20270009, and exactly 64 simulations per search: 160 attempts.
The actor, prior, cache and all other search settings stay identical.
Record CPU/wall and neural evaluations for comparability with b16. Do not
reinterpret b16 unsolved attempts as negative actions. Freeze this contract
and source before launch; never modify the b16 source or shards.

Independently audit all shards, full replay and TRAIN ownership, then count
new F-usable distinct stage-3 roots, paths and exact contexts. A useful
increase informs a separately fixed b64/b128 selective teacher collection;
otherwise reconsider search/teacher diversity rather than blindly running
the 1000-candidate roster. No R2/R4 trainers begin until >=200 distinct
audited F-usable TRAIN roots and a separate frozen training contract.
