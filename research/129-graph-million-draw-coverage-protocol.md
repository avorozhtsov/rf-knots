# 129 — Does a million graph draws provide a million representations?

TRAIN-only sampling experiment, no neural computation, optimizer, MCTS, TEST or
modification to97. The completed121 census found only4516 distinct graph pairs
among32768 graph draws. Before investing in larger training, measure how much
coverage comes from more draws versus changing replacement/group weighting.

Use the unchanged, replay-audited ordinary-graph loader and all50615 admissible
TRAIN pairs. Compare three explicit proposals on the SAME deterministic stream
of category, group, replacement-row and endpoint-swap random numbers, seed20261063.

1. `balanced-replacement`: the current class/group/row-with-replacement rule.
2. `balanced-cycles`: same class and group choices and endpoint orientation, but
   exhaust a seeded shuffled permutation within each group before repeating.
3. `class-cycles`: same class choices and orientation, but exhaust a permutation
   of all records in that class, ignoring the sampled group. This deliberately
   changes the within-class group prior to be proportional to record count.

All methods consume four top-level uniform random numbers per draw. Cycle
permutations have separate deterministic RNG seeded by method/class/group/cycle;
they cannot perturb subsequent class/group/orientation choices. Compare32768,
65536,262144 and1000000 draws. Save every selected pool-record index, group
identity and swap bit for independent replay; report source coverage, unique RAW
endpoints, class/group exposure and maximum pair multiplicity. A pool-record
index is not a knot identity. One million draws cannot create new representations
outside this frozen support and is not a million self-play games.

The primary question is coverage, NOT which distribution learns best. Method3's
possible gain trades equal-group weighting for representation coverage. Method2
preserves long-run equal-group weighting but cannot compensate for rarely
selected large groups. Do not change the active sampler, choose loss weights from
TEST outcomes or claim scientific promotion from higher unique counts.

Freeze this protocol, standalone script/tests and upstream121 corpus lineage
before execution. Verify saved samples reproduce the same trace from seed, match
the current sampler exactly for method1, and contain every class/group identity
within its valid range. Reject changed sources, output reuse or invalid support.
