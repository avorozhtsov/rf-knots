# Joint F bank: preparation gate for the short R2/R4 training pilot

2026-10-07. The user has asked whether the next training-preparation step
has actually begun. This protocol freezes a TRAIN-only bank construction
from two independently audited sources; writing this protocol alone is
not optimizer training.

Source A: roster-v3 SHA
`268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506`,
the 2,000-condition curriculum1024 on-policy teacher manifest SHA
`658c128c11fed26a639994c5d54af8ad31550f7f189be52c144cbb76c192f6a2`
and independent full audit report SHA
`ee1049c12bdaca9142d13c2a8759e6bba91fe3c10d34815e1ddff3e58e1bbb86`.
Source B: protocol297 +16 move-cap only (not +32), 64-condition
manifest SHA
`e4423cd705cc5c351e83b886fe941d63b652dd24666cd69feff8c0e92a6be926`
and independent full audit report SHA
`8aecc3eaac6698b6f31645d1bb02e82612800d12cdf4075cf7f39e97bf70c8bc`.
Source B contributes only the eight fixed stage-3 roots, both objectives,
both seeds, with the same +16 resource cap within each comparison. Do not
mix +16 paths with original-tight-cap paths in one F group.

Recheck each source shard SHA, witness replay/start/caps, task identity,
source audit pins and reference witness. Deduplicate semantic action
sequences per root/objective/resource cap; a bounded no-solve contributes
no negative action. Apply the exact F rule used by research/273:
objective-specific cost, near-best margin 2, at most three positives,
K=4 strictly worse negatives, shared state **and remaining resource**
contexts. Retain only groups with nonzero F coefficients, but store all
distinct admitted completed witnesses for each retained group and their
source lineage. Record distinct root count, group count, stage/objective
coverage, path count, and SHA-pinned resumable output. Expect the old
384 roots plus up to eight new stage-3 roots; independently verify the
result rather than summing reports as proof.

Once the bank is complete and independently checked, freeze a separate
short R2/R4 optimizer contract before any step. Start both from the same
curriculum1024 checkpoint SHA
`03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8`,
same bank, sampled root order, optimizer settings and active-work accounting.
F must dominate shared-encoder gradients; combined CC/moves auxiliary
norm <=10% of F and distance/known-solution rehearsal <=20% of F, with
zero-F batches skipped and measured norms/cosines logged. R2/R4 differ
only in shared-weight graph propagation rounds. A short smoke checkpoint
for each must verify native MPS, AdamW/RNG resumability and actual
optimizer steps. Only then start the matched active 24-hour pilot; its
clock begins after both first checkpoints. The 1,000+ distinct-root
target gates later multiday scale-up, not this short start (>=200 already
available).

No viewed DEV/TEST150 checkpoint selection, no source mutation of prior
SHA-pinned experiments, no cloud purchase, database publication, commit,
merge or push. A new independent DEV panel must be frozen before reading
the R2/R4 comparison.
