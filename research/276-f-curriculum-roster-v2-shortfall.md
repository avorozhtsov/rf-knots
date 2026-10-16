# Second roster quota miss: reserve a safe easy-stage quota

2026-10-04. The separate v2 builder
`scripts/build_f_curriculum_roster_v2.py` exited 1 without publishing a
roster. Its fixed single-pass order found [149, 283, 283, 284] against
quotas [150, 283, 283, 284] after all 16,000 indices. The one-root
difference from v1's 151 easy roots is due to cross-stage invariant-group
deduplication and different stage quotas, not a loss of source files.
Exact reject counts: duplicate/excluded 4,344; ownership 1,099;
size-or-filled 9,558. Preserve v1 and v2 failures.

For v3, freeze quotas **140, 286, 287, 287** (sum 1,000), same seed,
index range, exclusions, replay and ownership. The 140 easy roots leave
a nine-root margin to the 149 seen in v2. If v3 also misses, do not keep
guessing quota numbers: build a full prospective eligible inventory and
freeze a new stratified selection from measured availability. Candidates
remain distinct TRAIN roots, not audited F-usable groups or network steps.
