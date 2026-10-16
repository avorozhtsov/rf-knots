# Q-run bootstrap DEV panel v3: exclude shared simplification tails

2026-10-05. The frozen v2 builder completed its 16,000-index search and
failed supply: 0/16 short, 0/16 long; rejection counts length 9,626,
root owner 5,088, excluded/duplicate 738, exact-state overlap 548.
Keep the v1 and v2 failures and their sources unchanged.

An independent 500-index diagnostic found 46 length-eligible,
validation-owned candidates. All 46 shared short exact states with TRAIN;
overlap counts fell from 46 at word length 1 and 2 to 3 at word length 19,
and none overlapped at word length >=20. These are common simplification
tails, not evidence that the roots match. This diagnostic is only a supply
check, not a score or model selection.

Freeze a separate v3 panel with the same ranked seed 20261204, indices
0..15999, quotas 16 short (word length 20..64) and 16 long (129..192),
old query-group exclusions, validation-owned root, and full replay. Require
exact `(word, strands)` disjointness from all 1,000 TRAIN reference paths
for states with word length >=20. Shorter common tails are explicitly
allowed; the panel therefore tests new roots and substantial states, not
trajectory-wide ownership or disjointness. The two lambda instances have
identical starts and resource caps. All rows are selected score-blind.

Audit input SHA, all witnesses, ownership, exclusions and overlap before
starting the frozen three-arm 64-simulation matched search. TEST150 is not
used for selection. If v3 supply fails, preserve that failure too.
