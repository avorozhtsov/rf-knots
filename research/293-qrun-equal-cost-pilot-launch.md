# Equal-cost diversity pilot launch and frozen evidence

2026-10-06. The separate pilot from research/292 is running; this note
records launch evidence, not a search result or a winner.

The TRAIN-only bank `artifacts/qrun-equal-cost-bank-v1/bank.json` has SHA
`ebfc9f467a783255e48aa2023e8060cd0309519521ec3bbd515ce774fe7b1264`.
It was built only from the fully audited teacher and has one independently
replayed, TRAIN-owned, exactly equal-cost alternative for 609
root/objective pairs, by stage L10/L1000: [59,60], [99,98], [97,90],
[58,48]. All 1,000 reference witnesses remain available.

The frozen training source SHA is
`b8004fb155b2b4fa4a8cb007d64f6a36f6f5b4768f126ad388ec6cbe73618cd8`.
Both arms started from curriculum1024 SHA
`03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8`,
used native MPS, and finished all 320 optimizer steps. All 10 checkpoints
per arm passed SHA, optimizer and RNG-state audit. The control A endpoint
in `artifacts/qrun-equal-cost-pilot-v1-A-envfix` has SHA
`44386ba7f43faefd8f0138bedd093ffe7fd8fc810e77f6a9f4a74fd970098a5b`;
the diversity B endpoint in `artifacts/qrun-equal-cost-pilot-v1-B` has SHA
`0d7b45c8a37450c94a06cf9c40eebcc30332fe5309f353d0945c84c53f4a4763`.
Each arm saw the same 320 roots per stage and the same 427 draws with an
available alternative. B chose the alternative on 212 of those draws; A
always used the reference. Recorded active wall/CPU seconds were
91.36/58.15 for A and 79.36/48.70 for B. Equal optimizer steps do not
imply identical elapsed compute; search evaluation will count resources.
The first attempted A launch lacked the canonical pgx import path and
failed before creating a manifest or optimizer step; its separate empty
artifact was preserved. The working A run changed only PYTHONPATH.

The new, score-blind DEV panel `artifacts/qrun-equal-cost-dev-panel-v1/panel.json`
has SHA `8601189cf8e9544434220fac03b3a58f2d4a1200aa3eb26b7be062f3f8a5ae46`.
An independent audit verified all 32 reference replays, validation root
ownership, paired L10/L1000 caps, exclusion of TRAIN and prior viewed
Q-run DEV roots, and exact-state disjointness from TRAIN for word length
at least 20. It has 16 short and 16 long roots; CC counts are short
10×CC1/6×CC2 and long 1×CC1/15×CC2. Common shorter tails remain possible.

The matched DEV search source SHA is
`b3d0047f852cb8192ee08573deeee6543d3b10d6d163d75853109b1b3fede540`;
contract SHA `3467236e542f2867eb211fcc927332c84616644dea900500d8a8be6bb8263773`.
It compares curriculum1024, A320 and B320 on the same 32 roots × both
objectives × seeds 20271209/20271210, 64 simulations =384 conditions
through canonical pgx-mcts-bench. The first shard was SHA-checked and its
witness fully replayed with start, caps and cost; the worker then resumed.
Until all 384 tasks finish and pass independent audit, do not use this
panel to select a winner. TEST150 remains excluded. This pilot is not the
F-dominant R2/R4 experiment and does not start its 24-hour timer.
