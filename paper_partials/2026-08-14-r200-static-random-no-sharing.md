# Fifth R200 arm: frozen random order without sharing

Date: 2026-08-14. Status: **launched; no result claimed yet**.

The fifth Semantic-v2 R200 arm tests whether the ACS curriculum changes learning
relative to a fixed random presentation order. It reuses the canonical three
scientists and exactly the post-R24 states used to start the optimized
static/no-sharing control. There is no random-order R24 phase and no donation.
The common training/evaluation seed remains `20262120`, so only the schedule is
changed relative to the static/no-sharing control.

The registered run is
`SV2-3S-R200-SIM64-F5-AR-EV4-RANDOM-NO-SHARING`, with artifacts at
`/srv/braid/artifacts/semantic-v2-r200-optimized/static-random-no-sharing` and
systemd unit `sv2-r200-opt-static_random_no_sharing.service`. It uses SIM64,
initial F5, adaptive compute and rehearsal, eight self-play games, 96 optimizer
steps, batch 64, EV4, a 128-action horizon, and ten-rung blocks. Every rung has a
finite compute dose: an unsolved representation remains in coverage and capped
L10/L1000 denominators, is durably recorded, and does not block the next rung.

The schedule is a uniform permutation of sorted representation identifiers with
seed `2026081401`. Its SHA-256 is
`c1ca7e85eeb1d4f65d9c3e92e7c82fca12daa614a8ea9a6fe5533593c72ccf44`.
The first ten identities are `12a_184`, `11a_231`, `12a_455`, `12a_1`,
`11a_284`, `10_26`, `12a_278`, `12a_6`, `12n_157`, and `12a_264`. This order
was accepted before observing any outcome and must not be rerolled.

The immutable protocol SHA-256 is
`98f71055d9ede08f0efa805cb8f4a09b73008f5d72a4ebbbd206454c9fb1a1ad`.
The R200 and R24-prior bank hashes are respectively
`a5fa348a0ff3204a28b40913ea04fa3d3f8ad6a0255c0fe8ca1cf3e5598096de`
and `0a837f72e87e7c41de3caf08e577b6c567751d2b2d48990f413e1f922e644e96`.
The executable source is based on commit `f557eee3`, with exact source-tree hash
`16cdf969354c863ab688cc960db63f22ccd07f95f80d814197df8ca70fc7c431`;
the deployment patch and per-file hashes are stored in its isolated worktree.

The original three scientists are protected controls and are not eligible for
early stopping in this arm. A larger architecture/innovation league may use the
same permutation, with checkpoints every ten rungs and preregistered conservative
pause rules, but its results must be reported separately so roster expansion is
not confounded with the fifth-arm order comparison.
