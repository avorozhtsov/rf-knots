# Semantic-v1 K=3 big-experiment preflight

Date: 2026-08-10.

## Frozen scientist selection

Foundation pretraining stopped after 11 of 12 planned scientist/seed jobs. The
unfinished `cyclic-memory` seed-72 job could not change the declared selection:
choose the lowest fully promoted seed for `strand-graph` and `raster-axial`, then
the only completed fully promoted `cyclic-memory` seed. This produced
`strand-graph` seed 71, `raster-axial` seed 71, and `cyclic-memory` seed 73.
`strand-graph` is also the compute-matched solo scientist. Exact source
checkpoint SHA-256 hashes are stored in
`pgx-mcts-bench/research/semantic-v1-k3-selection.json`.

This is engineering selection, not evaluation evidence. Every paper arm starts
from the same selected weights, and the arm-comparison banks were frozen before
any arm ran.

## Frozen pilot banks

The outcome-blind pilot split contains 200 training representations and 70
identity-disjoint evaluation representations selected from a 1,080-item
structural frontier with seed 20261920.

- training SHA-256:
  `269dd82129e8a69208798bb3be2d7e7f0f49576aa4bcb3b07cdccd5bc0208b02`;
- evaluation SHA-256:
  `c71c8ce8cecb9ace31d5683b3cfd6503fa48eebfc1d58bdc9cc740190410d75b`;
- protocol SHA-256:
  `3c627aa3fe1b17053df8bbd3d2683c753ba34f0150b3193a354379452340b200`.

The split is stored at
`pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-pilot-banks-20260810`.

## Corrected sharing intervention

The paper sharing arms now train verifier-confirmed, strictly better
receiver-native trajectories through the ordinary policy optimizer. Equal,
worse, and stale donations perform zero policy updates. A permanent
sharing-only option adapter is not attached. Every training block is guarded by
a paired portfolio evaluation and rolled back if solved-set size decreases or
capped objective increases. The corresponding run schema is
`collaborative-scientists-v9-direct-sharing`.

## Budget-critic repair

The first explicit quarter/half/full remaining-L probe found that all three
foundation solve heads were almost budget-insensitive; `raster-axial` moved
slightly in the wrong direction. Therefore the source checkpoints were not sent
directly to the long experiment. Each candidate receives an independent
rollback-guarded L1000 critic repair using low-budget censored negatives,
successful easy episodes, native rehearsal, and a monotonic margin. Acceptance
requires unchanged retained solve rate, monotonicity on every training item,
informative positive/negative caps, and mean solve-probability spread at least
0.01.

All three repairs passed:

| Scientist | Monotone curves | Mean training-panel spread | Retained SR | Retained cc before → after |
|---|---:|---:|---:|---:|
| `strand-graph` | 5/5 | 0.5325 | 10/10 → 10/10 | 2 → 2 |
| `raster-axial` | 5/5 | 0.6503 | 10/10 → 10/10 | 2 → 2 |
| `cyclic-memory` | 5/5 | 0.1609 | 10/10 → 10/10 | 10 → 8 |

An external 24-representation probe then varied remaining L between one quarter,
one half, and the full cap. Every scientist moved monotonically in the correct
direction. Full-minus-quarter changes were 0.0226, 0.0068, and 0.0064 in raw
probability for `strand-graph`, `raster-axial`, and `cyclic-memory`. Because raw
probability differences collapse near zero or one, the qualification rule uses
a minimum 0.05 change in log-odds rather than an arbitrary absolute percentage
point, together with strict aggregate monotonicity.
