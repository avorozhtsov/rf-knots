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

## Source-disjoint readiness result

The decisive qualification panel contained 24 source-disjoint table
representations, two paired stochastic attempts per representation, L1000,
native-action horizon 128, and a preregistered 70% representation-coverage
floor for every scientist at one common simulation dose. The panel excluded
the frozen 200-representation training bank and 70-representation evaluation
bank. Increasing MCTS simulations did not repair generalization:

| Scientist | 64 simulations | 128 simulations | 256 simulations |
|---|---:|---:|---:|
| `strand-graph` | 1/24 (4.2%) | 1/24 (4.2%) | 1/24 (4.2%) |
| `raster-axial` | 7/24 (29.2%) | 6/24 (25.0%) | 6/24 (25.0%) |
| `cyclic-memory` | 1/24 (4.2%) | 2/24 (8.3%) | 3/24 (12.5%) |

The result is a gate failure, not a weak pass. No common dose approached 70%,
and 256 simulations were explicitly tested. The final 256-simulation budget
probes remained aggregate-monotone for all three scientists. The
full-minus-quarter log-odds changes were 0.683, 0.314, and 0.0467 for
`strand-graph`, `raster-axial`, and `cyclic-memory`; the last also narrowly
missed the declared 0.05 nonconstant-signal check.

The complete resumable artifact is
`pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-roster-readiness-v2-20260810`.
Its executable-source SHA-256 is
`6c6d7aec5d27d76b78514e03d8fb3243ec7bb19e182f7b6a9d47e6b2a46d8629`,
which is reproduced by `pgx-mcts-bench` commit `df1a55d`. Consequently the
100-representation assessor run, adaptive paper arms, paid 1,000+ run, and
hard-knot campaign were not launched.

## Direct-sharing and resumability smoke

A separate low-compute static sharing/control smoke used identical checkpoints,
banks, task order, search seeds, and training allocation. It was deliberately
interrupted after committed round 2 and resumed through rounds 0--9 without a
duplicate or missing round. Direct ordinary-policy sharing admitted 19 unique
strictly better donations; the no-sharing control admitted zero.

This engineering success did not pass the scientific gate. On a disjoint
40-representation bank with one paired attempt and four simulations per move,
the final portfolios were:

| Objective | Sharing solved | Control solved | Sharing-only | Control-only | Capped-objective delta, sharing - control |
|---|---:|---:|---|---|---:|
| L10 | 3/40 (7.5%) | 3/40 (7.5%) | none | none | +46 |
| L1000 | 4/40 (10.0%) | 3/40 (7.5%) | `12a_864` | none | +11,973 |

Thus the final L10 solved sets were exactly equal, while sharing added one
L1000 solve but worsened capped objective on both objectives. The direct-sharing
gate now explicitly requires at least 70% held-out coverage in both paired arms
for every objective, in addition to at least ten donations, no solved-set
regression, and no capped-objective regression. This prevents a sparse smoke
like this one from being mislabeled a positive sharing result. The artifact is
`pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-interruption-resume-smoke-20260810`.

## Decision and correction

The selected networks learned the six small synthetic foundation stages, but
that was not enough neutral pretraining for unseen 8--12 crossing table
representations. This explains why foundation-stage solve rates near 0.8--1.0
coexist with the much lower source-disjoint rates above: the denominators and
task distributions are different. It is not evidence that the architectures
can never scale, and it is not fixed by increasing 64 simulations to 256.

The next valid stage is a neutral, source-disjoint bridge curriculum shared by
all future arms: ordinary self-play only, static outcome-blind ordering, no
donations, no adaptive scheduling, balanced replay, adaptive new-task dose, and
adaptive rehearsal. It must use representations disjoint from the frozen pilot,
evaluation, readiness, assessor, and hard-knot sets. Only bridge checkpoints
that re-pass the same 24/100 representation gates at at least 70% may enter the
five-arm pilot. The present three checkpoints remain reproducible development
seeds, not paper-arm starting checkpoints.

Verification at the final implementation commit: 300 `pgx-mcts-bench` tests
passed; 213 `rf-knots` tests passed and ten optional tests were skipped. Model
binaries and generated run artifacts were not committed.
