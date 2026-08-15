# DKT72-PD-v1 external L1000 evaluation

Date: 2026-08-14. Status: **completed held-out evaluation**.

## Benchmark name and scope

We name the executable 72-row Table 1 benchmark **DKT72-PD-v1**. `DKT`
identifies Dranowski, Kabkov, and Tubbenhauer; `72` is the complete published
Table 1; and `PD` records that each braid is deterministically derived from the
authors-workbook planar-diagram presentation. These are not the inflated
diagrams or replayable witnesses on which the paper established its bounds.
The frozen dataset SHA-256 used by this evaluation is
`83c591d4b8ea002e5082ede424766fe7eb05c87958defe70413f679495a289fb`.

## Frozen scientist selection

Five live checkpoints with at least 12 completed R200 rungs were screened by
within-arm capped L1000 performance while retaining architecture and curriculum
diversity. Because adaptive and static arms have different task panels, the
screen is not a causal cross-arm ranking.

| evaluation alias | architecture | curriculum | completed R200 rungs |
|---|---|---|---:|
| `ans-strand-graph` | strand graph | adaptive, no sharing | 39 |
| `ss-raster-axial` | raster axial | static, sharing | 31 |
| `ans-raster-axial` | raster axial | adaptive, no sharing | 39 |
| `as-strand-graph` | strand graph | adaptive, sharing | 60 |
| `ans-cyclic-memory` | cyclic memory | adaptive, no sharing | 39 |

Their coordinated states were copied before evaluation and identified by
SHA-256. Continued R200 training therefore cannot change this evaluation.

## Evaluation protocol

The held-out evaluation performs no learning and uses L1000, SIM256, four
paired stochastic attempts per representation, native action horizon 128, and
failure cap `U_1000 = 20128`. The primary per-scientist statistics are supported
representation coverage, attempt solve rate, capped L1000 mean, best verified
crossing-change and semantic-move counts, network evaluations, and wall time.
The five-scientist portfolio is the best verified L1000 witness per knot.

The current frozen games declare `max_strands=5` and `max_len=48`. Exactly 25
of the 72 workbook-PD braids fit that capacity; 47 require 6--10 strands. The
evaluator records these 47 as `unsupported_capacity`, not as ordinary failed
search attempts and not in capped-loss means. Thus this run is both a
25-representation performance evaluation and a direct measurement of the
representation-capacity blocker that must be removed before making full-panel
claims.

The low-priority Nebius unit is `dkt72-pd-v1-l1000-eval.service`; its immutable
artifact root is
`/srv/braid/artifacts/dkt72-pd-v1-l1000-eval-20260814`. It contains the dataset,
selection manifest, frozen states, per-attempt verified witnesses, logs, and
resumable atomic item results.

## Results

All five scientists completed the identical 25-representation supported panel.
`Coverage` counts representations with at least one verified solve among four
attempts. `L1000 mean` substitutes 20128 only for a supported representation
with no verified solve; the 47 capacity-unsupported representations are absent
from its denominator. `Bounded mean` additionally clips verified objectives at
20128 and is reported because a verified but highly inefficient route can
exceed the failure substitution.

| scientist | coverage | solved attempts | L1000 mean | bounded mean | network evaluations | wall time |
|---|---:|---:|---:|---:|---:|---:|
| adaptive/no-sharing cyclic-memory | **12/25** | 41/100 | 14313.88 | 14313.88 | 3,240,256 | 3.30 h |
| static/sharing raster-axial | **12/25** | **46/100** | 19887.72 | 15613.32 | 2,309,916 | 1.43 h |
| adaptive/sharing strand-graph | 11/25 | 44/100 | **13838.68** | **13838.68** | **2,113,568** | **0.86 h** |
| adaptive/no-sharing strand-graph | 9/25 | 31/100 | 15248.16 | 15248.16 | 2,453,579 | 1.17 h |
| adaptive/no-sharing raster-axial | 8/25 | 28/100 | 17818.72 | 17141.96 | 2,665,604 | 1.74 h |

Cyclic-memory is the strongest individual under the predeclared
coverage-first ordering. Adaptive/sharing strand-graph has the best individual
L1000 quality and compute efficiency. Static/sharing raster-axial ties the
coverage lead but obtains much more expensive routes. The five-scientist
portfolio covers 13/25 and has L1000 mean 12839.44, demonstrating one additional
solve beyond either coverage leader.

Four of the 25 supported knots remain open in the pinned KnotInfo snapshot:
`13a_236`, `13a_271`, `13a_1069`, and `13n_4588`. The portfolio used five
crossing changes on each, against current upper endpoint three, so this run
produced **no strict current upper-bound improvement**. This is a negative
result on workbook-PD representations, not evidence against later scientists,
larger-strand networks, or evaluation of the authors' inflated diagrams.

The machine-readable leaderboard, exact solved-set differences, portfolio
witnesses, and upper-bound checks are in `leaderboard.json` under the immutable
artifact root. Its SHA-256 is
`9dac80495135276498d73dbe07bc07d4920b9a24ec7c83093e46623f8b7c5272`.
