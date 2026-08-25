# R200 paired snapshot after heterogeneous progress

Date: 2026-08-16. Status: **preliminary, frozen at 2026-08-16 00:04 UTC**.
This document supersedes the numerical snapshot in
`2026-08-13-interim-r200-innovation-and-objective-ablation.md`; it does not
replace that historical record.

All five services remained active while this snapshot was read. Completed
representation counts were 59 adaptive/no-sharing, 72 adaptive/sharing, 70
static/no-sharing, 39 static/sharing, and 20 static-random/no-sharing. Every
comparison below is recomputed on exact common representation identities.
Raw prefix averages are deliberately absent.

## Paired outcomes

`Coverage` is portfolio representation coverage on the common panel. `Capped
mean` uses 328 for an unsolved L10 representation and 20128 for an unsolved
L1000 representation. `Innovation` is the sum of the three within-comparison
scientist innovation scores. `Evals` includes native learning/evaluation and
selection probes on the common panel; the event schema does not contain
per-round wall time, so no paired wall-time estimate is invented.

| comparison | objective | common | arm A coverage / capped mean / innovation / evals | arm B coverage / capped mean / innovation / evals |
|---|---:|---:|---|---|
| static no-sharing (A) vs sharing (B) | L10 | 39 | 34/39 / 95.15 / 2.4888 / 34,607,927 | 34/39 / 92.87 / 2.4258 / 73,618,503 |
| static no-sharing (A) vs sharing (B) | L1000 | 39 | 34/39 / 6904.56 / 1.4120 / 34,607,927 | 34/39 / 6163.67 / 2.3335 / 73,618,503 |
| adaptive no-sharing (A) vs sharing (B) | L10 | 57 | 52/57 / 92.44 / 1.8150 / 80,138,674 | 56/57 / 72.12 / 4.0630 / 59,783,008 |
| adaptive no-sharing (A) vs sharing (B) | L1000 | 57 | 48/57 / 7080.61 / 0.6996 / 80,138,674 | 55/57 / 5684.65 / 9.4995 / 59,783,008 |
| static control (A) vs static-random/no-sharing (B) | L10 | 6 | 5/6 / 97.17 / 0.0726 / 17,657,782 | 5/6 / 97.50 / 0.0799 / 6,915,924 |
| static control (A) vs static-random/no-sharing (B) | L1000 | 6 | 5/6 / 6363.83 / 0.3481 / 17,657,782 | 5/6 / 7197.67 / 0.0996 / 6,915,924 |

The larger adaptive panel now favors sharing on both coverage and capped
quality. This reverses the direction of the much smaller 18-identity snapshot
from 2026-08-13 and is precisely why the earlier result was not promoted as a
treatment conclusion. Static sharing has similar coverage to static
no-sharing but better L1000 portfolio quality at more than twice the measured
network evaluations. The random-order comparison remains only six identities:
its solved sets are identical and it is not yet scientifically decisive.

## Exact identity accounting

Static sharing/no-sharing has 39 common completed identities. Static
no-sharing has 31 additional completed identities; sharing has none. On both
objectives, the common-panel portfolio intersection has 33 identities,
no-sharing-only solve `11n_82`, and sharing-only solve `11n_111`.

Adaptive sharing/no-sharing has 57 common completed identities. The
no-sharing-only completed identities are `12a_184` and `12n_664`. The
sharing-only completed identities are `11a_142`, `11a_151`, `11a_25`,
`11a_261`, `11a_284`, `11a_316`, `11a_348`, `11n_11`, `11n_135`, `11n_25`,
`11n_82`, `12n_531`, `12n_615`, `7_7`, and `9_20`. On L10 the sharing-only
portfolio solves are `11a_257`, `11a_282`, `11a_308`, and `11n_185`; there are
no no-sharing-only solves. On L1000 the sharing-only solves are `11a_274`,
`11a_282`, `11a_293`, `11a_326`, `11n_111`, `11n_160`, and `11n_185`; again
there are no no-sharing-only solves.

The static-random/control intersection is exactly `10_123`, `11a_231`,
`11a_284`, `11a_47`, `12a_184`, and `7_1`. The random-only completed identities
are `10_145`, `10_26`, `12a_1`, `12a_139`, `12a_228`, `12a_264`, `12a_278`,
`12a_455`, `12a_483`, `12a_565`, `12a_6`, `12n_157`, `12n_535`, and `12n_55`.
The control-only list is preserved verbatim in the adjacent CSV. Both
objectives have the same five solved identities in both arms; `12a_184` is the
shared failure.

## Provenance

Manifest/event-tree SHA-256 pairs are:

- adaptive/no-sharing: `3b282deecea1be161554dbbcdfa2f8b3bc60831d936bd6596b43c322b87a7337` / `c93f5528db31fc968208430ba59c55f3a887fa2afbd1686fc3576f3191e84af5`;
- adaptive/sharing: `3d26407b4b822da7919eb19a55ab8f037ff6b2521b0372c1a1fad8ada1dc075d` / `64be5e01a87ca436884733fb3397f536e9d8eb4ab0e4f86970d171e7954c124a`;
- static/no-sharing: `a175738c63780ddf550db41b7d961d237ad2d193b6fbb3c1794ac3b853814ed8` / `7925af6dd631db0dd3a0c67b573e17e36bb4ebb9dc99f771f8ebec100da71d15`;
- static/sharing: `5542b89c622782af9e4f976c20f93ad3bdfc437e349dd622099d850da43c1a3b` / `126d09339e9d308cafedc171c792f7f496f9a0fc61b671b629ab941be6760222`;
- static-random/no-sharing: `cbfd8d5d9febc6772113a01444323a448867f788aee6dc6fa1624de5cca60ee8` / `920426b125bf8f2284fad89884b5a17b84cb6784c539ae85fc7388af26d247b9`.

Static-random additionally retains seed `2026081401` and order SHA-256
`c1ca7e85eeb1d4f65d9c3e92e7c82fca12daa614a8ea9a6fe5533593c72ccf44`.
The reproducible summary rows and completed-set differences are in
`data/2026-08-16-r200-paired-snapshot.csv`.
