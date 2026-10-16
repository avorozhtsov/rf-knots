# Raster pair: complete distance matrix; search still pending

2026-09-21. All eight protocol97 initial/final distance reports are complete.
The evaluation controller has completed9/13 jobs including the training audit,
and started Q254-final MCTS. This is a distance-only interpretation, not the final
13-job summary. No training/search contract was changed from these TEST scores.

## Verification

All nine completed job records were checked against their committed hashes and
input pins. For each metric report, model/state/panel identities, exact query
counts, per-query certified intervals and fingerprint groups were validated.
Statistics and constant controls were recomputed with the final-summary helpers;
interval violations and equal-group means were also independently recounted
directly from the predictions. All predictions are finite and nonnegative.

Root: `artifacts/pretrained-raster-log-evaluation-v1/metric/`.

| JSON report | SHA256 |
| --- | --- |
| q254-initial-short-u | `b150409daedd92a1abcc0d91cdbc69c67285f2a30f265848eb51b4807042fb6d` |
| q254-initial-general | `013141836f02c040ae53d129fd90054224fd671ad8fdf58fe7a87c55b3f55dc9` |
| q254-final-short-u | `e4ce1a41cfc678f936b521cccfb1a3844a84be7f7eb7146b12cff0045861399e` |
| q254-final-general | `07898b85ecca00dc2c2e2bcd564c51cc2b59f7ac1ab64c96af4fe389415c98c1` |
| random-initial-short-u | `1b28c761e45f9ed7206637dd511a2eb5857bd047e9707da336726f2ad95b7850` |
| random-initial-general | `6b87325d0524fa91881c6a176d22a0705dcb1ab93269ac344d26f4c12e59a241` |
| random-final-short-u | `1515fb374fa84bb819aea77939a54f0191c8d15cf85e6644a90e1fdf32d02dc1` |
| random-final-general | `e11e655b305fc6184d2b9f19fcc93561b7fa0b6b20374bc40db640fcf6244542` |

## Calibration

Equal-group mean distance outside the certified interval; lower is better.
Short-u has64 roots plus64 zero-CC lifted views, not128 independent knot types.
General distance has126 pairs in120 fingerprint groups.

| Initialization / state | Short-u original | Short-u lifted | General distance |
| --- | ---: | ---: | ---: |
| Q254 / initial | 0.156534 | 0.168295 | 0.954959 |
| Q254 / final | 0.279559 | 0.276860 | 0.905518 |
| Random / initial | 0.167398 | 0.131110 | 0.956831 |
| Random / final | 0.200404 | 0.261296 | 0.910908 |
| Constant prediction1 | 0.125000 | 0.125000 | 0.494444 |

Both arms improve general-distance calibration versus their own initial weights,
but worsen short-u calibration. All remain worse than constant1 on these three
aggregate interval-error measures. Constant1 is a distribution-specific control,
not a generally correct metric or proof that no useful ranking signal exists.

Final Q254 minus final random equal-group error:

- Original short-u:+0.079155;17 groups improve,34 worsen,13 tie.
- Lifted short-u:+0.015564;20 improve,24 worsen,20 tie.
- General distance:-0.005390;68 improve,37 worsen,15 tie.

General positive distances remain severely underestimated. For exact-distance1
pairs, final Q254 predicts mean0.031768 (maximum0.071674), and random predicts
mean0.023021 (maximum0.074322). For exact-distance2, the means are0.103356 and
0.109232, with MAE1.896644 and1.890768 respectively. Near-zero errors on the32
exact-zero coordinate pairs do not establish positive-distance calibration.

## Ordering is distinct from calibration

Only disjoint certified intervals define a target ordering. The short-u ordering
uses256 exact1-versus-exact2 comparisons; general ordering uses3960 comparisons.
These are dependent cross-pairs, not independent statistical observations.

| Initialization / state | Short-u original | Short-u lifted | General all | General positive-bound only |
| --- | ---: | ---: | ---: | ---: |
| Q254 / initial | 0.746094 | 0.734375 | 0.943939 | 0.802083 |
| Q254 / final | 0.718750 | 0.503906 | 0.938131 | 0.789583 |
| Random / initial | 0.824219 | 0.718750 | 0.940404 | 0.785417 |
| Random / final | 0.738281 | 0.359375 | 0.959848 | 0.848958 |

The positive-bound general subset has960 comparable pairs. Reporting it avoids
mistaking easy ordering against exact-zero coordinate pairs for uniformly strong
positive-distance ordering. Substantial ordering is already present before
training, including the fully random model; these numbers alone do not show
that training learned complex knot invariants. Neither cross-root ordering nor
absolute distance calibration directly establishes action ranking during search.

The raw-raster model does not share the old encoder's normalization-based equality
between originals and lifts. Lifted short-u ordering deteriorates in both arms;
the architecture's ability to accept large inputs is not learned size invariance.

## Decision and limits

No consistent distance advantage from this Q254-local transfer is demonstrated:
its small general-error advantage coexists with worse short-u errors and weaker
general ordering than the matched random-final arm. This does not establish that
Q pretraining or embeddings are useless, nor identify the cause of these errors.

Continue the already frozen MCTS comparison and both replay/cost audits. Do not
promote a checkpoint, rescale distances using TEST, stop a branch selectively,
or modify targets/budgets based on this matrix. Both arms use embeddings and have
the same upper architecture; this is not an embedding-versus-no-embedding test.
One training seed, shared70/74 TEST panels, synthetic families and incompletely
audited Q historical exposure limit generalization claims. The independent
search utility requirement remains open until the remaining four jobs finish.
