# Completed69/70/74 metric results; search evaluation still pending

2026-09-21. This note covers ONLY the completed metric outputs from the old
graph-mixture experiment. It is not the final combined76 report, a new raster97
result, checkpoint promotion, or an MCTS success claim. The new97 protocol,
training and13-job evaluation were frozen before these scores were inspected;
they remain unchanged.

## Verification and exact scope

The74 controller is completed3/3. Its stable inputs, gate inputs, all three
final-checkpoint reports, report identities, prediction-derived summaries and
comparison hash were rechecked with its existing verifier. The first six
committed70 jobs, including the final paired-training audit and all three
short-u metric reports, also passed their existing output verifiers. The
remaining70 MCTS/replay jobs are unfinished and are not summarized here.

The three models are the common RF joint57 checkpoint before69 continuation,
then graph-mixture and matched-control models after4096 updates each. "Initial"
below means that trained RF starting checkpoint, NOT random initialization and
NOT Q-Run weights. Graph continuation adds the audited ordinary-CC graph pair
mixture; control replaces those draws with the original metric source while
retaining the common suffix, sentinel and policy supervision.

## Short-u:64 queries, including32 exact1 and8 exact2

All numbers are mean errors in distance units; lower is better. Interval error
is distance outside the certified interval, zero inside. The remaining24 queries
have interval[1,2], not exact distance labels.

| Model | Interval error | Exact1 MAE | Exact2 MAE |
| --- | ---: | ---: | ---: |
| Initial RF57 |0.387907 |0.663971 |0.189257 |
| Graph mixture |0.397208 |0.664855 |0.187392 |
| Control |0.369314 |0.614497 |0.195966 |
| Always predict1 |0.125000 |0.000000 |1.000000 |

Graph minus control interval error is+0.027894 across64 distinct fingerprint
groups:25 improve,31 worsen,8 tie. Graph is slightly better on the eight exact2
queries, but not on the overall metric or exact1 subset. All three learned
models lose to the constant1 baseline on aggregate interval error. That simple
baseline does not distinguish exact1 from exact2, so retain both aggregate and
exact-subset results rather than treating it as a useful distance model.

As a descriptive ranking diagnostic, the fraction of exact1/exact2 query pairs
ordered correctly (half credit for ties) is0.839844 initial,0.875000 graph and
0.886719 control. These256 cross-query comparisons are NOT256 independent
observations and are not a significance test. They also do not measure ranking
of candidate moves within a search node.

## General distance:126 pairs,120 query fingerprint groups

The panel has32 exact0 coordinate checks,32 exact1,30 exact2 and32 interval
pairs. This synthetic pure-block family is kept separate from short-u.

| Model | Interval error | Exact1 MAE | Exact2 MAE |
| --- | ---: | ---: | ---: |
| Initial RF57 |0.915816 |0.973550 |1.933216 |
| Graph mixture |0.913181 |0.965236 |1.948432 |
| Control |0.903921 |0.963581 |1.919026 |
| Always predict1 |0.492063 |0.000000 |1.000000 |

The graph model predicts mean0.034764 on exact1 pairs and0.051568 on exact2
pairs; maxima are0.139139 and0.252526 respectively. This is severe underestimation
on this general-distance family, not evidence of useful absolute calibration.
It is NOT global collapse of every embedding: the same model's short-u predictions
are much larger, and the general exact1/exact2 ordering fraction is0.670833
(initial0.731250, control0.744792).

Graph minus control equal-fingerprint-group interval error is+0.011220:31 groups
improve,74 worsen and15 tie. On positive-lower-bound pairs alone the graph error
is1.278453 versus0.333333 for constant1. Near-zero error on the exact0 coordinate
checks (approximately7.05e-8) must not obscure this failure on positive distances.

These observations do not identify a single cause. Dilution of a local crossing
change by a shared complicated background is one hypothesis, not a verified
diagnosis. Recalibrating a scalar on this TEST set would not be an independent
validation and is not being done.

## Important size-shift interpretation

Every saved prediction is identical between70's original and lifted views for
each old model. The actual checkpoint architecture has `normalized_knot=true`:
its knot branch normalizes the word internally even though the evaluator's
external `normalization` field says `none`.

A direct read-only check found64/64 equal normalized left views. Original raw
lengths are64..407, lifted lengths256..420; the shared normalized inputs still
range52..398 crossings and3..33 strands. Thus this does not mean the network
only saw short inputs, but the lifted-view equality itself is an architectural
preprocessing result, not an extra learned length-generalization success.
The new97 raster encoder uses raw diagrams; its paired initialization comparison
is internally controlled, but old-versus-new scores also differ in preprocessing.

## Consequences, without changing the running experiments

- Do not promote graph-mixture continuation as a general Gordian-distance
  improvement: these completed metrics do not support it.
- Do not conclude that embeddings are useless. There is a limited ordering
  signal, severe calibration/distribution limitations, and search outcomes are
  still pending. This is one seed and specific synthetic families.
- Finish the old MCTS/replay matrix and the already running matched raster97
  experiment. No current sources, weights, training schedule or TEST membership
  are changed after seeing these scores.
- If a later experiment addresses local-CC sensitivity at varying background
  complexity, use TRAIN-only diagnosis and a newly reserved confirmation panel;
  do not present reuse of these observed TEST scores as a fresh validation.

## Immutable report identities

General comparison:
`artifacts/pure-braid-short-distance-evaluation-v1/comparison.json`
SHA256 `33e21cad118f15781504ef1b4ee464ed821516be96774c1350709aaa23ff66b9`.

Short-u reports under `artifacts/graph-mixture-evaluation-v1/metric/`:

- `initial-short-u.json`:
  `4c976d18d9257f1850f1e41aaee5d079ca39bec8614797fabde04aab68d1ad68`.
- `graph-short-u.json`:
  `b63f8dd973ec86c364d7254d2931f300d0e66b26ae599e550d67e87df6eaa8b2`.
- `control-short-u.json`:
  `d5f140fc4fc7257105641659c2aed2b2b980f1a4268aa86c33d76522f5619938`.

At the current execution observation the new Q254-local97 worker was advancing
at272/2048 saved updates, with its evaluator waiting. The older initial-model70
MCTS worker was still live at48/128 committed conditions. No partial search score
is used to choose or alter training here.
