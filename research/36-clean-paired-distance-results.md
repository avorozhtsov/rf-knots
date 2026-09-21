# Clean paired iteration: completed distance stage, search still pending

2026-09-20. This is a descriptive report of the NINE completed distance jobs
from the frozen note33/35 protocol, not a promotion or the final42-job report.
Both training arms completed2048 steps before any prospective panel scoring.
Final training hashes and matched initialization/RNG checks are in note34.
No training parameter, checkpoint choice or search budget is changed in response
to these results. Remaining search/comparison/portfolio jobs continue unchanged.

## Evidence and independent scope

All nine metric outputs under
`artifacts/embedding-clean-paired-evaluation-v1/metric/` have `status=completed`.
Their bytes were checked against the controller's completed-output hashes;
each reported model SHA equals its pinned final checkpoint. Interval errors
were independently recomputed from predictions and certified bounds using the
read-only summary consumer, not copied blindly from aggregate fields.

The general panel has66 pairs in24 query fingerprint groups:29 exact0,14 exact1,
six exact2 and17 interval-labelled pairs. The short-u panel has64 roots in55
query groups. Previous metric-path and accepted-rollout audits established no
query-group intersection with this declared training lineage/history. These
are synthetic panels; fingerprint groups are not knot identities, and this is
not an audit of every historical experiment ever run.

Original and lifted views give EXACTLY equal predictions in all nine reports
(maximum paired prediction difference0). We report them once below, not as twice
as many independent tests. The lifted views are generated stability checks;
their identical predictions are not evidence of solving genuinely harder long
words or of arbitrary-size extrapolation.

## Mean interval error (lower is better)

Each row reports original-view errors; the lifted view is numerically identical.
`Value` adds BCE supervision of parent-rollout success; `Control` keeps its
coefficient zero. Both also train the shared encoder, relative policy and metric
objective. `Parent` is the frozen pre-iteration model.

| Task / bounds | Parent | Value | Control | Constant0 | Constant1 | Constant2 |
|---|---:|---:|---:|---:|---:|---:|
| General Gordian / original | 0.426197 | 0.444129 | 0.443257 | 0.515152 | 0.530303 | 1.136364 |
| Short-u / original | 0.330315 | 0.314418 | 0.309185 | 0.500000 | 0.093750 | 0.500000 |
| Short-u / determinant-strengthened | 0.330315 | 0.314418 | 0.309185 | 1.093750 | 0.093750 | 0.500000 |

Equal-weight-per-fingerprint-group errors have the same ordering:

| Task (both bounds conventions for short-u) | Parent | Value | Control |
|---|---:|---:|---:|
| General Gordian | 0.436707 | 0.456655 | 0.454950 |
| Short-u | 0.302083 | 0.289186 | 0.284565 |

Thus the iteration slightly reduces aggregate short-u error but worsens general
pair error on this panel. Value supervision does not beat the matched control
on either aggregate metric. In particular, the short-u improvement is NOT a
useful-distance promotion: all models remain substantially worse than constant1
on this constructed one/two-CC panel. A constant's usefulness here does not imply
it is a universal unknotting-number predictor.

## Exact-class absolute errors

| Task / class / count | Parent | Value | Control |
|---|---:|---:|---:|
| General d=0 /29 | 0.082705 | 0.075145 | 0.079120 |
| General d=1 /14 | 0.749141 | 0.799589 | 0.799508 |
| General d=2 /6 | 1.593118 | 1.659328 | 1.638230 |
| Short-u=1, original bounds /9 | 0.585918 | 0.531620 | 0.525868 |
| Short-u=1, determinant-strengthened /32 | 0.584928 | 0.542765 | 0.532925 |
| Short-u=2, either bounds convention /6 | 0.403742 | 0.459044 | 0.455703 |

Strengthening lower bounds increases certified exact-u1 coverage9 ->32 without
changing predictions or the aggregate interval errors in these particular
reports. It must not be described as32 additional examples or as a second test.
Better exact-zero stability and lower u1 error coexist with worse exact-d1/d2
and exact-u2 errors. Small exact classes and correlated representations do not
support a broad significance or topology-generalization claim.

## Post-hoc error direction and ordering diagnostic

This section was computed AFTER the distance results, from the same immutable
prediction rows. It is exploratory error analysis, not a predeclared statistical
test, extra independent evaluation, calibration fit or checkpoint selection.

| Exact class | Count | Parent mean prediction | Value mean prediction | Control mean prediction |
|---|---:|---:|---:|---:|
| General d=1 | 14 | 0.250859 | 0.200411 | 0.200492 |
| General d=2 | 6 | 0.406882 | 0.340672 | 0.361770 |
| Short-u=1, determinant-strengthened | 32 | 1.584928 | 1.542765 | 1.532925 |
| Short-u=2 | 6 | 1.596258 | 1.540956 | 1.544297 |

All exact general-d1/d2 predictions underestimate their targets. Every exact-u1
prediction overestimates1, while every exact-u2 prediction underestimates2.
Within the short-u panel, the two exact classes have almost the same mean score.
The fraction of exact-u1/exact-u2 score pairs in the correct order is0.572917
for Parent,0.520833 for Value and0.541667 for Control (ties count one half).
Those32*6 score comparisons are NOT192 independent observations: each row is
reused and fingerprint groups can recur. No significance claim follows.

An especially clear cross-task ordering problem is visible without choosing a
threshold: the LOWEST exact-u1 prediction exceeds the HIGHEST exact general-d2
prediction for every model:

- Parent:1.424878 >0.756416.
- Value:1.385168 >0.613060.
- Control:1.345011 >0.622002.

Since u(K)=d(K,unknot), those true distance1 cases should be below those true
distance2 cases. A single monotone increasing post-processing of these scalar
scores cannot repair that reversed order. Merely rescaling the current output
is therefore not a complete fix on these panels. This does NOT show that another
training geometry or a better-trained encoder cannot help, nor that useful
information is absent from the full64-dimensional embedding. Search utility is
still being measured separately with the frozen models and protocol.

## Immutable metric output hashes

File basenames below are relative to the metric directory above.

| File | SHA256 |
|---|---|
| metric-parent-gordian-original.json | 79e02e540ff064c5859199731cfea62f702aedfe2578839404af1a30b5d3e1b1 |
| metric-parent-short-u-original.json | cf465f6209f195ad3aa6c26e4c253748c366f70f18ebae8cc64713e1ad914172 |
| metric-parent-short-u-determinant.json | 5e696aa191b303a4e60c6c33b52177c4bd515434c3c820d8e66d1fedeff8c5b9 |
| metric-value-gordian-original.json | edcbdf553a6ade85fca6462499b7892e2bd2b379203c0654f82a7117860112a9 |
| metric-value-short-u-original.json | 5e2a2383d37a3dd0197d803c1e77d00c325b1b04f660df2a55448c37d8ea4423 |
| metric-value-short-u-determinant.json | 3b7cff8a4eeefe6c2b63968d4dbe54c0607e9f8962c5c695d21b23ece3ffeb6f |
| metric-control-gordian-original.json | 6f42078830c285eaf9e654f07723be8cb3de82e1d01b4f43e1c2bee6edb5b422 |
| metric-control-short-u-original.json | 5abe02c32a1b297e082bd172baa1ed7007640e596e2c564ec293bba8f794b07d |
| metric-control-short-u-determinant.json | 185351b7859fb87ca80597dc51b7e86739c744be6334bff9fa7779ab8a92d4b3 |

## Next required evidence

Complete all four anytime cost-search variants, all ten fixed-budget MCTS
conditions, replay-audited paired comparisons and the charged parent-rollout
portfolio comparison. Report real compute and new/lost/common solves, retaining
censored failures. Neither these distance errors nor same-training-root Brier
results establish better L10/L1000. Conversely, this bounded experiment cannot
prove that global embeddings are useless after much larger joint RL training.
