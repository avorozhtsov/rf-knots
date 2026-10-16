# Resource conditioning: complete fresh distance results, search still active

2026-09-24,03:54 UTC onward. All eight139 distance reports completed before the
controller began its first search at03:15:54 UTC. This is a completed distance
evaluation, NOT completion of the24-job experiment or the broader learning goal.

## Independent consumption of completed reports

The03:54 process check found original controller15420/session78472 alive and
child21909 actively running units-initial-distance search (38m23s CPU in38m29s,
99.1% CPU). Controller's completed prefix contains the four integrity/input/cache
gates and all eight metric reports. Its first search has no committed condition
batch yet; next_condition=0, counts empty. No success count or throughput is
inferred from that still-running batch. No process was restarted or stopped.

A separate read-only consumer verified the complete12-entry prefix with
`run_resource_evaluation.verify_job`, including recorded report hashes and
source/input/checkpoint bindings. It then recomputed every metric report through
`summarize_resource_experiment.validate_metric_report`, checking query identities,
certified labels, individual errors, group statistics, constant controls, ranking
and encoded-endpoint counts against raw predictions. Initial units/raw prediction
lists match exactly on all views. No new model inference, fitting, optimizer
step, TEST selection or label change was performed by this analysis.

## Main result: calibration is still unsuccessful

Mean distance outside the certified interval; lower is better. For an exact
label this is absolute error. These are record-weighted means, in CC units.
Initial is shared by the two arms; raw and units are their fixed2048-step models.

| TEST view | Queries | Constant1 | Initial | Raw final | Units final |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-u original |64|0.125000|0.156791|0.301461|0.410660|
| short-u size-shift |64|0.125000|0.174236|0.305269|0.370740|
| general distance |125|0.488000|0.957717|0.843800|0.875179|

Equal weighting of conservative query invariant groups gives the same qualitative
conclusion. These groups are not established knot-type identities.

| TEST view | Groups | Constant1 | Initial | Raw final | Units final |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-u original |62|0.129032|0.159725|0.302795|0.402062|
| short-u size-shift |62|0.129032|0.175773|0.314370|0.371520|
| general distance |124|0.487903|0.965440|0.850605|0.882237|

Both trained models worsen short-u calibration against their own initialization
and lose to constant1 on all three aggregate views. General distances improve
against their near-zero initial predictions but remain substantially worse than
constant1. Units is worse than raw on all three final aggregate interval errors.
This does not establish that units is worse for policy/search, which is a
different pending comparison.

Paired before/after group changes (better / worse / equal):

- Short-u original: units6/45/11, raw7/44/11.
- Short-u size-shift: units9/39/14, raw12/38/12.
- General: units100/8/16, raw101/8/15.

## Exact labels and ranking do not rescue the aggregate result

Short-u scoring has32 exact1,8 exact2 and24 interval[1,2] queries per view (142).
Original exact1 MAE is0.097721 initially,0.287638 raw-final,0.422764 units-final.
Exact2 MAE is0.863444 initially,0.823675 raw-final,1.023085 units-final. A modest
raw exact2 improvement does not outweigh the deterioration on the other queries.

On the general panel, mean predictions for true exact1 are0.002419 initially,
0.131994 raw-final and0.076496 units-final; for true exact2 they are0.003497,
0.168089 and0.121605. Thus positive distances remain severely underestimated.
Near-zero error on the32 exact0 examples does not imply calibrated positive
distances or invariance under every zero-CC move.

Descriptive general ranking including zeros is92.12% initially,91.35% raw-final
and92.74% units-final. Much of this comparison benefits from separating exact0
from positive labels. A post-hoc descriptive restriction to positive certified
labels gives67.89%,64.76%,69.72% respectively, over928 exact1-versus-exact2
comparisons. These are dependent pair comparisons, not928 independent samples,
and this restriction is an explanatory analysis, not a new selection criterion.

Short-u original exact1/2 ranking is75.00% initially,66.80% raw-final and71.09%
units-final. Size-shift ranking is67.58%,55.86%,51.17% respectively. Neither
trained model demonstrates robust short-u ordering across the shifted views.

## Size and exposure limitations

Original short-u includes40 queries within declared length254/strands21 support,
14 length-only,1 strands-only,9 exceeding both; every lifted view exceeds both.
Even the within-support original subset worsens: record errors0.164473 initial,
0.324071 raw-final,0.477501 units-final. Therefore the overall short-u regression
cannot be attributed solely to crossing the size limits. The single strands-only
query cannot support a broad statistical claim.

General has78 within-support and47 strands-only queries. Final errors improve
versus initialization within both subsets but stay badly miscalibrated. Panels
are fresh and query-separated from audited RF optimizer lineage; complete Q254
pretraining exposure remains incompletely audited. Lifted views are correlated
representations of the original queries, not new independent knot types.

## Interpretation and next action

Explicit resource units alone did not fix short-distance learning in this
matched2048-update static training regime. This is useful negative evidence for
that intervention, not proof that global embeddings cannot help unknotting, that
million-scale learning cannot help, or that the policy received no useful signal.
No inference from a small TEST establishes the training scale needed for a
breakthrough. More search evaluations without optimizer updates are not learning.

Keep the frozen six-case search and replay sequence running. It will compare
initial/final policies at L10/L1000 and final uniform/distance tree priors. Global
embeddings remain policy inputs in every case; the prior contrast does not test
removing the entire embedding channel. No policy/search winner is selected from
distance errors alone, and no TEST examples are admitted to current training.

For the NEXT experiment, this result motivates investigating training-distribution
coverage and interactions between metric/policy objectives rather than treating
raw budget magnitude as the established sole cause. Those are hypotheses, not
diagnosed causes. Any follow-up must be labelled exploratory with respect to this
now-seen TEST and use a new frozen evaluation contract. No new learning run,
reweighting, recalibration or residual head is launched by this report.

The broader goal remains active: useful calibrated distances and reliable
search benefit are unestablished. No public bound or database release follows.
Next ordinary controller observation is no earlier than04:54 UTC (05:54 London),
unless a terminal/error event or explicit user request requires earlier action.

## Immutable metric evidence

Files are under `artifacts/raster-resource-units-evaluation-v1/metric/`.

| File | SHA256 |
| --- | --- |
| units-initial-short-u.json |`96948fc8f2c4ff48c7a19e45ab0854b2504f230ceff60451dd8650f3e3b7b9ed`|
| units-initial-general.json |`4d8e42f601aac26ba449b4552ed971b731ed70cc9e3a7b5eb9f495f542ca948f`|
| units-final-short-u.json |`e973dd0d9664278110b32b09fe4129f9630f1eb4a88cca6254f9ad894d54acbb`|
| units-final-general.json |`9581a96e1309bb5127a59535d99e0da1f9d6df33c80a114ae4b8fe1017c3b8b0`|
| raw-initial-short-u.json |`4fb0a959fa7fda47fee6cb0f64be9767691a4bb6716369a7bfeae8ae5b72e64c`|
| raw-initial-general.json |`2b88a339c71cc12faf6c6261355d3e3589e2241883005d5ec3a544ab0aa3504f`|
| raw-final-short-u.json |`426351a8cca39c992e95651213f16d385cfd9c18a5eecc7363ca4f7b28662f01`|
| raw-final-general.json |`4f43a96cbf618965a3daa1821a11845cbf9c41d508f41d488ec61304204066b7`|

Primary recomputation uses the already frozen functions `verify_job`,
`validate_metric_report`, `compare_metrics`, `metric_strata` and `interval_ranking`;
the final controller will independently reconsume them in the complete report.
All12 completed child bindings were checked before this analysis. The current
controller manifest is live and is deliberately not treated as immutable final
evidence or markedcompleted by this partial scientific report.
