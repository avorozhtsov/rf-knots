# TRAIN fitting is real, but transfer and class coverage remain weak

2026-09-24. Completed exploratory diagnostic145 after seeing the139 distance
results. This is not a new independent test or a causal attribution experiment.
The existing139 search was left running unchanged; no weights were updated.

## Execution and evidence

`scripts/diagnose_resource_train_fit.py` ran once, session5145/PID33646, exited0.
Its initial CPU use was observed live. It reconstructed all2048 actual135 draws
using the audited samplers, policy-choice consumption and seed20261071. Before
any scoring, its final digest and relative counts matched BOTH completed arms:
`babfad51362e7d8064b3bac8bd3814418e3f834e2b3d84879c456e9a48fab84e`.

The frozen selection is every full metric batch at zero-based updates0,128,...,
1920:512 draw records,434 unique unordered raw pairs, not434 independent knots.
Counts:112 original,128 suffix,256 ordinary graph,16 repeated sentinel. The
largest selected word has206 letters and17 strands. The selection is systematic
and score-blind, not a uniform corpus sample or a generalization estimate.

Shared initialization, units-final and raw-final each performed1024 encoder
calls, total3072, CPUfloat32/one thread/microbatch1. State hashes before/after
matched, optimizer_steps=0, and no native prediction required numerical clamping.
Six new tests plus eight relevant existing tests passed; new files passed Ruff.
The diagnostic used a CPU thread concurrently with the139 MCTS thread, so this
period must not be treated as a controlled exclusive-hardware speed benchmark.

A separate consumer subsequently rechecked input/report hashes, all2048 saved
draw-hash chain links, selected update coverage, every saved prediction's selected
metadata, native squared loss, interval violation, aggregate/source statistics
and the16 weighted-batch loss calculation. It did not repeat neural inference
or replay the entire data loader a second time. Model fit is measured on already
used TRAIN draws; full provenance checks do not turn them into TEST examples.

Evidence directory: `artifacts/resource-units-train-fit-diagnostic-v1/`.

| Artifact | SHA256 |
| --- | --- |
| report.json |`78a80335568b55896f1236bb589fc649b3d1d6fb303a8996a11a9d3f877e9b98`|
| selection.json |`e5f53a1df5745b539282a05aedda7e90ea99b0a3f06e4c7fb0f88f9922a0da92`|
| units-initial.json |`0127bd033e98ef7a930337aa01c7c1913423736258498e382c6ed91a3c395764`|
| units-final.json |`12c233c561374d261669cc23d5bc2d952daaf069503eafb76095ecfa7a43a25f`|
| raw-final.json |`1d9659e9606ff1d95711fbcf6dbd52b4d61fa09ae9d76b61f45439a669e6513b`|

## Actual TRAIN fit

Mean interval violation per draw; lower is better. Do not compare these means
as though TRAIN and TEST had identical label/source distributions.

| Source | Draws | Constant1 | Initial | Raw final | Units final |
| --- | ---: | ---: | ---: | ---: | ---: |
| All |512|0.439453|0.957753|0.314613|0.318775|
| Original corpus |112|0.571429|0.777269|0.468778|0.512403|
| MCTS witness suffixes |128|0.476563|0.488078|0.218873|0.234161|
| Ordinary graph |256|0.328125|1.241829|0.313532|0.296033|
| Repeated mirror sentinel |16|1.000000|1.433309|0.018687|0.004176|

Mean weighted training-batch metric loss fell from0.827687 to0.133111 raw and
0.136712 units. This is the squared loss with actual witness-priority weights,
not the unweighted interval-error column. The effective metric weight fractions
in these batches are33.29% original,43.40% suffix,21.94% graph,1.37% sentinel.
Half the draw records being graph pairs does NOT mean half the metric weight.

Both models learn on their data and beat constant1 on this aggregate TRAIN
sample. But fit is incomplete, and it is not correct to call the failure purely
generalization/overfitting: suffix exact1 MAE worsened from0.081197 to0.213430 raw
and0.280489 units even on TRAIN, while suffix exact0 and exact2 improved. The
large aggregate gain hides a within-TRAIN class tradeoff.

For graph exact2 rows, mean predicted distance improved from0.045781 to1.672452
raw and1.682733 units (target2); for graph exact1 it became0.638898 and0.738649
(target1). This contrasts with general TEST means near0.132/0.076 for exact1
and0.168/0.122 for exact2 (144). These are different cohorts, not matched pairs
isolating size, source, exposure or architecture as the cause.

## A concrete coverage hole in the ordinary graph TRAIN source

The selected256 graph draws all have words no longer than48 and at most9 strands.
This alone would not prove the full pool lacks longer examples. A separate
read-only audit therefore loaded the actual frozen `GraphMetricSampler` support.
Its graph database SHA256, bound by the paired audit, is
`a4554a78573d5ac73dd0ae0a92aa0ea98260ab132cde92c6ae3e58615133ff1b`.

| Graph class | Pool records | Invariant groups | Either endpoint length>48 | Either endpoint length>128 |
| --- | ---: | ---: | ---: | ---: |
| exact1 |23572|895|9|7|
| exact2 |1214|113|0|0|
| interval |25829|426|1|1|
| Total |50615|1434 class-specific groups|10|8|

Group counts are class-specific, not1434 distinct knot types. The entire pool
does extend to length254/strands21, but that maximum is not evidence of broad
long-word supervision. Only10 records extend beyond48; none of the1214 exact2
graph records does. A direct read-only SQLite endpoint query also confirms the
full254/21 maxima. It includes three exact-zero training entries subsequently
excluded by the graph sampler, so its101236 endpoint rows are not the sampler's
101230 active endpoint appearances.

Analytically summing the actual equal-class/equal-group/within-group sampler
probabilities givesP(length>48)=0.003389725 per graph draw and
P(length>128)=0.003203329. This sampler actually upweights the few long records
relative to uniform record sampling; the problem is not simply that group
balancing suppresses them. Repeating it one million times still cannot create
more than the existing10 distinct long graph pairs or any long exact2 graph pair.

This claim is ONLY about the ordinary graph source, not all TRAIN data. In the
selected original112 draws,87 have a long endpoint and16 have both endpoints
long;58 are anchored at the empty unknot. All128 suffix draws are unknot anchors,
107 with a long left endpoint. Those sources do supply longer examples, but
their relations are not interchangeable with broad long-to-long short-distance
supervision. No unsupported assertion that the entire corpus has zero long
exact2 supervision follows from this audit.

## Next evidence-driven direction, not a new run

Preserve the ongoing139 MCTS comparison and finish its replay audits. Separately,
the next data experiment should add diverse long-to-long pairs with replayable
short CC certificates and independently justified bounds, including exact2 when
available, plus verified zero-CC views. Verify real support and per-class size
coverage before spending millions of draws on it. Do not inflate the number of
independent examples by counting duplicates or correlated views as new knots.

Then compare a frozen new mixture against the current data at matched training
resources and on fresh, excluded TEST groups. Explicitly retain short-u and
within-size tasks so apparent long-pair gains cannot hide regression elsewhere.
The present evidence motivates this experiment, but does not prove that more
data alone suffices, that architecture/optimization are adequate, or that a
particular weighting/geometry change is the cure. No residual/value adapter,
new objective, new training run or public database update was introduced here.

Reproduce with the145 script and a NEW output directory; the existing completed
directory is intentionally non-overwritable. Model files, frozen135/139 code,
running search and all existing artifacts were preserved. Routine139 monitoring
remains due no earlier than04:54 UTC; this diagnostic was a separate finite job.
