# Raster training: task scales and local gradient disagreement

2026-09-21. Protocol118 completed on four saved states of protocol97, with zero
optimizer steps and no change to weights, training/evaluation code or MCTS.
This is TRAIN diagnosis, not a new accuracy or search-utility result.

## Reproducible evidence

Root: `artifacts/raster-task-gradient-diagnostic-v1/`.
Final report SHA256:
`f4911c95002d5db7cc076e07b633b7324bf9f9db06142a77592e391da494fdb2`.
Inputs report SHA256:
`4b147aa50054f9deb77d7342a9f9bf7cf0902bf4585c9a866f496694263c0b09`.
Selection SHA256:
`3565cdf41970cd791a5b310f309ac314a23ea86611a0fb686e411219a6687271`.

All1218 pinned files were rehashed after completion. An independent read-only
NumPy calculation from the saved gradient tensors recomputed every full-network
and module norm, dot product, cosine, combined norm and hypothetical clip factor.
It checked vector coverage, shapes, absent-gradient zero slices, both original
reference hashes, and each reported unchanged-state digest directly against the
corresponding checkpoint tensors. Four case-report and four vector-file hashes
match. Every two-pass selected-log-probability discrepancy was exactly0.

Vectors contain267659 scalars in68 parameter tensors:267657 model parameters
plus the two metric-objective scalars. The unused RM scalar has no gradient in
this CC-only objective. Do not confuse model count with model-plus-objective.
Four targeted tests passed; Ruff passed. Total measured component computation
was286.02s, excluding data verification, reference construction and summaries.
Process exited0. No trained checkpoint was produced.

Same draw16 of seed20261047 at all four checkpoints:32 metric pairs,8 positive
states (four L10/four L1000), plus EVERY complete relative group. Selection trace
digest `17401ff5fa47321c09857ca59c2b4c6b5b53d36eaab765d826be54920b3b3645`
matches the saved final draw hash. Metric intervals: exact0:4, exact1:8,
exact2:10, [0,1]:1, [1,2]:9. The scheduled group at that draw was group0, but this
diagnostic crosses the same metric/positive batch with all four groups.

Groups0..3 contain334/52/74/54 rows, coefficient masses129.25/23/74/54.
ALL relative rows have L1000 resource context. Thus there is no corresponding
relative L10 comparison group here, although positive supervision covers both.
This is514 relative contexts, not millions of new self-play episodes. Full shared
state/action cancellation and the original candidate pools are preserved.

## Gradient magnitude is not scalar loss

Euclidean norms over all model/objective parameters, after actual task weights:
metric1, multi-positive0.1, each scheduled whole-relative-group0.16. Relative
groups are separate alternatives, NOT four terms added into a single update.

| State | Metric | Positive | Relative0 | Relative1 | Relative2 | Relative3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q254 initial | 3.651 | 213.123 | 4068.688 | 2971.561 | 13898.191 | 6501.324 |
| Q254 final | 3.234 | 6.639 | 773.848 | 2924.792 | 26825.247 | 2954.070 |
| Random initial | 1.982 | 255.862 | 5131.866 | 2864.937 | 15927.301 | 7183.554 |
| Random final | 5.350 | 3.699 | 2.351 | 2302.441 | 27476.996 | 9293.529 |

The initial positive gradient dominates the metric gradient despite weight0.1.
At the final states, their whole-network magnitudes are much closer. Relative
groups generally remain much larger, but random-final group0 is an important
exception, not a result to exclude or average away.

This reaches the shared encoder, not just the output policy head. At Q254-final:
local raster norms are metric0.677, positive0.811, relative83.7..589.6; shared
graph/pooling norms are metric3.152, positive2.852, relative334.5..3715.3.
At random-final: local metric1.884, positive1.296, relative1.433/190.6/1429.0/818.7;
shared graph/pooling metric4.989, positive1.814, relative1.271/236.4/2534.1/1019.6.

For the reconstructed ordinary step, the hypothetical common clip multipliers
are Q initial0.004692/final0.135078 and random initial0.003909/final0.159749.
For the Q final scheduled variants they range3.73e-5..0.001291. Random final
group0 retains0.153202; other groups range3.64e-5..0.0004343. These are calculated
at fixed weights from saved component vectors, not a replay of the actual
optimizer trajectory. Clipping scales the metric contribution as well.

## Direction: a real but non-universal conflict

At Q254-final, the dot product of each reconstructed scheduled total gradient
with the metric gradient is negative: group0 -12.04, group1 -28.95,
group2 -1178.59, group3 -58.66. The ordinary total is positive,+10.60.
For sufficiently small plain-SGD steps, those scheduled directions would locally
increase this particular metric-batch loss. A positive clip multiplier does not
change that sign. The shared graph/pooling metric-versus-relative cosines are
-0.0181,-0.0345,-0.1009,-0.0473: not opposite directions, but large relative norms
make even small negative alignment consequential in the sum.

At random-final, ALL four scheduled total-gradient dot products with the metric
gradient are positive (25.94,42.76,1234.59,475.23). Therefore the evidence does NOT
support saying that relative preferences inherently damage metric learning.
AdamW also uses momenta, coordinatewise preconditioning and weight decay: these
gradient calculations do not prove the direction or effect of actual AdamW steps,
nor causally explain TEST calibration or MCTS outcomes.

## What did learn on this TRAIN batch?

The unweighted metric mean decreases Q0.6803→0.1225 and random0.6531→0.1480.
The unweighted positive mean decreases Q63.2116→0.8753 and random59.1076→0.5881.
This is improvement on a selected TRAIN batch, not generalization evidence.

Whole-group log-preference scores relative to each arm's OWN initial policy:

| Final arm | Group0 | Group1 | Group2 | Group3 |
| --- | ---: | ---: | ---: | ---: |
| Q254 | +281.38 | -130.30 | -896.87 | +26.22 |
| Random | +1163.05 | -286.89 | -1457.92 | -231.31 |

Positive scores favor the audited relative comparison versus the fixed initial
reference; negative scores do worse on that objective. Thus not all relative
TRAIN objectives improved even while the positive mean improved strongly.
Random-final group0 has nearly saturated preference success (loss0.03195), which
explains its small parameter gradient without inventing an implementation bug.

## Source-level mechanism worth testing, not yet established causally

`mcts_loop.build_batch` supplies raw remaining resources and the cost coefficient.
`SiteHead._context_features` concatenates their RAW values alongside log1p values
and exhaustion flags before context FiLM. All four relative groups therefore
include a raw1000 channel. This is an explicit numerical-scaling candidate, not
a claim that raw1000 alone caused the observed norms; no channel intervention
was part of118. The positive batch also includes four L1000 contexts.

The bounded derivative in97 is with respect to selected log probabilities:
`dR/dlog(pi_i)` has magnitude at most2|c_i|. Parameter gradients additionally
multiply the neural log-probability Jacobian. That bound does not bound the
parameter-gradient norm. This distinguishes a correct log-space loss repair
from a complete solution to input/task conditioning.

## Post-hoc resource-column attribution, without new neural computation

Additional analysis of the SAME24 saved component vectors, not a new sampling
or training experiment: `scripts/report_raster_resource_gradients.py` writes
`artifacts/raster-resource-gradient-attribution-v1.json`, SHA256
`775bfa1ff50f0a037f114f2bcf2d91be0046f4f842e13f719e3da14d5005bd6f`.
An independent NumPy calculation verified its norms, column partition, shares
and coordinate-change formula; saved source/case/vector hashes were rechecked.
No neural forwards or optimizer steps were executed. This extension was selected
after examining118 and is explicitly post-hoc, not a preregistered causal test.

The first context linear layer has64x208 weights:192 normalized global-embedding
coordinates,5 raw resource channels,5 log-resource channels,5 exhaustion flags
and1 presence flag. The raw cost-coefficient column is index196, zero-based.
Across the16 relative-gradient cases, raw resource columns account for
99.9861–99.9930% of this layer's squared weight-gradient norm; raw lambda alone
accounts for98.7073–99.8274%. This is a PARAMETER-GRADIENT share in ONE layer,
not feature importance or a percentage of explained search outcomes.

In particular, every relative context has lambda1000. For this first affine
layer, elementary differentiation gives `grad(W_lambda)=1000*grad(bias)` when
accumulating over such contexts. The saved float32 accumulation satisfies that
identity with relative L2 discrepancies1.14e-7..4.80e-6. The large column is
therefore consistent with input units, without requiring a special preference
for that feature to be learned. Small incoming-weight gradients on unit-normalized
embedding coordinates do NOT establish that the embedding is ignored.

The whole-network denominator changes the conclusion: raw lambda's column is
only7.8319–30.4162% of the FULL squared parameter-gradient norm. As an algebraic
sanity check, rescale `x_lambda -> x_lambda/1000` and compensate
`W_lambda -> 1000*W_lambda` to preserve the function. Only that parameter-column
gradient rescales by1/1000 in exact arithmetic. Its hypothetical full norm ratio
is `sqrt((G2-C2+C2/1e6)/G2)`, where G2 is total squared norm and C2 the column's.
Across these cases this ratio is0.8342–0.9600: a4.0–16.6% reduction, not orders of
magnitude. No such intervention was applied to any model or active run.

This function-preserving coordinate change is NOT equivalent under AdamW or
weight decay and says nothing about a fresh initialization using scaled inputs.
It also does not remove the large gradients already measured in shared layers.
Thus consistent input units remain a reasonable hypothesis for a NEW controlled
training contract, but "divide lambda by1000 and the training problem is fixed"
is not established. Neither coefficient adjustment nor embedding removal follows
automatically from gradient norms, which depend on parameter coordinates.

## Next decision

Preserve the current scientific evaluation and wait for its paired MCTS/replay
results under the hourly monitoring schedule. No checkpoint promotion follows
from118; search utility remains unestablished. Do not discard embeddings from
this experiment, automatically shrink a coefficient, add residual-value pieces,
or claim training longer will necessarily fix it.

Before another large training run, a principled candidate is to test consistent
resource-feature scaling and initial policy conditioning on TRAIN/validation,
then separately assess the periodic relative-update schedule. This concerns
the unified architecture/input contract, not an auxiliary residual component.
Keep full-group cancellation, multiple valid positive actions and censored-failure
semantics. Any comparison must be a NEW frozen experiment, with a fresh evaluation
contract, and must not use TEST70/74 to choose scales, losses or checkpoints.

The present limitation remains important: one metric/positive draw, four fixed
relative groups, one seed and endpoint snapshots. It supplies a concrete reason
to audit conditioning before scaling compute, not a universally optimal training
recipe or proof about learned knot invariants.
