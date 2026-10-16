# Local MPS metric-backward parity and cost gate

2026-09-21. Engineering-only check before considering future long-view training.
No optimizer, policy training, MCTS change, cloud resources or checkpoint update.
Local Torch2.13 reports MPS available. Keep protocol97 CPU evaluation frozen.

Use both final97 model/objective checkpoints. From123's96 saved TRAIN variants,
select one exact-positive-distance pair at each1/4/16-round level: maximum summed
endpoint length, breaking ties by lower parent index. This selection is based on
geometry/labels only, before new neural outputs. It is a heavy-case cost probe,
not a representative validation set or estimate of generalization.

For each of the six arm/pair cases, private CPU and MPS float32 copies retain
identical weights. No persistent tile cache. CPU one thread, original tilebatch64
and architecture. Run two unmeasured warmups per backend, then three measured
forward/backward repetitions, alternating backend order by case/repetition.
Synchronize MPS before/after the timed region. Time the existing single-pair
metric-loss path including tensor preparation/transfers and CPU adjacency
preprocessing, excluding model transfer/loading, gradient extraction, comparison
and serialization. AdamW, policy/relative tasks and view generation are excluded;
do not claim full-training or MCTS throughput from this probe.

Fail closed on unavailable MPS, unsupported operators, nonfinite values or any
changed model/objective state. Explicitly disable automatic MPS fallback; the
encoder's documented CPU topology preprocessing is still allowed. Do not
silently substitute CPU neural computation or relax tolerances after results.
Record errors and completed partial measurements if a backend fails.

Predeclared numerical checks: metric loss and predicted distances must agree at
rtol1e-3, atol1e-4. Full parameter gradients, and separately local raster, shared
graph/pooling, knot head and metric-objective gradients, must have relative L2
error<=0.01 and cosine>=0.999 when the CPU norm exceeds1e-6. For reference norm
<=1e-6 require absolute L2 error<=1e-5 instead and report undefined zero-vector
cosines as null. Remaining task-disconnected blocks must also pass this rule.
This is approximate FP32 parity, NOT bitwise/optimizer-trajectory equivalence.

Save every measured gradient vector and scalar output with source/checkpoint and
view hashes. Recheck state bytes and pinned files. A pass establishes only these
bounded metric forward/backward cases, not whole-training backend readiness,
better embeddings, checkpoint promotion, or arbitrary-size solvability.
