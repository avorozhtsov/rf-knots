# Local MPS: long-view metric backward gate passed

2026-09-21. Protocol125 passed all18 CPU/MPS comparisons, using both final97
checkpoints and three fixed TRAIN-derived pair sizes. This validates selected
metric forward/backward calculations, NOT whole-training readiness or MCTS speed.
No optimizer step, checkpoint change or cloud-resource operation was performed.
The active scientific evaluation remains on its frozen CPU contract.

## Provenance and verification

Root: `artifacts/raster-mps-metric-gate-v1/`.
Report SHA256:
`f76c09abb2d2a342fb96d12715bd653dcde304295de590a7a203e6f3311a03db`.
Input SHA256:
`382363511fed3ad6bba55f1a681abd96a0ec93c68fc5c297125a57329ff908df`.
Measurements SHA256:
`d3f61e87c8cb2f006cd933d29c96dc4f314835ddc6cf871e82af02b132a96d66`.

Torch2.13.0, CPU one thread, MPS float32, unchanged tilebatch64. Automatic MPS
fallback was explicitly disabled (`PYTORCH_ENABLE_MPS_FALLBACK=0`). The encoder's
documented CPU adjacency preprocessing remains part of the timed path; this is
not a claim that every operation is performed on GPU. No unsupported-operator
failure occurred. The probe process exited0; its two targeted tests and Ruff
passed. No numerical tolerance was relaxed after results.

All1231 pinned files and36 saved gradient tensors were verified. Independent
NumPy calculations recomputed scalar agreement, whole-vector and per-module
gradient errors against125's thresholds. All device-state digests equal their
independently audited final-checkpoint states from118, not merely each other.
CPU/MPS model states remained bitwise unchanged before and after computation.

Largest relative gradient L2 error, across full vectors and module blocks with
non-tiny reference norms:0.0001601 (about0.0161%). Smallest defined gradient
cosine:0.9999999872. Maximum absolute difference in loss/predicted distance:
8.345e-7. These are approximate FP32 parity results, not a proof of identical
optimizer trajectories, continuation sampling or eventual trained weights.

## Selected cases, not a validation cohort

Cases are selected by maximum summed endpoint length among exact-positive123
variants at each level, not by neural score. Different levels may have different
parents. Every endpoint actually passed to MPS here is nonempty; testing augmented
views of an empty anchor is not the same as testing the canonical empty-input
network path. No held-out panel was scored.

| Rounds | Variant / parent index | Endpoint lengths | Endpoint strands |
| --- | --- | --- | --- |
| 1 | 9 / 3 | 247,29 | 6,4 |
| 4 | 25 / 8 | 329,142 | 14,7 |
| 16 | 11 / 3 | 789,537 | 32,22 |

The192-view data gate reached45 strands, but this selected MPS computation gate
reached32, not45. Do not conflate data-generation and neural-backward coverage.

## Synchronized timings

Two warmups per backend/case, then three measured repetitions. Across18 paired
repetitions, CPU ran first nine times and MPS first nine times. MPS was synchronized
before/after timing. There are36 saved measured backward calls plus24 unmeasured
warmups. Times include single-pair metric construction, transfers, adjacency,
forward and backward; loading/copying model weights, extracting gradients and
writing files are excluded. There is no persistent embedding/tile cache.

| Arm / rounds | CPU seconds range | MPS seconds range | Median paired CPU/MPS ratio |
| --- | ---: | ---: | ---: |
| Q254 / 1 | 0.513–0.635 | 0.165–0.216 | 3.17 |
| Q254 / 4 | 0.845–0.998 | 0.262–0.332 | 3.25 |
| Q254 / 16 | 2.748–6.551 | 0.886–1.144 | 5.13 |
| Random / 1 | 0.524–0.549 | 0.157–0.216 | 3.15 |
| Random / 4 | 0.856–0.936 | 0.264–0.330 | 2.87 |
| Random / 16 | 3.084–5.262 | 0.715–1.053 | 4.48 |

For the smaller selected cases, this indicates approximately3x lower measured
metric forward/backward time on local MPS. The long-case CPU timing varies widely;
three repetitions on a shared host do not establish a stable4–5x training speedup.
No specific cause of that variability was diagnosed here. Do not extrapolate an
ETA for millions of examples or the current MCTS from this table.

## Next boundary

Local MPS is a viable candidate backend for the NEXT metric-training contract on
these long views. Before moving complete joint training there, separately check
canonical-empty inputs, the positive-policy loss, full two-pass relative groups,
mixed gradient accumulation and optimizer/checkpoint resume semantics. This gate
did not execute AdamW, generate views inside the timed region, train policy/value
heads, compare learned quality or authorize switching the active search backend.

The outstanding scientific requirement remains the complete paired MCTS/replay
evaluation and its interpretation. A faster validated calculation is useful
engineering evidence, not proof that the embeddings improve short distances or
L10/L1000 and not grounds for checkpoint promotion.
