# Frozen Q254 is a mid-stage snapshot; bounded root-overlap audit completed

2026-09-21. Protocol101 completed without changing training, queued evaluation,
TEST membership, native PGX files or frozen model weights.

Artifact: `artifacts/q254-registered-root-exposure-v1/report.json`.
SHA256: `520063c60da8271ae2b2f9711db30bccf822ceed0c58e725aba045b48d9e9029`.
All1094 input/source pins were independently rechecked after completion.
Three focused tests passed for prefix identity, native bank hash semantics and
the distinction between invariant collisions and exact cyclic-word matches.

## Correct interpretation of the transferred weights

The frozen checkpoint is still exactly
`36d1122f5494d502e556994083a1a69adf1643d5be95cd9f80ffc13b68e68d63`.
However, its Q254 label names the generation/stage, not its terminal state.
It was frozen on2026-08-24 after23 completed roots in a50-root continuation whose
registered prior bank contains204 roots. The ordered23-root prefix reconstructs
the exact frozen metadata hash
`357573e412b7859586b45379165480f1e2ba8deb63b862c4d22f22b33119f0c7`.

The current later source report completed50 roots. Its current state hash is
`344411bcc6b374a5a9e615a21fa84c7ec42c3fa8d89bb7e0ceb9bece685f2bca`,
not the freeze's historical
`5cb60218a2665482c0b2695cc98ee79cb17c6f4c94ce8d14f1101a460bfdb0aa`.
This is expected later progression, NOT corruption of the verified frozen model.
The inherited Q204 state hash and both registered bank hashes also matched.

Consequently describe the ongoing transfer as the **frozen mid-Q254-stage
raster snapshot**, not the final Q254 policy or a certified best Q-Run model.
Processed-root counts do not enumerate all optimizer updates or MCTS states;
partially processed next-root exposure is not ruled out by this metadata.
Do not silently replace this checkpoint in the running matched experiment.

## Measured registered-root overlap

Exact determinant and signed signature were computed for all254 registered
root records. Comparisons use determinant/absolute-signature groups, keeping
mirror collisions conservative. Both227 prior-plus-completed-prefix roots and
all254 registered roots gave the following results for TEST queries:

| Query endpoints | Count | Fingerprint collisions | Exact cyclic-word matches |
| --- | ---: | ---: | ---: |
|70 original short-u left queries |64 |0 |0 |
|70 zero-CC lifted left queries |64 |0 |0 |
|74 general-distance left queries |126 |0 |0 |

Among74's126 right/reference endpoints, one fingerprint collision occurs at
panel index117: fingerprint(149,0), registered root `11a_272`. There is no exact
cyclic-word match. This is not a knot-identity certificate or proof of leakage.

All70 right endpoints are explicit unknot anchors and are not claimed novel.
Their fingerprint(1,0) collides with a later registered stage root, but not with
the227-root prefix subset. Again, that collision does not identify the registered
root as an unknot; determinant/signature are incomplete invariants.

## What remains unproved

There are no query fingerprint collisions within the checked registered roots.
This does NOT certify full Q-pretraining independence: inherited pretraining,
all replay/MCTS states, and crossing-change descendants were not enumerated.
Accordingly the machine-readable report explicitly sets
`independence_proven=false`. Existing RF TRAIN/TEST audits and this bounded
historical check are different claims; neither replaces the other.

The queued97 summary remains unchanged and already disclaims full Q exposure
auditing. Read this provenance supplement alongside it when interpreting transfer
benefit. Both arms still have embeddings, so the experiment does not isolate the
causal benefit of adding the embedding channel.

At the closing observation the Q254-local worker remained active at208/2048
saved updates; the13-job evaluator was waiting for both completed training arms,
and the earlier70 MCTS run had48/128 initial-model conditions committed.
