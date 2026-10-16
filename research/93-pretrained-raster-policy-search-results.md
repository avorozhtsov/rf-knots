# Actual Q254-raster network connected to policy training and cost MCTS

2026-09-21. Protocol92 completed, exit0. Report:
`artifacts/pretrained-raster-policy-search-gate-v1/report.json`, SHA256
`a6b10a71da3e7394f4c6f9c1deee55ee3472e77f97221b06e076d3307983648c`.
All1083 transitive source/input hashes rechecked after completion. Protocol88
initialization tensors remained exactly unchanged; zero optimizer updates.

## Implemented interface

`scripts/pretrained_raster_policy.py` consumes the new network's named outputs,
not the old policy/value tuple. It validates action legality/resources, exact
effective masks, finite scores, microbatch ordering, and mode restoration.
Failures do not silently become uniform policies. Optional integer head context
is supported; the headless global RF search does not invent a moving head.

Policy and distance-prior results contain only action priors. Accessing `.value`
raises explicitly. The existing terminal-cost MCTS and policy continuations ran
successfully with these objects, demonstrating that this search path does not
consume a fabricated probability of success.

The raw-word distance wrapper separately embeds the current diagram and empty
one-strand anchor. It does not call the old normalized-view cache. All embeddings
are local to one inference call, so semantic edits and optimizer updates cannot
reuse stale neural data. This deliberately prioritizes correctness; persistent
frozen-model caching would be a separately tested performance improvement.

## Real supervised backward

Audited selected positive states: lengths3,3,3,3,154, with positive-set sizes
2,2,2,2,1. They are fixed, score-blind selections from TRAIN. The complete
relative group contains52 states, total raw word length326. No censored-search
failure is included as a negative, and no single arbitrary witness ordering is
required where multiple best-known actions are available.

Positive mean loss28.23235 and signed relative loss6.33333 describe the NEW,
untrained policy head; they are not training improvements. Full positive and
relative backward took0.887 seconds on one CPU thread in this selected case.
This is not representative of the largest relative group or the training mix.

All eight gradient groups were finite and nonzero: copied raster trunk, local
projection, strand graph, multiscale context, pooling, global heads, positional
projection and policy head. Microbatch2 and microbatch1 preserved the declared
losses within5e-5 and parameter gradients within rtol1e-3/atol5e-5; maximum
absolute gradient difference0.000416. These are float32 numerical tolerances,
not a claim of bitwise equality. No optimizer was applied and no trained
checkpoint was saved.

## Actual search and complete trajectory replay

Four simulations per root/lambda, four conditions in total, no oracle incumbent
or reference action injection. Both one-crossing roots yielded their direct
one-move witness (L10=L1000=1). Neither trefoil condition was solved. This is an
engineering gate on trivial roots with an untrained head, not a solver benchmark.

All12 continuations were replayed, including the censored ones. Each stopped at
candidate-pool exhaustion under the visited-state filter; this is NOT evidence
of unsolvability or complete-search exhaustion. Direct tree terminal transitions,
not those continuations, supplied the two one-crossing witnesses.

Actual counts:20 policy state evaluations,11 policy forward batches,101 policy
action scores;28 raw tree/anchor encodings in15 encoder batches,52 tree action
scores. The small search loop took0.0843 seconds, excluding model/input loading,
supervised-gradient checks and independent transcript audit. Neither throughput
nor solve rates may be extrapolated to long hard words.

Six new interface tests passed. Combined focused tests for tiling, global
architecture, metric accumulation, policy interface, cost continuation search,
embedding priors and supervision:54 passed. Targeted Ruff passed. Tests cover
nonfinite/illegal rejection, integer head context, raw-view/parameter freshness,
mode restoration, microbatch positive/relative loss and gradient consistency,
and real terminal-cost search with value access prohibited. An initial float32
comparison tolerance was corrected before the real-weight gate; actual errors
are reported above rather than hidden as exact equality.

## What remains

This removes the policy/search integration blocker for a genuine matched
Q254-local-initialized versus random-local-initialized training run. The graph,
embedding and policy heads still need training. Hold upper-layer initialization,
training data/draws, effective loss, optimizer and evaluation fixed between arms.
The old graph/control experiment and its independent evaluation queues must
finish unchanged; engineering success here does not replace their outcome.

Do not reuse old TEST feedback for selecting this new architecture and label it
a fresh result. The Q254 pretraining exposure also needs explicit treatment:
RF fine-tuning split independence does not by itself prove unseen knot types
relative to all historical Q training. Report that limitation when evaluating
the pretrained arm. A useful-distance/search claim still requires completed
training, independent scores and replayed L10/L1000 outcomes.
