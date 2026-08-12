# Search and evaluation optimization audit

Date: 2026-08-12. Status: locally verified; not applied to active Nebius runs.

The audit started from `pgx-mcts-bench` commit
`bb886413f21ca549a89a34655d5550c5ea07b75a` with uncommitted Semantic-v2 and
raster-capacity work. The benchmark used the `raster-axial` checkpoint from
`artifacts/launch-bundles/sv2-r24-2193b41-f26fd02/checkpoints/raster-axial.pt`,
R200 representation `11a_282`, objective ratio 10, a 128-native-ply horizon,
64 MCTS simulations per move, four seeded attempts, three PyTorch CPU threads,
and `OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0` on the local Apple host.

## Correctness defect

`FixedWordGame.reset(seed)` deliberately ignores `seed`. The deployed
Semantic-v2 evaluator also used temperature zero and disabled root noise.
Consequently, four nominal `EV4` attempts were exact deterministic duplicates.
The old data contain one effective attempt repeated four times, not four
independent solve observations. The corrected protocol uses independent seeded
Dirichlet noise at every search root and records this choice in the manifest.
Attempts remain paired across treatment arms through their declared seeds.

## Equal-work timing

All timing rows below scheduled 33,280 network evaluations and searched four
full 128-ply failures. They differ only in execution strategy.

| implementation | wall time | speedup vs sequential |
|---|---:|---:|
| four sequential stochastic attempts | 54.29 s | 1.00x |
| batched EV4 search | 33.07 s | 1.64x |
| batched EV4 plus cached serial legality maps | 28.74 s | 1.89x |
| above plus JIT internal-ply transition | 24.23 s | 2.24x |
| above plus exact-observation inference cache | 19.09 s | 2.84x |

The legality cache keys the pure local-action to wrapped-action gather map by
`(head, word_length)`. The JIT change compiles the internal move's native-budget
transition once per game instance. Both preserve the semantic action sequence
and objective definition.

An instrumented 16-simulation run observed 2,090 NN input samples but only 826
distinct observations: 1,264 samples (60.5%) were duplicates. The accepted
search cache stores policy logits and scalar value for the exact encoded
observation, while legality is still applied separately at each node. It is
bounded by an LRU limit and invalidated whenever PyTorch parameter version
counters show that an optimizer or checkpoint load changed the network.

A subsequent attempt to `vmap` braid transitions was rejected. MCTS produces
varying active batch sizes, causing repeated JAX shape compilation; even after
warm-up it took 36.83 s versus 24.23 s for scalar JIT transitions. This negative
gate is why batched state transitions are not part of the implementation.

Retention evaluation now batches different fixed representations that share the
same scientist and objective. Their initialization and final verification stay
knot-specific; only MCTS inference is shared. On eight R200 knots at 16
simulations and a 32-ply horizon, equal work fell from 2.86 s to 2.06 s (1.39x).
A sixteen-knot sweep measured 4.81, 3.97, 3.60, and 3.40 s at batch sizes 1, 4,
8, and 16 respectively, selecting a bounded batch size of 16 (1.41x). This path
is used by block-boundary rehearsal and donation-retention guards.

Finally, each expanded MCTS node now retains its ordered action and child lists
instead of rebuilding them on every PUCT visit. This smaller exact change moved
the 64-simulation EV4 benchmark from 19.09 s to 18.70 s.

A warm 16-simulation profile before the final two state-engine changes spent
about 51% of wall time in serial braid transitions/legal views, 32% in neural
forward passes, and the rest chiefly in MCTS selection and bookkeeping. This is
evidence against beginning with a C++ rewrite: batching and removing repeated
Python/JAX dispatch produce larger, safer gains first.

## Training path

The end-to-end `raster-axial-v3` benchmark at stage 1, actor batch 8,
8 simulations, two measured optimizer steps, and three CPU threads reported
46.91 normalized seconds per iteration: 3.68 s self-play, 31.67 s normalized
training, 11.51 s normalized evaluation, and 0.05 s checkpointing. Thus optimizer
work becomes the next dominant component after search batching.

Compiling `forward_with_auxiliary` using `torch.compile(mode="reduce-overhead")`
reduced twelve post-warmup optimizer steps from 2.84 s to 2.57 s locally (9.3%).
The same probe on Nebius could not compile because the VM image lacks
`Python.h`. This optimization is not enabled: it needs a prepared image and a
full numerical/retention equivalence gate before use.

## Validation

The focused search, Semantic-v2, collaboration, raster, tape, and controller
tests passed after the changes: 114 tests initially, followed by 51 focused
state/search tests after JIT compilation. Active Nebius checkouts and processes
were not changed or restarted.
