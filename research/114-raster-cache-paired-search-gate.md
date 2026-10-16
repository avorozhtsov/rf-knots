# Bounded paired-search cache engineering gate

2026-09-21. Follow-up to112/113, not a replacement for the active protocol97
scientific evaluation. Its scheduled monitor remains hourly. No current worker,
source, checkpoint or scientific search budget is modified.

Use both fixed final checkpoints from112. For each model, run three pairs with
seeds20261056,20261057,20261058. Each pair uses fresh identical model copies,
one uncached and one with the isolated111 local-token cache (capacity512).
Run serially, CPU float32/one thread, eval/no-grad. Alternate mode order between
pairs and between arms: three cached-first and three uncached-first pairs total.

Four synthetic conditions per run:

- Word `(1,)`, two strands, CC cap0, move cap4, at each weight10 and1000.
- Word `(1,1,1)`, two strands, CC cap1, move cap8, at each weight10 and1000.

For all roots, growth length2 and growth strands1. Four tree simulations, rollout
horizon8, insertion candidates4, widening batch4, cpuct1.5, temperature1. These
tiny fixtures test terminal transitions and bounded continuations, not general
knot difficulty. Both members use identical root order, seeds and resources.

Use the existing raw-raster search function and its complete continuation replay
auditor. Replay all successful and censored continuations and verify witnesses.
Compare complete recorded discrete transcripts, tree statistics, incumbents,
witnesses and L10/L1000, excluding only floating-point chosen-action probabilities
from exact equality. Compare those probabilities with fixed rtol1e-5/atol1e-5
when action transcripts align. Do not tune seeds/tolerances after observing a
difference. Any discrete divergence must be recorded and prevents claiming
trajectory equivalence, even if both searches are individually valid.

Count actual tile CNN inputs using the existing input-convolution forward hook,
separately from logical state/embedding evaluations and cache hits. Check cached
CNN tile counts against cache miss counts. Hash model/objective tensors before
and after; recheck all source/checkpoint pins. Save each run's rows and counters
with hashes, the complete input configuration, run order and final comparison.

Time each full search+replay call, including cache-key and weight-signature work,
but excluding checkpoint load, model copy, JSON writes and common warmup. Warm up
the loaded uncached model on `(1,)` before making the fresh copies; each cached
run starts with an empty cache. Report paired timings, not a universal speedup.
Concurrent host load, three seeds, very short words and synthetic repetition
limit external validity. Timing ratios with divergent transcripts are not
equal-work comparisons.

This is not an embedding ablation, a before/after training comparison, a new
held-out quality result, or evidence for million-game RL throughput. Even a
passing gate does not authorize retrofitting protocol97 or mixing the matrices.
