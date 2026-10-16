# Longer-word full-policy cost gate, predeclared

2026-09-21. Follow-up to115's lack of short-word speedup. This is a synthetic
inference benchmark, not new training, MCTS or held-out quality evaluation.
Protocol97 and its hourly monitor stay unchanged.

Use both fixed final checkpoints from112, the unchanged111 cache (capacity512),
CPU float32/eval/no-grad/one thread. Compare full encoder+graph+policy forwards
on microbatches of one, including cache signatures and keys. Exclude input-batch
construction, model loading/copying, output comparisons and JSON writes.

Workload grid: (L,N)=(64,5),(256,17),(1024,65), two deterministic families:

1. The structured N-cycle plus inverse-pair construction from110.
2. A stabilized trefoil `(1,1,1,2,...,N-1)` grown to L by inserting inverse pairs
   at pseudorandom positions with seed20261060+L+N, random generator1..N-1 and
   sign. These are synthetic encodings with known construction, not typical
   search-state samples or a test of learned invariants.

For each original word, build a reference-verified five-state edit chain:
original; CC at L/3; CC at2L/3; INSERT(+1,-1) at the middle; positive stabilization.
Root caps:4 crossing changes,128 moves, growth length8, growth strands4.
Use six legal policy candidates at each state: CC at0/middle/last, INSERT with
generator1/positive sign at0 and top active generator/negative sign at last,
and positive stabilization. Set head to step modulo active length and weight1000.
Save all requests/resources/heads and verify every applied transition and candidate.

Three timing repetitions of the identical chain per model/family/size. Each
mode gets a fresh model copy and the cached mode starts empty. Counterbalance
mode order by parity of arm+family+size+repetition (18 pairs each order;36 pairs
total). Warm up the loaded uncached model twice on `(1,)` before copies. Keep
per-state timing, actual CNN tile counts, cache counts and retained key/tensor
payload sizes. Payload excludes Python-container/allocator overhead and temporary
forward tensors; do not call it process peak memory.

For every paired state, compare encoded tokens, all global heads, legal logits,
normalized probabilities and exact masks at fixed rtol1e-5/atol1e-5. Record argmax
differences separately, retain failures and do not relax tolerance. Verify all
weights/objective tensors and source/checkpoint hashes after execution. Use a
new immutable output directory with per-pair artifacts and a final summary.

Report paired timing distributions by size/family/model, separating cold first
state from subsequent edits. This remains a workload-specific measurement under
concurrent host load, not a universal speedup, a MCTS-throughput projection, an
embedding-quality result, or permission to modify the active scientific matrix.
