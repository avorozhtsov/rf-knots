# Longer-word cache timing: workload-specific benefit, not MCTS throughput

2026-09-21. Protocol116 completed without changing the ongoing scientific
evaluation, its frozen sources/checkpoints or its hourly monitoring cadence.
No training, MCTS or held-out quality evaluation ran in this benchmark.

## Execution and artifacts

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/probe_raster_cache_long_policy.py \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-cache-long-policy-cost-v1
```

Exit0; output directories must be new. Final report:
`artifacts/raster-cache-long-policy-cost-v1/report.json`, SHA256
`0fac8b517383c4ccdbdab532bbc33e308949dd8c3b6aa61d9bd03e64159cf13f`.
The saved complete synthetic requests are in `inputs.json`, SHA256
`f8475ff500f5009d332da59998306aa29d1121f0d432381754f38be3d5ac96b5`.

All143 input/source pins and36 per-pair artifact hashes were independently
verified. The six saved workloads were regenerated from the fixed construction
and compared exactly. Coverage was36 distinct model/family/size/repetition pairs,
with18 runs of each mode order. All aggregate summaries were recomputed from
per-state measurements. Twenty focused tests passed in2.77s without skips;
targeted Ruff passed.

## Correctness and scope

Each pair compares five full-policy forwards along the same reference-verified
edit chain. Both final trained models were used. No weights/objective tensors
changed. All180 state comparisons passed the declared rtol1e-5/atol1e-5. No
argmax action changed. Maximum absolute discrepancies were:

- site tokens:4.6492e-6;
- knot/layout/isotopy embeddings:3.0175e-7/2.9802e-7/2.8312e-7;
- legal logits:2.2888e-5;
- probabilities:2.4438e-6.

As before, passing combined relative/absolute tolerance is not bitwise equality
or a proof of equal future stochastic search paths. Every selected action was
legal, but only six candidates per state were scored, not the full action space.

## Full-policy wall time

Median ratio of uncached time to cached time over three repetitions. Values
above1 favor the cache. Cold is the first state with an empty cache; edits are
the four subsequent CC/CC/INSERT/STABILIZE states. Times include the complete
local/global/graph/policy forward and cache-signature/key costs, not just CNNs.

| Model | Synthetic family | L / N | Cold | Edits | Whole chain |
| --- | --- | --- | --- | --- | --- |
| Q254 | structured | 64 / 5 | 3.142 | 5.044 | 4.514 |
| Q254 | structured | 256 / 17 | 5.481 | 29.983 | 15.636 |
| Q254 | structured | 1024 / 65 | 36.509 | 46.012 | 43.425 |
| Q254 | random-inserted trefoil | 64 / 5 | 0.987 | 3.201 | 2.232 |
| Q254 | random-inserted trefoil | 256 / 17 | 0.999 | 8.426 | 3.378 |
| Q254 | random-inserted trefoil | 1024 / 65 | 1.622 | 1.759 | 1.729 |
| Random | structured | 64 / 5 | 2.946 | 4.880 | 4.333 |
| Random | structured | 256 / 17 | 5.664 | 29.956 | 16.226 |
| Random | structured | 1024 / 65 | 36.303 | 47.489 | 44.911 |
| Random | random-inserted trefoil | 64 / 5 | 0.987 | 3.163 | 2.196 |
| Random | random-inserted trefoil | 256 / 17 | 0.992 | 8.307 | 3.305 |
| Random | random-inserted trefoil | 1024 / 65 | 1.739 | 1.843 | 1.838 |

Across this specifically weighted synthetic workload, uncached forwards totaled
97.216937s and cached forwards27.415377s. Actual CNN tile inputs were80820 versus
20706. Do not use the aggregate as a forecast for production search: family/size
weights, local repetition and repeated near-neighbor edits were deliberately
controlled rather than sampled from real MCTS.

The very large structured-family ratios reflect repeated local tiles and must
not headline a general speedup claim. The less regular inserted-trefoil family
showed whole-chain median ratios1.729–3.378, while its smaller cold cases were
near parity/slightly slower. Even this family is synthetic, and only three timing
repetitions were run under concurrent host load. There is no million-game or
end-to-end MCTS-throughput estimate here.

## Bounded cache storage

At the observed maximum,512 retained entries used262144 bytes of token tensor
payload and172032 bytes of packed tile-key payload (424KiB combined). This
excludes Python containers/integers, allocator overhead, model weights, transient
CNN tensors, global graph tensors and retained comparison outputs. It is not
process peak memory. Uncached tile counts were independently checked against
each five-state chain's total input length `5*L+5`.

## Decision

The prototype now has numerical parity on longer full-policy inputs and a
measured benefit on these controlled edit chains, alongside115's contrary result
on tiny words. This supports considering bounded local reuse in a **future,
separately versioned** inference implementation, not retrofitting protocol97.
No cache-key/weight-invalidation safeguards were removed to obtain these timings.

Stop expanding the cache investigation for now. The decisive outstanding work
is still the main paired search/audits and scientific interpretation of learned
distances and L10/L1000. Cache correctness and speed are engineering evidence,
not a substitute for embedding usefulness, independent quality evaluation,
training gradients, or a demonstrated scalable reinforcement-learning result.
