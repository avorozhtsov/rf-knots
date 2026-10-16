# Graph-mixture: completed distance and search evaluation

2026-09-21. The complete protocol69/70/74 matrix is now available. This closes
the old graph-mixture evaluation, not the ongoing protocol97 Q254-local raster
experiment and not the broader goal of useful embeddings for unknotting.

## Evidence and verification

- Core evaluation completed14/14 jobs; general-distance supplement completed3/3.
- Uniform search completed128 conditions in16 committed shards. Its full replay
  audit passed:768 episodes,377312 continuation moves,377952 full-path moves and
  70361725 recorded policy-action scores. No successful solution was found.
- Uniform manifest SHA256:
  `4508270c047cef529de08d7f38e035f4da685c92ddb1b29ed596ff10d53d3ad7`.
- Uniform replay-audit SHA256:
  `b496d3b8808a7949ece385633152473c3d7d085a05aeb8d2323880887801ae7d`.
- Combined report: `artifacts/graph-mixture-summary-v1/report.json`, SHA256
  `e09843f9455fff501c55f2577df289693591f486cd85ac056eba26a9cf017ad6`.
  Human-readable companion: `artifacts/graph-mixture-summary-v1/report.md`.
- The final-only summarizer verified the complete declared matrices, stored
  output hashes, model identities, shared roots/resources and replay audits;
  it replayed reference/solution witnesses and computed certified cost floors.
  Afterwards all1200 report pins were rechecked, Markdown was checked against
  the JSON renderer, and search coverage was independently recounted from all
  four sets of committed condition rows.
- Audits replay recorded trajectories and costs, not independent wall-time or
  search-tree-work measurements. Uniform committed search batches took about
  40minutes; this is not an isolated equal-compute comparison.

## Search results

All arms used the same64 roots at each lambda, with the frozen eight-simulation
search/resource contract. Initial is the previously trained RF baseline, not a
random network. Graph and control each underwent4096 updates from that baseline.
None of these arms imported Q254 policy weights.

| Arm | L10 successes | L1000 successes |
| --- | ---: | ---: |
| Initial | 3/64 | 2/64 |
| Graph-mixture | 2/64 | 1/64 |
| Control | 3/64 | 2/64 |
| Uniform | 0/64 | 0/64 |

Initial and control solved different conditions despite equal totals. All three
graph successes were also control successes. On their two common L10 successes,
graph costs changed by-10 and+36 (mean+13); on the common L1000 success, by-4.
Thus graph did not uniformly improve either coverage or cost. Costs are compared
only where both models succeeded; a capped unsuccessful search has no observed
solution cost and is not assigned an artificial loss or impossibility label.

## Distance results

These are equal-group interval errors; lower is better. General-distance values
use equal weight per group, so they differ from pair-weighted averages in103.

| Arm | Short-u | General distance |
| --- | ---: | ---: |
| Initial | 0.387907 | 0.920990 |
| Graph-mixture | 0.397208 | 0.919798 |
| Control | 0.369314 | 0.908577 |
| Constant prediction1 | 0.125000 | 0.494444 |

Graph is worse than control on both aggregate errors. Its general exact-distance2
MAE is1.948432 over30 pairs, versus1.919026 for control. Good exact-zero agreement
does not resolve poor positive-distance calibration. Original/lifted short-u
agreement reflects identical internal normalized views, not demonstrated learned
invariance to arbitrarily large raw representations.

## Decision and next step

There is no evidence here to promote the old graph-mixture checkpoint over the
control. The small numbers of search successes, one training seed, synthetic
families and finite search budgets do not establish a general negative result
about embeddings. Neural-versus-uniform differences also do not isolate the
embedding channel from the learned policy.

Do not retune on these TEST results, relabel censored failures as bad actions,
replace production models or publish new database claims from this report.
The original held-out sidecar supply remains unavailable; the pure-block panel
is a separate evaluation family, not a repair of that missing test source.

Continue protocol97 unchanged: finish the matched Q254-local/random-local
training, verify paired initialization/data/accounting, then run its frozen
distance and MCTS evaluation. That experiment concerns a different scalable
raw-raster architecture; it is not a controlled comparison against these old
RF arms. Million-scale online RL, reliable arbitrary-size solving, and the
isolated causal contribution of the global embedding remain unestablished.
