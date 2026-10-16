# Matched Q254-local versus random-local training of the scalable network

2026-09-21. Prospective protocol after gates88/90/92, before optimizer updates.
The original69/70/74 experiment continues unchanged. This is a first learning
comparison of the real scalable raster/graph architecture, not million-scale RL.

## Initialization contrast

Both arms use exactly the267657-parameter protocol88 architecture and identical
135817 upper-layer parameters from its initialization checkpoint. `q254` keeps
the131840 transferred local spatial parameters. `random` replaces only those
parameters using the native Q254 RasterWindowRepresentation constructor with
seed20261048; input convolution and four masked axial blocks, including native
normalization initialization. Copy no Q policy/value head or Q304 graph weights.
Check parameter names/shapes, exact upper-layer equality and nonzero local
differences before training. Each arm saves its immutable initial checkpoint.

## Training contract

2048 updates, seed20261047, CPU/one thread/float32, AdamW lr1e-4, wd0.01,
global gradient norm clip1. All encoder, global and policy parameters trainable;
there is no value head. Fresh identical optimizer configuration and chord
MetricEmbeddingObjective, witness_priority=true, ranking_weight=0. The metric
scale is learned; no hardcoded CC1 cosine target is introduced.

Per update, identical pseudorandom draws in the same order:

- Seven group-balanced old TRAIN metric pairs.
- Eight audited TRAIN suffix-to-explicit-unknot metric pairs.
- Sixteen ordinary graph pairs from the already audited class/group sampler.
- The existing exact chiral trefoil/mirror distance2 sentinel, giving32 pairs.
- Eight positive policy states, sampled condition first then state, preserving
  each state's complete observed best-known action set.
- One uniformly selected complete relative-success group, then eight rows sampled
  uniformly with replacement. Multiply each already prefix-cancelled coefficient
  by group_size/8. This estimates the original group's SUM without length
  normalization. It does not create new negative states. Unit-test the estimator.

Loss = metric +0.1 positive-set mean +0.01 sampled relative-success sum.
The relative clip0.2 uses each arm's own fixed INITIAL policy probabilities,
which differ as a consequence of the specified initialization contrast; no
claim that these are behaviour probabilities or PPO. Check all reference rows
at initialization and preserve them across resume. Use metric/policy microbatch1
with the exact weighted-mean accumulation tested in90/92. No cross-batch rank
loss is silently dropped. Fixed draws do not depend on the model's scores.

Per arm:65536 metric draws,16384 positive state draws,16384 sampled relative
rows; these are NOT unique knots, MCTS rollouts or a million self-play episodes.
No online MCTS collection during these optimizer updates. No early stopping,
TEST checkpoint selection, architecture change, GPU rental or production release.

## Durability and continuation

Serialize q254 then random under one monitored launcher, with per-arm exclusive
writer locks. Save model, objective, optimizer, Python and Torch RNG, aligned
draw digest and source/input contract every16 steps. Resume only compatible
progress; never overwrite an unrelated final checkpoint. Recheck pinned sources
and completed inputs. Compare final draw digests/RNG and initial upper layers.
This run remains separate from all dirty/generated prior work and69 checkpoints.

## Independent evaluation, committed before results

Use the existing fixed70 short-u panel (64 original queries plus64 size lifts)
and74 general-distance panel (126 pairs), with their passed independence audits.
At protocol creation both evaluation queues were live but no metric result for
these panels existed. They are SHARED prespecified TEST panels, not a newly
generated panel or a new independent replication of later70/74 findings.

Evaluate both initializations and both final checkpoints for interval violation,
exact-distance MAE and rank summaries, alongside constant-distance controls;
report exact0/1/2 separately and retain intervals as intervals. Do not change
training based on the older arm's upcoming scores. Q254's historical pretraining
exposure is not fully audited: RF fine-tuning disjointness is established, but
unseen knot types relative to all Q training are NOT claimed.

For both FINAL models, run all64 short-u roots at lambda10 and1000 with the
same root caps, eight total cost-tree simulations, temperature1, cpuct1.5,
widening_batch4 and four insertion proposals used by70. Raw global distance
priors and the protocol92 policy interface consume no learned value. Use70's
seed20261034 and report actual forward/encoding/action counts and wall time.
No oracle incumbent. Replay successes AND censored trajectories. Use the same
semantic cost-floor audit to distinguish verified optimal costs from upper
bounds; failure remains censored. This remains a small-search-budget comparison,
not a decisive rejection of embeddings if no gain is observed.

Training completion alone is not success. Require final tensor/optimizer/source
audit, all prespecified metric/search results and the explicit limitations above
before any promotion or decision to spend on a larger GPU run.
