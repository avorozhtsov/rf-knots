# Matched scalable raster training with full-group log preferences

2026-09-21. Supersedes the intentionally interrupted94 as a NEW output lineage.
The change is motivated by TRAIN-only diagnosis95 and all-group real-network
gate96, not by newly scored70/74 TEST metrics. Never resume94 unchanged.

## Unchanged initialization and tasks

Both arms restart from their ORIGINAL matched initializations, not update16:
Q254 spatial trunk versus native random spatial trunk seed20261048, with exactly
the same upper layers from88. Same267657 parameters, all trainable, no value
head. CPU/one thread/float32,2048 updates, seed20261047, AdamW lr1e-4/wd0.01,
global gradient clip1, atomic checkpoint every16 updates, serial q254 then random.

Per step:7 original TRAIN metric pairs +8 suffix/unknot pairs +16 ordinary graph
pairs +one exact chiral distance2 sentinel;8 multi-positive policy states sampled
condition then state. Chord interval loss, witness_priority=true, ranking_weight0,
metric microbatch1 and positive-policy microbatch1. No false failure/value labels
or prescribed single action where the audited target allows alternatives.

## Revised relative objective and compute schedule

For a COMPLETE audited relative-success group, with common state/action terms
already cancelled and fixed initial-reference log probabilities, use95's

    S = sum_i c_i * (log pi_theta - log pi_initial)
    C = sum_i abs(c_i)
    R = 2*C*softplus(-S/C), or0 when C=0.

This has dR/dlogpi=-c at initialization, and bounded magnitude<=2|c| everywhere.
The same common C scales the entire cancelled comparison. There is no separate
normalization of trajectory lengths, exp(log importance ratio), hard ratio clamp,
or nonlinear loss applied to a sampled estimate of S. Gate96 verified every full
group on both the initial and interrupted states, with two-pass agreement and
unchanged weights; full-autograd gradient maximum error was1.836e-5.

The four groups contain334,52,74,54 states and their measured initial backward
costs were53.34,0.56,5.37,2.31 seconds. Mean≈15.4 seconds, versus≈5.87 seconds per
whole sampled94 update. Therefore apply one full-group relative backward every
16th update, with weight16*0.01=0.16. Other updates use only metric +0.1 positive
loss. Before clipping, at fixed parameters, this has the same time-averaged
expected relative gradient as weight0.01 every step. It is NOT claimed equivalent
to per-step updates after adaptive optimization/gradient clipping.

Choose a relative group uniformly every step; only consume it when the scheduled
relative update is due. No eight-row samples or unused row-index draws remain.
The resulting data order differs from94 after its first step, but is exactly
aligned BETWEEN THESE TWO ARMS. Each arm uses its own initial reference scores,
as a consequence of the initialization contrast. Record full-group counters,
actual relative-state count, data-order digest and both RNG states.

Per arm:65536 metric draws,16384 positive states,128 complete relative-group
updates. The actual number of relative states is determined by the fixed seeded
group draws and recorded, not mislabeled as16384. Counts are draws, not unique
knots or new MCTS self-play. Source/sampler/target hashes must remain unchanged.

## Recovery and evaluation

Separate directory `artifacts/pretrained-raster-log-training-pair-v1/`; exclusive
writer locks per launcher/arm; save initial, model/objective/optimizer/RNG state
and counters. Refuse source/contract changes or unrelated final overwrites.
Preserve every interrupted94 artifact and source. No release or GPU rental.

Retain94's prespecified evaluation contract: both saved initial and both final
models on the same audited70 short-u original/lifted and74 general-distance
panels; constant controls, exact0/1/2 and interval/ranking summaries. These are
shared TEST panels, not new independent replications. Q254 historical pretraining
exposure remains incompletely audited; no unseen-knot-type claim relative to Q.

Both final models: all64 short-u roots at L10 and L1000, the exact70 caps/seed,
eight cost-tree simulations and fixed policy continuation/proposal settings.
No value estimate or oracle incumbent; count real encoding/policy/action costs;
replay successes and censored paths; audit semantic cost floors. No checkpoint
selection, stopping decision, loss choice or schedule tuning from TEST scores.
Training completion and successful numerical gates do not themselves establish
useful embeddings or improved L10/L1000.
