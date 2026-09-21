# Paired full-network training with ordinary graph metric data

2026-09-21. Fixed after completed data audit68, before new optimizer updates.
Both arms start from completed joint57 checkpoint SHA256
`e58270a34569bfad11127ef4f69a8e67bcdcf4825213f88bce03a73a2233ef84`.
This is continuation of an experimental lineage, not production promotion.

## Matched arms

4096 steps, seed20261031, CPU/one thread, AdamW lr2.5e-5, weight_decay0.01,
global gradient clip1.0. Same full encoder, embedding and policy heads; scalar
value remains frozen and unused because its target has not been calibrated.
No new parameters, window/strand limits, residual head or model-width change.
Fresh optimizer in both arms, not a mismatched inherited optimizer.

Every step draws from all pools in the same order in both arms:
95 old group-balanced metric examples,32 explicit-unknot TRAIN suffix examples,
64 ordinary graph examples,32 positive-policy states and one relative-success
group when available. Unused draws are consumed in both arms.

- `graph`: first31 old +32 suffix +64 graph +one existing exact chiral
  trefoil-mirror distance2 sentinel =128 metric pairs.
- `control`: all95 old +32 suffix +the same sentinel =128 metric pairs.

Both arms apply metric loss +0.1 set-valued positive-policy loss +0.01
relative-success loss, using the same audited diamond supervision as57.
Relative coefficients still cancel common state/actions before clipping;
clip0.2 refers to fixed supervised reference probabilities, not PPO behaviour
probabilities. No failure/value labels and no graph-local edge used as an
optimal unknotting-policy label.

Graph sampling: exclude the3 trivial canonical-zero pairs from this added
pool; the existing metric pool supplies zero-distance supervision. Uniformly
choose exact1, exact2, or interval; then uniformly choose an unordered pair
of endpoint (det,abs(signature)) groups, then a row. A fourth fixed RNG draw
randomly swaps endpoints, never mirrors them. Raw signed words remain intact.
Uncertain intervals stay intervals; fingerprint(1,0) is not an unknot label.

The comparison measures the complete mixture replacement, including changed
class/task distribution. It does not isolate graph diversity alone. Count
524288 metric draws and131072 policy-state draws per arm; these are not unique
knots, MCTS evaluations, trajectories, or million-scale RL.

## Engineering and completion

Use the already tested exact NumPy bulk adjacency helper in **both** processes,
with process-local replacement restored on exit. Source files remain unchanged;
the helper and all training dependencies are hash-pinned and snapshotted.
Existing CPU bit-equality forward/backward tests are a launch gate. Do not
claim a GPU or end-to-end speedup from this optimization.

Require completed source68 plus its passed independent audit and the full
policy certificate audit. Verify all transitive pins. Save model, objective,
optimizer, Python/Torch RNG and contract every16 steps. Resume only matching
committed progress; never overwrite an unrelated final checkpoint or launch
duplicate workers. Run graph, then control serially. No early stopping,
checkpoint choice by TEST, GPU order, commit, push or deployment.

After completion audit tensors, finite optimizer state, frozen value tensors,
initial identity, sampled counts and aligned final RNG. Evaluate with the
prospective contract70, not a repeat renamed as fresh TEST. Existing tests
remain diagnostic-only if revisited. Do not change protocols during training.
