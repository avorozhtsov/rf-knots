# 06 — Growing the network across curriculum levels

> **Revised by measurement.** This note assumes capacity is a constraint and sizes a schedule up to
> 30M parameters. The ladder found 7.7× the parameters bought nothing over 2× the simulations, while
> 16 simulations reached stage 0 — search dominates at this scale. The operators below are still the
> right operators; the premise and the schedule are not. See
> [11-network-growth-branch.md](11-network-growth-branch.md) for the executable version, including
> the receptive-field arithmetic that decides where depth growth can help at all.

Question: *are there approaches to gradually increase the complexity of the NN, inheriting
results from previous levels / smaller nets?* Yes, three distinct families. Use all three, in
this order of priority.

## 0. Best option: make the net size-invariant, so you never have to grow it

Before reaching for growth operators, choose an architecture whose parameter count does not depend
on the problem size:

* **transformer over the braid word** — `L` is a sequence length, not a parameter count.
  Increasing `L` from 32 → 256 costs compute, not weights (with RoPE/ALiBi it also extrapolates).
* **1-D dilated CNN** over the word — translation-invariant, `L`-agnostic.
* **GNN over the diagram graph** — invariant to crossing count by construction.

With any of these, the curriculum (`K`, `L`, strand count, crossing bound) changes only the
*data distribution*. You then grow depth/width for *capacity*, not for *input size* — a much
easier problem. This is worth a lot; get the encoding right and 80% of the growth problem
disappears.

Caveat: the action space `M × L` *does* depend on `L`. Fix by making the policy head
**positional** — output a per-position logit vector `[L, M]` from the sequence representation
(a pointer-network style head), which is naturally `L`-agnostic. Do this from day one; retrofitting
it is painful.

## 1. Function-preserving growth (Net2Net family)

Expand a trained net so the larger net initially computes the *same function*:

* **Net2WiderNet** — split hidden units, divide outgoing weights among copies.
* **Net2DeeperNet** — insert identity layers.
* **Transformer layer stacking / depth up-scaling** — the modern version. Duplicating and
  stacking trained blocks to initialize a deeper model saves **>50% of pretraining compute**
  ([Stacking Your Transformers, arXiv:2405.15319](https://arxiv.org/abs/2405.15319)); see also
  bert2BERT, LiGO (learned growth operators), and
  [progressive depth up-scaling via optimal transport (arXiv:2508.08011)](https://arxiv.org/abs/2508.08011).

Practical notes: add a small perturbation to broken symmetries (exact duplicates have identical
gradients), and re-warm the learning rate after each growth event —
[When is Warmstarting Effective for Scaling Language Models? (arXiv:2605.13405)](https://arxiv.org/abs/2605.13405)
is the reference for getting this wrong.

## 2. Data-side inheritance (what AlphaZero practitioners actually do)

In practice the strongest self-play systems do **not** transplant weights. They keep the
**self-play replay buffer** and train the new, larger net on it from scratch (or warm-started),
then swap it in once it beats the incumbent in the arena. KataGo increases network size mid-run
this way, reusing the accumulated self-play window. It is simpler, more robust, and avoids the
symmetry-breaking pathologies of weight surgery.

For this project that means: **the problem bank and the outcome matrices are the durable asset**,
not the weights. Persist them from round 0 with full provenance. If you have to choose one thing
to engineer carefully in month 1, it is the replay/problem store, not the model.

## 3. Distillation between levels

Level `t` net (small, expensive MCTS) → level `t+1` net (large, cheap MCTS): train the larger net
on the *search-improved targets* `π ∝ N^(1/τ)` from the smaller net's tree, not on the smaller
net's raw policy. This inherits search quality, not just function values, and is strictly better
than plain distillation in AlphaZero-style systems.

## 4. Why this matters for the self-play loop specifically

[arXiv:2603.02218](https://arxiv.org/abs/2603.02218) argues that **capacity growth is a necessary
condition** for self-play to keep improving rather than plateau: if the loop generates more
learnable information each iteration but model capacity and inference budget are fixed, the extra
information cannot be absorbed and the system saturates. So growth is not an efficiency
optimization here — it is part of the mechanism. Schedule it against a measured quantity
(e.g. grow when validation loss on the anchor set stops falling while problem-bank difficulty
`b_j` keeps rising), not on a fixed timetable.

## Recommended schedule

```
level 0: L=32,  n≤5, 0.5M params, 32 sims     →  train to plateau on anchors
level 1: L=64,  n≤6, 2M params  (stack ×2),   64 sims,  warm LR restart
level 2: L=128, n≤8, 8M params  (stack ×2 + width ×1.5), 128 sims
level 3: L=256, n≤8, 30M params, 256 sims
```

Keep the *same* replay buffer and problem bank throughout; re-label old positions with the
current net's MCTS periodically (AlphaZero "reanalyse"). Report every level against the same
frozen anchor set so the curve means something.
