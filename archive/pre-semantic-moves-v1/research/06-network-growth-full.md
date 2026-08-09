# 06 — Network growth: the survey, and the measurement that changed it

This note was two: a design-phase survey of growth operators that assumed capacity
was the binding constraint and sized a schedule up to 30M parameters, and a later
branch plan written after the ladder measured that it is not. They are merged
here because they are one question, and reading the survey without the
measurement gets the premise wrong.

**The short version.** The operators below are still the right operators. The
premise was not, and the 30M-parameter schedule the survey ended with was sized
for a compute regime this project is not in. Establish a capacity-bound regime
first; everything after that is engineering.

Part I is what was measured. Part II is what it implies about where growth could
help at all. Part III is the operators. Part IV is how to run it so the result
means something, including the conditions under which the branch should stop.

---

# Part I — What the ladder measured

From `pgx-mcts-bench/artifacts/ladder-run/ladder.md`, ten stages, promotion at 80% solve rate:

| candidate | params | simulations | highest stage |
|---|---:|---:|---:|
| `search-heavy` | 48K | 128 | 8 |
| `wide-net` | **372K** | 64 | 8 |
| `u1-puct` | 48K | 64 | 7 |
| `search-light` | 48K | **16** | **0** |

**7.7× the parameters bought nothing that 2× the simulations did not.** Starving search was fatal;
starving capacity was not measurable. Any growth experiment that does not first show a
capacity-bound regime is measuring noise.

The corollary matters for [05](05-compute-budget.md) too: the networks are 48K–372K parameters and
MCTS is **batch-1 latency bound** — a measured 605 µs per simulation of which 463 µs is the forward
pass, ~77%. At that size the forward pass is dispatch overhead, not arithmetic. Growing the network
10× would barely change the wall clock, which is the one genuinely encouraging thing here: growth
is nearly free to *try*. It is also why the schedule this note used to end with was the wrong
target — it was sized for a compute regime this project is not in.

---

# Part II — Two different problems, decided by receptive field

The receptive field of `Representation` is `1 + 2 × (1 stem conv + 2 blocks × 2 convs) = 11`
letters, which is what the comment in `BraidAlphaZeroNet` already states. That single number splits
the branch in two:

| net | receptive field | word / window | sees | depth growth |
|---|---:|---:|---|---|
| parallel | 11 | L = 48 | 23% | **meaningful** — needs 12 blocks for RF ≥ 48 |
| serial, w = 7 | 11 | 7 | 100% | **pointless** — already saturated |
| serial, w = 11 | 11 | 11 | 100% | exactly at the boundary |

So for the serial formulation the only levers are width and the window itself; for the parallel one,
depth is the interesting axis. `s-w11-128` sits exactly on the line and is the arm to watch.

---

# Part III — The operators

Ordered as the survey ordered them, which is still right: avoid needing to grow,
then grow the function, then grow the data, then distil.

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

## The schedule that is no longer recommended

The survey ended with a four-level schedule rising to 30M parameters and 256
simulations. It is deleted rather than corrected: it was sized against an assumed
capacity bound that Part I measured away, and a schedule that specific reads as
advice long after its premise has gone. What survives is the part that was never
about size — **keep the same replay buffer and problem bank throughout**,
periodically re-label old positions with the current net's MCTS (AlphaZero
"reanalyse"), and report every level against the same frozen anchor set so the
curve means something.

---

# Part IV — How to run it

Ordered by cost. A–C preserve the function by construction, D by optimisation, E not at all.

**A. ReZero block append (depth).** Append a residual block as `x + α·body(x)` with `α` a learnable
scalar initialised to zero. Exactly function-preserving at insertion.

**B. Net2WiderNet (width).** Duplicate a random subset of channels, halve their outgoing weights,
copy the BatchNorm running statistics. Add small noise to the duplicates or the pair moves
identically for ever — Part III notes this and it remains correct.

**C. Frozen column plus lateral adapter.** Freeze the weights that cleared stage *k*; add a narrow
parallel column with lateral connections into the heads. Cannot regress, by construction.
Parameters grow linearly in stages, which is the price.

**D. Grow arbitrarily, then distil.** Change anything — including the serial window width, which
changes the observation shape and so is out of reach for A–C — then distil the incumbent's policy
and value over the replay buffer until KL < ε before resuming RL. Distil the *search-improved*
targets `π ∝ N^(1/τ)`, not the raw policy, per Part III below.

**E. Control: cold re-initialise, keep the replay buffer.** This is the arm that can make the whole
branch unnecessary, and it is what KataGo actually does. If growing-and-forgetting matches
growing-and-preserving, none of A–D earns its complexity.

## The trap worth writing down

The obvious way to make an appended residual block an identity is to zero-initialise the final
BatchNorm's `γ`. **This does not work, and it fails silently.** With `γ = 0` the gradient reaching
the block's convolution weights is proportional to `γ`, so it is zero: the block is an identity at
insertion and stays one for ever. It will not error, will not diverge, and will produce a clean
null result that looks like "growth does not help".

With a scalar `α` the gradient is `⟨∂L/∂out, body(x)⟩ ≠ 0`, so `α` leaves zero on the first step.
The same trap applies to the zero-outgoing-weight variant of widening, which is why B duplicates
and halves rather than padding with zeros.

## Protocol

**Function preservation is a unit test, not an experiment.** Immediately after growth, over a batch
of replay states:

```
max |Δ logits| < 1e-5    and    max |Δ value| < 1e-6
```

A–C must pass exactly; D passes to tolerance. This costs seconds and catches the entire class of
weight-surgery bugs before any training run is spent on them.

**The experiment** then measures whether growth helps, using the ladder machinery unchanged. Growth
points are the `checkpoints/<candidate>/stage<NN>-{before,after}.pt` snapshots.

1. Solve rate immediately after growth — unchanged for A–C. This is the "did not break what it
   learned" check, and it is the one the unit test cannot make for the *trained* function.
2. Iterations to clear the next stage, against no growth.
3. Highest stage reached.
4. **Retrospective**: re-evaluate stages 0..k−1 with the grown weights. The un-grown baseline
   improves earlier stages while training on later ones — `s-window-128` went from 4.18 to 1.17
   crossing changes on stage 3 (u = 1) without ever training there again — so a growth operator
   must be shown not to cost that transfer.

**8 seeds per arm.** Three gave a false positive on the proposer that survived two rounds of
reporting; this is the same shape of claim.

## The falsifiable version

A vaguely-worded growth experiment will produce a vaguely-worded result. Here is a sharp one.

The observation hand-feeds the destabilisation predicate — a `top_generator` plane and a `top_count`
scalar — precisely *because* a receptive field of 11 cannot evaluate a global predicate over a
48-letter word. The comment in `env.py` says so explicitly.

**Prediction:** if depth growth to RF ≥ L buys real capability rather than parameters, a grown
network should tolerate those two channels being removed. If it cannot, growth did not buy the
thing the receptive-field argument says it should, and approach A is dead for the parallel net
regardless of how the stage counts look.

This is worth running early. It is cheap, it is a clean yes/no, and it tests the mechanism rather
than the outcome.

It also does double duty as the **zero-human-knowledge audit**. Those two channels are a
human-computed answer to a question the network was supposed to work out, which is the same class
of object as a polynomial feature — see [12 §4](12-serial-formulation.md). So the ablation is both
"did growth buy receptive field" and "can the system be run at the standard the experiment claims".

## Cost

Five arms × 8 seeds, resuming from a stage-8 snapshot and climbing to 12, is roughly 60 core-hours
by the screen's measured rate — about 10 hours on 6 cores of the laptop, or an hour and a few
dollars on a rented 64-vCPU box. This is a cheap branch. The expensive part is deciding what it
means, not running it.

## What would kill this branch

Any one of these should stop it rather than prompt a workaround:

* **E matches A–D.** Preservation is not worth the code; keep the replay buffer and re-initialise.
* **No capacity-bound regime exists** even at stage 12. Then growth is a solution to a problem this
  project does not have, and the compute belongs in search or in the two-tape equivalence
  formulation instead.
* **The channel-ablation prediction fails.** Growth added parameters, not receptive field.

Recording these in advance is the point. The failure mode for a branch like this is that it becomes
unfalsifiable — every null result reads as "needs more tuning".
