# 11 — Network growth: a research branch, and first whether it is needed

[06](06-network-growth.md) surveyed the growth literature during the design phase and recommended
a schedule ending at 30M parameters. Since then the ladder has been run, and it measured something
06 could not know: **capacity is not the binding constraint at this scale.** So this branch has two
halves, in this order — establish that growth buys anything, then find a growth operator that does
not destroy what was learned. Doing them the other way round is how you end up with a working
Net2Net implementation and no result.

## What the ladder measured

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
10× would barely change the wall clock, which is the one genuinely encouraging thing here: growth is
nearly free to *try*. It is also why the 30M-parameter schedule in 06 is the wrong target — it was
sized for a compute regime this project is not in.

## Two different problems, decided by receptive field

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

## Approaches

Ordered by cost. A–C preserve the function by construction, D by optimisation, E not at all.

**A. ReZero block append (depth).** Append a residual block as `x + α·body(x)` with `α` a learnable
scalar initialised to zero. Exactly function-preserving at insertion.

**B. Net2WiderNet (width).** Duplicate a random subset of channels, halve their outgoing weights,
copy the BatchNorm running statistics. Add small noise to the duplicates or the pair moves
identically for ever — 06 notes this and it remains correct.

**C. Frozen column plus lateral adapter.** Freeze the weights that cleared stage *k*; add a narrow
parallel column with lateral connections into the heads. Cannot regress, by construction.
Parameters grow linearly in stages, which is the price.

**D. Grow arbitrarily, then distil.** Change anything — including the serial window width, which
changes the observation shape and so is out of reach for A–C — then distil the incumbent's policy
and value over the replay buffer until KL < ε before resuming RL. Distil the *search-improved*
targets `π ∝ N^(1/τ)`, not the raw policy, per [06 §3](06-network-growth.md).

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
