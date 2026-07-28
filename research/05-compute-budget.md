# 05 — Compute budget

## First, the reality check on "all knots with 5 crossings"

There are **two** prime knots with 5 crossings:

| knot | unknotting number |
|---|---|
| 5₁ (Solomon's seal, (2,5) torus knot) | 2 |
| 5₂ | 1 |

Counts of prime knots by crossing number: 3→1, 4→1, **5→2**, 6→3, 7→7, 8→21, 9→49, 10→165,
11→552, 12→2176, 13→9988.

So "train a NN to unknot all 5-crossing knots" needs **zero GPUs** — it needs a two-row lookup
table, and the whole ≤10-crossing table (250 knots including non-prime) fits in a text file with
unknotting numbers already known.

**The unit of difficulty is not the knot type, it is the diagram.** A single knot type has
infinitely many diagrams, and the RL problem is: given an arbitrary diagram of ≤ C crossings,
find a short simplifying / unknotting sequence. That is the thing that costs GPU time, and it is
also the thing the Scrambler in [03](03-knot-env-pgx.md) generates.

Restate the question as: *"what does it cost to train an agent that reliably simplifies diagrams
of up to C crossings, produced adversarially?"*

## What the published work actually spent

Before budgeting, look at what the two live papers in this area used. The answer
reframes the question.

**[arXiv:2603.07955](https://arxiv.org/abs/2603.07955) (Dranowski, Kabkov, Tubbenhauer)
reports no hardware at all.** Nineteen pages, and the strings "GPU", "CPU", "hours",
"hardware" do not appear. What it does report is the full configuration:

* **PPO**, `MlpPolicy` from **stable-baselines3** — the default MLP, a few thousand
  parameters. No MCTS anywhere in the paper.
* learning rate `3·10⁻⁴` with linear decay, `n_steps = 2048`, batch 256,
  10 epochs per update, `γ = 0.995`, GAE `λ = 0.97`, clip 0.2, entropy coef 0.01,
  value coef 0.5, max grad norm 0.5.
* state is a **planar diagram (PD) code**, manipulated through `spherogram`
  (SnapPy's link module), with R1/R2/R3 plus an "increase-shuffle" operation.

An SB3 MLP policy on a small observation vector runs *faster on CPU than on GPU* —
the bottleneck is environment stepping in Python, not matrix multiplication. So the
honest answers to your two questions:

1. **How many GPUs?** Almost certainly **zero**. The paper does not say, and nothing
   in the configuration needs one.
2. **What would it cost to reproduce?** Compute is not the constraint. This is
   CPU-hours, dominated by `spherogram` calls — days on a laptop, or on the order of
   **tens of dollars** of cloud CPU. They also publish code, trained models and
   datasets.

And with that budget they recovered the surprising bound `u(4₁ # 9₁₀) ≤ 3`.

The contrast with [arXiv:2409.09032](https://arxiv.org/abs/2409.09032) (DeepMind +
Oxford, 200-crossing diagrams, 57k knots, 2.6M hard diagrams) is the real lesson:
**this field has a laptop-scale result and an industrial-scale result, and the
laptop-scale one still produced new mathematics.** We are not compute-limited. We
are idea-limited. Budget accordingly — spend on the ideas in
[10-invariants-and-representations.md](10-invariants-and-representations.md), not on
GPUs.

One more useful detail from that paper: they name their two central difficulties as
**sparse progress** (R3 moves do not change the crossing number, so most correct
steps look like no progress) and **local minima / dead ends** (greedy crossing-reducing
strategies get stuck). Those are precisely the two things MCTS with a value network
addresses and PPO does not — and since no one in this literature has used tree search
at all, that gap is unoccupied rather than merely under-explored.

## Tiers (order-of-magnitude estimates — verify empirically at tier 0)

| Tier | Scope | Net | Hardware | Wall clock | Confidence |
|---|---|---|---|---|---|
| **0** — smoke | braids ≤5 strands, `L`≤32, `K`≤10 scramble moves | 1-D resnet, 0.1–1 M params | laptop CPU or 1 consumer GPU | hours | high |
| **1** — the real first result | braids ≤8 strands, `L`≤64, `K`≤50; recover unknotting numbers for the ≤10-crossing table | 1–5 M params, 64–256 MCTS sims | 1× RTX 4090 / A100 | 1–3 days per seed | medium |
| **2** — competitive with literature | diagrams to ~50 crossings, ≤12-crossing prime knot table, attempt open upper bounds | 10–50 M params | 4–8× A100/H100 | 1–3 weeks | low-medium |
| **3** — DeepMind-scale | 200-crossing diagrams, 57k knots, 2.6M hard-diagram corpus | ~100 M params + massive self-play | tens of GPU-months / TPU pod | months | low |
| **L** — 5 LLM agents | 5 LoRA adapters on a 7B base, RLVR rounds | 7B frozen + 5× rank-32 LoRA | 8×80 GB node | weeks | low |
| **L-lite** — 5 LLM agents, cheap | 5 LoRA on 1.5B, short contexts, knot problems only | 1.5B + LoRA | 2× 24 GB (4090/3090) | days | medium |

Reasoning behind tier 1: it is the same order as your 6×6 Go AlphaZero runs in `pgx-mcts-bench`
(comparable action-space size ~450 vs 37, comparable episode length, similar net). If a 6×6 Go
run costs you X, tier 1 costs roughly 3–10X because episodes are longer and rewards sparser.
**Measure X first, then extrapolate — don't trust the table.**

## Cost drivers, in order

1. **MCTS simulations per move.** Linear. This is the dial you actually control. Sparse-reward
   long-horizon domains want more sims, not a bigger net.
2. **Episode length `M`.** The Simplifier may need `M ≫ K` moves. Cap it and log the truncation rate.
3. **`L` (max word length).** Determines observation size and whether hard instances are solvable
   at all. Grows the net input linearly.
4. **Net size.** Least important early. In AlphaZero-style systems, search quality dominates
   until the net is badly underfit.
5. **Lower-bound invariants** (signature, Rasmussen `s`, Khovanov homology). These are *CPU*
   costs, not GPU, and Khovanov homology is expensive (exponential in crossings). Cache
   aggressively; compute once per knot type, never per diagram.

## What I would actually budget

For a first credible result (tier 1, 5 seeds, plus the U1–U5 sweep from `pgx-mcts-bench`
re-run on this env): **one A100-class GPU for ~4–6 weeks**, or a 4090 for ~2 months, or a few
hundred dollars of spot cloud. That is the honest number for "publishable single-author
experimental-math result", and it does not require the LLM half at all.

The LLM half (tier L) is a separate, larger commitment and should not be started until tier 1
has produced a working environment, a verifier, and a difficulty model.
