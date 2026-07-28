# 09 — Would this supersede *Learning to Unknot*?

**Short answer: on their task, no — and "supersede 2010.16263" is the wrong bar
anyway, because that paper has already been superseded on the RL-unknotting axis
by two more recent ones.** What is new here is the adversarial *generator* and
the league, not the simplifier. If the project is pitched as "a better unknotter"
it will be a weak paper. If it is pitched as "where do hard instances come from",
it is a real one.

## What [arXiv:2010.16263](https://arxiv.org/abs/2010.16263) actually did

Gukov, Halverson, Ruehle, Sułkowski, *Learning to Unknot* (2020), published in
*Machine Learning: Science and Technology*. Three separate contributions:

1. **An instance generator.** An algorithm to randomly generate `N`-crossing
   braids and their closures, with an explicit discussion of the *induced prior*
   on the distribution of knots.
2. **Supervised classification of UNKNOT.** Reformer and shared-QK Transformer
   architectures on braid words as language; all architectures did well.
   Two findings worth keeping: accuracy *increases* with braid word length, and
   the networks' confidence correlates with the degree of the Jones polynomial.
3. **RL simplification.** TRPO to find sequences of Markov moves and braid
   relations that simplify a knot, and prove triviality by exhibiting the
   sequence. TRPO beat the other RL algorithms and random walkers. Braid
   relations turned out more useful than one of the Markov moves.

Contribution 3 is the same task as this environment's Simplifier, on the same
representation, with the same move set.

## Where the overlap is total

Braid-word state, Markov moves plus braid relations, RL agent that simplifies and
certifies by exhibiting the move sequence. If the deliverable were "an RL agent
that unties random braids", this project would be a 2020 paper reimplemented in
JAX. It should not be pitched that way.

## Where this differs, in descending order of how defensible the claim is

### 1. The instances are adversarial, not sampled from a fixed prior — *strong*

Their instances come from a random generator with an analysed prior. Ours come
from a **trained Scrambler that is scored on making instances hard**.

This is not a stylistic difference, and their own result is the evidence.
"Accuracy increases with the length of the braid word" is exactly what you expect
when long random words are *easier*: random generation piles on cancelling pairs
that a classifier can learn to see through. My own calibration on this
environment says the same thing quantitatively — exact BFS on uniformly random
scrambles gives

| `K` (scramble moves) | mean optimal solution depth |
|---|---|
| 3 | 2.56 |
| 4 | 3.16 |
| 5 | 3.96 |

**~0.7 moves of difficulty per move spent, and never once harder than `K`.**
A random generator is a weak adversary, and every result measured against it
inherits that weakness. Replacing it with a trained one is a real methodological
change, and it produces an artifact — a corpus of hard instances with certified
generation budgets — that is useful independently of any RL claim.

The recent literature agrees this is where the value is: the DeepMind/Oxford work
([arXiv:2409.09032](https://arxiv.org/abs/2409.09032)) treats a corpus of
**2.6M hard unknot diagrams** as a headline contribution in its own right.

### 2. Search with an exact model, rather than model-free policy gradient — *plausible, untested*

They used TRPO: model-free, no lookahead. This environment supplies an exact,
cheap, jit-able transition function, so MCTS with value backup is available and
`../pgx-mcts-bench` already implements it. On combinatorial search problems with
sparse terminal rewards, search-plus-learning usually beats model-free policy
gradient by a wide margin — that is the entire AlphaZero result.

Stated honestly: this is a **hypothesis with good priors, not a finding**. It is
also cheap to test, and the test is a legitimate contribution on its own
(AlphaZero vs TRPO on a shared, exactly-specified unknotting environment, at
fixed compute). Do not claim it before running it.

### 3. The propose/solve league — *new, but unproven and further away*

Nobody in the knot-theory ML literature has a population of agents scored on
proposing informative problems. That is genuinely unoccupied territory
(see [04-related-work.md](04-related-work.md)). It is also M4, months out, and
should not be part of the pitch until M2 and M3 have landed.

## Where this does **not** compete

* Their classification results and the Jones-polynomial confidence correlation —
  a different task, untouched here.
* Their analysis of the prior induced by random braid generation — in fact this
  project *depends* on that kind of analysis being right, and should cite it as
  the baseline it is measuring against.

## The bar that actually matters

`arXiv:2010.16263` is not the state of the art on RL unknotting. Two papers are
ahead of it:

* **[arXiv:2409.09032](https://arxiv.org/abs/2409.09032)** (DeepMind + Oxford,
  *Experimental Mathematics* 2025) — diagrams up to **200 crossings**, unknotting
  numbers for **57k knots**, and **43 knots of ≤12 crossings whose unknotting
  number was previously unknown**. That is a mathematical result, not a
  benchmark score.
* **[arXiv:2603.07955](https://arxiv.org/abs/2603.07955)** (Dranowski, Kabkov,
  Tubbenhauer) — policy **and value** over Reidemeister moves, recovers the
  surprising bound `u(4₁ # 9₁₀) ≤ 3`, and describes a self-improving
  workbook-driven loop that systematically improves upper bounds across the prime
  knot list. Structurally this is the nearest neighbour to what is being built
  here, minus the population and minus the proposer game.

So the honest framing of the project's contribution is:

> Not "a better unknotter". Rather: *where do hard instances come from, and does
> training against an adversarial generator produce a simplifier that transfers
> to instances no random generator would ever emit?*

That question is answerable at M3 with a concrete experiment: take the Simplifier
trained against the trained Scrambler, and the Simplifier trained against the
random Scrambler at matched compute, and evaluate both on (a) the standard prime
knot tables and (b) the published hard-unknot-diagram corpus. If adversarial
training transfers and random training does not, that is a result worth writing
up, and it is orthogonal to everything in the three papers above.

If it does *not* transfer, that is also worth knowing, and is a cheaper thing to
discover at M3 than at M4.

## Consequence for the roadmap

Add to [08-roadmap.md](08-roadmap.md) M3 as the decisive experiment:

- [ ] train Simplifier-vs-random-Scrambler and Simplifier-vs-trained-Scrambler at
      **matched compute**
- [ ] evaluate both on held-out instances neither generator produced: prime knot
      tables, and the published hard-unknot corpora
- [ ] report the transfer gap — this, not the win rate, is the paper
