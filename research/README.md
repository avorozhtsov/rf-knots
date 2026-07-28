# rf-knots — research notes

A design study for a **population self-play system where N agents both *propose* and *solve*
problems from a fixed, machine-verifiable family**, with an AlphaZero-style inner search loop,
aiming eventually at L-function / LMFDB-style mathematics.

These notes are written to be argued with, not obeyed. Where I state numbers they are
order-of-magnitude estimates and marked as such.

## Contents

| File | Question it answers |
|---|---|
| [01-game-design.md](01-game-design.md) | Why 5 agents rather than 2, and what the scoring rule must be |
| [02-alphazero-backprop.md](02-alphazero-backprop.md) | What "backpropagation like AlphaZero" actually means in a propose/solve league |
| [03-knot-env-pgx.md](03-knot-env-pgx.md) | Concrete pgx environment design for unknotting; feasibility verdict |
| [04-related-work.md](04-related-work.md) | Annotated bibliography (self-play proposers, RL for knots, murmurations) |
| [05-compute-budget.md](05-compute-budget.md) | GPU estimates per tier, and why "all 5-crossing knots" is the wrong unit |
| [06-network-growth.md](06-network-growth.md) | Growing the network across curriculum levels without retraining from scratch |
| [07-domain-choice.md](07-domain-choice.md) | Knots vs. rational sums vs. LMFDB; the "how do we set direction" problem |
| [08-roadmap.md](08-roadmap.md) | Concrete milestones, starting from the existing pgx-mcts-bench code |
| [09-vs-learning-to-unknot.md](09-vs-learning-to-unknot.md) | Would this supersede arXiv:2010.16263? What the real contribution is |
| [10-invariants-and-representations.md](10-invariants-and-representations.md) | Unknotting number as a graph distance; where invariants help and where they are useless; mosaics vs. grid diagrams |

Implementation reference: [../docs/representation.md](../docs/representation.md) — how a knot is
encoded and what the agent may do to it, with the Reidemeister/Markov correspondence.

## Short answers

1. **5 vs 2.** 5 is right, but not as a 5-player game. Make it a *league* with role asymmetry.
   The decisive technical reason: with one opponent you can only observe solve-rate ∈ {0,1} per
   problem, so you cannot estimate item difficulty at all. The proposer reward you want
   ("hard but not impossible") is a function of the *distribution* of outcomes across solvers,
   and needs ≥4 independent solvers to be non-degenerate. Population self-play also has direct
   empirical support: a single agent self-calibrates toward easy problems, populations don't
   ([PopuLoRA, 2026](https://arxiv.org/abs/2605.16727)).

2. **Unknotting in pgx.** Yes — with the *braid word* encoding, not planar diagrams.
   Fixed-shape `int8[L]` state, action = (move_type × position), all legality masks are pure
   vectorized JAX. See [03](03-knot-env-pgx.md). The version I recommend building first is a genuine
   2-player zero-sum game — **Scrambler vs. Simplifier** — which is directly AlphaZero-able and
   reuses `../pgx-mcts-bench` almost unchanged.

3. **Prior art.** Yes, a lot, mostly 2025–2026: Absolute Zero, R-Zero, Search Self-play,
   Propose-Solve-Verify, PopuLoRA. And separately, RL for unknotting is an active,
   *published-results* area (DeepMind/Oxford determined 43 previously-unknown unknotting numbers).
   See [04](04-related-work.md). Nobody has combined the two. That gap is the project.

4. **GPU for 5-crossing knots.** There are exactly **two** prime knots with 5 crossings
   (5₁ with u=2, 5₂ with u=1). Zero GPUs; a lookup table. The question only becomes
   compute-bound when the unit is *diagrams*, not knot types. Tiers in [05](05-compute-budget.md).

5. **Growing the net.** Yes: function-preserving growth (Net2Net), transformer layer stacking
   (~50% pretraining compute saved), and the AlphaZero-practice version — warm-start the bigger
   net on the accumulated self-play buffer. Better still: pick an architecture that is *invariant*
   to problem size so the curriculum changes the data, not the net. See [06](06-network-growth.md).

## What has changed since this was written

Two positions in these notes have been revised by evidence or by argument, and
the originals are marked rather than deleted:

* **The proposer's objective.** "Reward problems that are hard for others" is
  wrong as a model of how mathematics selects problems, and is superseded by
  *reward the proposer for how much the solver improves* — see the revision
  section in [01](01-game-design.md). The frontier-targeting `4p(1-p)` term
  survives; the adversarial framing does not.
* **Which paper is the bar.** [09](09-vs-learning-to-unknot.md) argues the
  project should not be pitched as "a better unknotter", and that the real
  contribution is the question of where hard instances come from and whether
  training against them transfers.

## The one design idea I'd defend hardest

Your own objection is the crux: *generating hard instances by random scrambling is trivially
easy, so "propose hard problems" is a degenerate objective.* The fix is to bound the proposer's
**generation budget** and reward **hardness per unit budget**, gated on the proposer being able
to solve its own instance.

That turns the proposer's job into a real search problem with an independently valuable output:
*small diagrams that are hard to simplify*. In knot theory those are literally a studied object
("hard unknot diagrams"). So even if the LLM/self-play half of the project underdelivers, the
artifact — a corpus of adversarially-generated hard instances plus improved unknotting-number
upper bounds — is a publishable result on its own. Design for that.
