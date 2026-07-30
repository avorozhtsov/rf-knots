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
| [11-network-growth-branch.md](11-network-growth-branch.md) | Growth as an executable branch: whether capacity is the constraint at all, the receptive-field split, and what would kill it |
| [12-serial-formulation.md](12-serial-formulation.md) | The serial/Turing formulation: why it scored 0 then 9, the missing head register and how big it must be, what "zero human knowledge" actually excludes, and knot equivalence as a two-tape machine |

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
   **But measure first:** 7.7× the parameters bought nothing that 2× the simulations
   did, so establish a capacity-bound regime before growing anything —
   [11](11-network-growth-branch.md).

6. **GPU rental.** Not yet, and the reason is measured: MCTS is batch-1 latency
   bound (605 µs per simulation, 77% of it the forward pass on a 48K-parameter
   network), so a GPU's kernel-launch overhead makes it *worse*. Batch the leaf
   evaluations across parallel games first — that alone measured 7.8× on the same
   laptop. Then rented CPU is the cheap win: a sweep that takes this machine two
   weeks is order $85 on 64 vCPU.

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
* **What "zero human knowledge" excludes.** The experiment is unknotting with zero
  human knowledge, which makes computed invariants an *oracle* arm rather than a
  fair one — useful for bounding the upside before the learned version is built,
  never the headline. This reclassifies proposals 4 and 5 in
  [10](10-invariants-and-representations.md). It also exposes that the observation
  already carries three human-knowledge channels, so the standard was being applied
  unevenly. Audit in [12 §4](12-serial-formulation.md).
* **The serial formulation works, and the reason it did not was a readout bug.**
  It scored 0 against the parallel candidates' 7–8, then cleared the whole ladder
  once the policy head became positional. [12](12-serial-formulation.md).
* **Agent-written head registers are a clean negative.** Giving the head K binary
  registers with a TOGGLE action each cost performance monotonically in K: at K=8
  it collapsed onto the exact rung where the *pre-fix* serial candidates died. A
  TOGGLE never changes the word, so it dilutes the action space with branches MCTS
  cannot make progress on. The lesson is general — a register that nothing reads
  is noise, and the read side has to come first. What reads it is an automatic
  whole-tape accumulator. [12 §3](12-serial-formulation.md).
* **Jones needs a bigger carrier than Burau.** `TL₅` decomposes into irreducibles
  of dimension 1, 4 and 5, and the 4-dimensional block *is* the reduced Burau
  representation — so a 4×4 carrier reaches Alexander and one of Jones's three
  blocks, not Jones. [12 §3](12-serial-formulation.md).
* **The A/B objective is not inert after all — it is inert in one formulation.**
  Parallel candidates emit the same policy at both ends of `log(A/B)` to two
  decimal places, as the Pareto argument predicted. Serial candidates respond by
  5–6× in moves, because head travel is charged. [12 §2](12-serial-formulation.md).
* **Whether the network needs growing at all.** [06](06-network-growth.md) assumed
  capacity was a constraint and sized a schedule up to 30M parameters. The ladder
  measured otherwise: 7.7× the parameters (`wide-net`, 372K) reached the same stage
  as 2× the simulations (`search-heavy`, 48K), while 16 simulations reached stage 0.
  Search dominates; capacity was not measurable. The growth operators in 06 are
  still the right operators, but the branch now has to establish a capacity-bound
  regime before it can measure anything —
  see [11](11-network-growth-branch.md).

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
