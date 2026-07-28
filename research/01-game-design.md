# 01 — The N-agent propose/solve league

## Why not "a 5-player game"

For 2-player zero-sum games there is a minimax value, and self-play is a contraction toward it —
that is *why* AlphaZero works. For N ≥ 3 general-sum (or even constant-sum) games none of that
survives: Nash equilibria are non-unique, self-play can cycle indefinitely, and "exploitability"
loses its meaning. Worse, N-player propose/solve has a specific bad equilibrium:

> **Niche collusion.** Each agent specializes in a disjoint sub-family, proposes only problems
> inside its niche, scores maximum "nobody else solved it", and nobody learns anything.

So: do not seek an equilibrium. Treat the 5 agents as a **population producing a curriculum**,
and let the scoring rule be a *measurement instrument* rather than a payoff to be optimized to
equilibrium.

## Why 5 beats 2 — the quantitative argument

The reward you want for a proposer is "the problem was informative": maximally discriminating
between solvers. Standard psychometrics (2-parameter IRT / adaptive testing) says the
information an item carries about ability θ is maximized when P(solve) ≈ 0.5.

With **one** other solver, the observed solve rate per item is in {0, 1}. You cannot distinguish
"hard" from "impossible", which is exactly the failure that kills proposer/solver self-play
(R-Zero's pseudo-label accuracy falls 79% → 63% by iteration 3 as the challenger outruns the
solver). With **four** other solvers × k samples each you get a usable estimate:

| solvers | samples each | attainable p̂ granularity | can separate hard-vs-impossible? |
|---|---|---|---|
| 1 | 1 | {0, 1} | no |
| 1 | 8 | 1/8 | only sampling noise, one strategy |
| 4 | 4 | 1/16 | yes — variance *across strategies* |

Independent agents give variance across *strategies*, not just across samples from one policy.
That is the signal a curriculum needs. 5 total (1 proposer + 4 solvers per item) is close to
the minimum that works; there is nothing magic about 5, and 6–8 is fine if compute allows.

## Round structure

```
round t:
  budget K_t                      # judge-controlled generation budget (curriculum knob)
  for each agent i in 1..5:
    propose 2 problems, each with a *certificate* (a solution the proposer itself found)
  problem pool P_t = 10 problems, deduplicated by invariant fingerprint
  for each agent i, for each problem j in P_t (including its own):
    k independent solution attempts, verified by the programmatic verifier
  outcome tensor X[i][j][s] in {0,1}
  compute rewards, update policies, update ratings, maybe raise K_{t+1}
```

## Scoring rule

Let `p_j` = solve rate of problem *j* over the **other** four agents (all k samples pooled).

**Solver reward** — credit for solving what others cannot:

```
r_solve(i) = Σ_j  solved(i,j) · (1 - p_j^(-i))
```

**Proposer reward** — see [the revision below](#revision-the-proposer-should-be-rewarded-for-teaching-not-for-winning);
the form given here is the *frontier-targeting* part, which survives, but it is
not the headline objective:

```
r_prop(a, j) = own_solved(a,j) · 4·p_j·(1 - p_j) · novelty(j) · (K_t / cost(j))
                └── gate ──┘   └── Fisher info, peaks at 0.5 ──┘   └── budget efficiency ──┘
```

* `own_solved` gate: you may not profit from a problem you cannot solve yourself. This is the
  single most important term — it is what every working system in the literature (AZR, R-Zero,
  SSP, PSV) relies on, and it is what stops "scramble randomly and win".
* `4p(1-p)`: item information. A problem nobody solves scores **zero**, same as a problem
  everybody solves.
* `novelty`: 1 − max similarity to the problem bank, measured with cheap invariants
  (for knots: (crossing number, Jones polynomial, signature, Alexander polynomial) hash).
  Without this the population converges on one problem shape.
* `cost(j)`: number of generation moves actually used. Reward hardness *per move spent*.

Then **center per round**: `A_i = r_i − mean_j(r_j)`. This makes the round constant-sum,
gives a natural group-relative advantage (exactly GRPO's baseline), and keeps competitive
pressure without pretending there is a minimax value.

## The generation-budget knob (the answer to your objection)

> "it is much easier to use brute force to generate complicated problems than to solve them"

True, and unfixable in general. What *is* fixable: charge for generation.

* proposer may apply at most `K_t` inverse/complicating moves;
* reward is normalized by moves actually used;
* `K_t` is raised by the judge only when the league's median `p_j` exceeds ~0.7.

Now the proposer must find *which* K moves produce a maximally hard instance — a search problem
of the same order of difficulty as solving. In the knot case this is a known open-ended
experimental question (hard unknot diagrams: diagrams of the unknot on which every simplification
must first *increase* the crossing number). The proposer's output is therefore a scientific
artifact, not just training data.

## Failure modes and countermeasures

| Failure | Symptom | Countermeasure |
|---|---|---|
| Trivial-problem collapse | proposers emit near-identity instances | own-solve gate is *necessary but not sufficient* — add the `4p(1−p)` peak and the novelty term |
| Impossible-problem collapse | p_j → 0, reward signal vanishes | `4p(1−p)` → 0 removes the incentive; cap K_t |
| Niche collusion | agents' problem distributions become disjoint | force every agent to attempt every problem; novelty measured in *problem space*, not per agent; periodically re-seed the weakest agent from the strongest (PBT exploit/explore). **Structurally removed** by the learning-progress objective below: a niche nobody can enter produces no improvement and therefore no reward |
| Rating inflation | everyone's θ rises, absolute skill flat | keep a **frozen anchor set** of held-out problems with known answers; report θ against anchors only |
| Verifier gaming | solver exploits the checker | verifier must be independent of both roles and, ideally, *certificate-checking* rather than *answer-matching* |
| Pseudo-label rot | majority-vote "ground truth" degrades with difficulty (R-Zero) | never use majority vote — only accept problems where a programmatic verifier decides, or where the answer is known by construction (planted solution) |

The last row is why a **planted-solution** family is so valuable: problems generated by inverting
moves from a known terminal state carry ground truth for free, at every difficulty.

## Revision: the proposer should be rewarded for teaching, not for winning

*Added after the objection that adversarial generation is a poor model of how
mathematics actually selects problems. The objection is correct and this section
supersedes "hardness is not the objective; discrimination is" as the headline.*

### The objection

Researchers do not construct problems their colleagues cannot solve. They look
for problems that are **natural next steps** — that follow from a recent result,
or that fall out of combining two existing ideas. Adversarial generation captures
"not solvable by current methods" and captures nothing else. In particular it
misses the two properties that actually make a problem worth stating:

* **naturality** — it arises from existing structure rather than being constructed;
* **generativity** — solving it unlocks other things.

An adversary optimises against the solver. Science optimises *for* the field.
Those point in different directions, and the difference is not cosmetic: an
adversary that finds an obstructive niche has succeeded by its own lights while
teaching nothing.

### What survives, and why

Adversarial generation was never justified as a model of science. It was
justified as an answer to an *engineering* question — with no dataset, where do
training instances come from? — and that question is real. The measurement in
[03](03-knot-env-pgx.md) is the evidence: a uniform-random generator buys about
**0.7 moves of difficulty per move spent** and never produces anything harder
than `K`. Something has to do better than random, so the proposer stays.

The `4p(1-p)` term also survives, but for a different reason than I first gave.
It is not "make it hard" — it peaks at a 50% solve rate and goes to **zero** for
problems nobody can solve. That is frontier-targeting, and frontier-targeting
*is* how science works: the interesting problems sit just beyond current methods,
not arbitrarily far beyond them. Keep the term; drop the framing.

### The replacement objective

> Reward the proposer for how much the solver **improves** after training on its
> problems — not for how often the solver **fails** on them.

Same machinery, opposite sign. Concretely, with a frozen natural benchmark `B`
(for knots: the prime-knot tables and the published hard-diagram corpora),

```
r_prop(a) = theta_B(solver after training on a's batch) - theta_B(solver before)
```

This is learning-progress curriculum rather than adversarial curriculum. It is
measurable, it is not gameable by obstruction — a problem nobody can solve
teaches nothing and scores nothing — and it removes the niche-collusion failure
mode structurally rather than by penalty, because a niche that no one can enter
produces no improvement.

Cost: it is far more expensive per proposal, since scoring requires actually
training on the batch. Mitigation is the usual one — score batches rather than
individual problems, and amortise with the difficulty head `d_phi` from
[02](02-alphazero-backprop.md), which learns to predict the expensive signal.

### Three mechanisms for "natural next step"

In increasing order of ambition, decreasing order of how implementable they are
in this environment:

1. **Naturality = drawn from the corpus, not constructed.** The natural instances
   here are the knot tables — the objects mathematicians already care about. So
   the generator's job becomes *"find hard diagrams of table knots"* rather than
   *"construct arbitrary hard words"*. This is already the M3 transfer experiment
   in [08](08-roadmap.md), and it is the closest thing to real practice that costs
   nothing extra.

2. **Synergy = requiring a combination.** Represent each solution as a bag of
   recurring **move macros** (frequent subsequences mined from solution traces).
   Score a proposed instance by whether its solution requires a *pair* of macros
   the solver has only ever used separately. That is literally "a natural synergy
   of two existing approaches", and in this domain it is computable, because
   solutions are move sequences and macros are recurring subsequences of them.
   **This is the one I would build.** No existing self-play paper has it, and the
   braid environment makes it cheap.

3. **Generativity = reuse.** A problem is interesting iff its solution introduces
   a macro that is *later reused* across many other problems. Also computable
   from traces, and the closest computable analogue of "load-bearing" —
   the property that distinguishes the Riemann hypothesis from an isolated
   conjecture. See [07](07-domain-choice.md) §3, where this was the thesis-level
   contribution; it now has a concrete implementation path.

### Sequencing

The proposer has to earn its place rather than being assumed:

1. plain unknotting on simple knots, no generator at all;
2. instances from an **independent** generator (random, then table-derived), no
   adversarial component;
3. a proposer — admitted **only if** it beats the independent generator on the
   frozen benchmark `B`.

Step 3 is a hypothesis with a falsification condition, not a design commitment.
If an independent generator drawn from the knot tables matches a trained
proposer, the proposer is complexity for nothing and should be cut.

## Measuring progress honestly

Fit a 2PL IRT model over the round matrix: `P(solve | i, j) = σ(θ_i − b_j)`.
* `θ_i` — agent ability, `b_j` — item difficulty.
* Curriculum target: raise `K_t` so that `median_j b_j ≈ mean_i θ_i`.
* Report `θ` **only** on the frozen anchor set, plus a fixed external benchmark
  (for knots: unknotting-number upper bounds on the standard prime-knot tables).
