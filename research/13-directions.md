# 13 — Directions: the big moves

[08-roadmap.md](08-roadmap.md) is the schedule: milestones, exit criteria, weeks.
This file is the other thing — moves large enough to change what the system *is*,
written down before they are scheduled so they can be argued with while they are
still cheap to abandon.

Each entry has the same shape: the idea, why it is plausible *here* rather than in
general, what would have to be built, **the cheapest experiment that would decide
it**, and what would kill it. An entry with no killing condition is not a
direction, it is an enthusiasm.

| | direction |
|---|---|
| 1 | [An adaptive schedule, ordered by what the networks already believe](#1-an-adaptive-schedule-ordered-by-what-the-networks-already-believe) |
| 2 | [An arena between solvers, with recombination](#2-an-arena-between-solvers-with-recombination) |
| 3 | [A challenge set that is actually unlabelled](#3-a-challenge-set-that-is-actually-unlabelled) |
| 4 | [Every player learns from the best solution anyone found](#4-every-player-learns-from-the-best-solution-anyone-found) |

They are not independent: 4 homogenises the population that 1 and 2 need to
disagree, and all four need the same store, which is why the last section is about
that rather than about any of them.

---

## 1. An adaptive schedule, ordered by what the networks already believe

**The idea.** The ladder is a hand-written list of 41 rungs in a fixed order. Replace
it with an ordering derived from the value heads of every network in the population:
present a problem when *some* player's value head says it is nearly solvable, and
retire it when they all do. The curriculum becomes a function of what the population
currently believes, not of what the author guessed in advance.

**Why here.** The fixed ladder has already been caught being wrong about its own
difficulty. Rungs are ordered by `crossing_weight * c + scramble`, where `c` is the
length of the generated word — and [../docs/rungs.md](../docs/rungs.md) shows that
number is not a property of the knot. `R(3,22)#0` sits at rung 31, near the top,
and is the **unknot**. A value head that had ever seen it would say so. The
ordering signal is already in the system and is being ignored.

There is a second reason, from [01](01-game-design.md): item difficulty cannot be
estimated from one solver, because the outcome per problem is a single bit. A
population gives a *distribution* of value estimates per instance, which is exactly
the quantity a curriculum wants and a single agent cannot produce.

**What it needs.**
* A value-estimate sweep: every checkpoint scores every candidate instance. Cheap —
  one forward pass per (network, instance), no search.
* A scheduling rule over those estimates — approaches A to C below.
* Instance supply that is not a fixed list, so the schedule has something to choose
  from. The generator already produces more than the ladder uses.

### Approach A — measure first: does a value head rank anything correctly?

Do not build a scheduler. Take the existing checkpoints, score the 41 rungs with
each value head, and correlate the predicted ordering against the *measured*
difficulty already in the ladder results (iterations to promotion). If value heads
do not rank known rungs in the order the ladder actually found them hard, they
cannot rank unknown ones either, and everything below dies for the cost of a sweep.
This uses runs that already exist and is the prerequisite for B and C.

### Approach B — a block schedule over trust-weighted solve probabilities

Train by schedule, but start from the highest rung the current checkpoints already
clear rather than from rung 0, and present instances in **blocks**: score once per
block, order within it, then run the block. Blocks are what make the sweep
affordable and stop the ordering thrashing between iterations.

Within a block, order by how solvable the population believes each instance to be,
discounting players by how much their past promises were worth.

**Order on `max` and not on `min`.** The intuition to start with instances *some*
player thinks are easy is `max_x`. The pessimistic `min_x` says "every player thinks
it is easy", which is a much more conservative curriculum — and, more importantly,
it puts the trust term on the wrong side. With `min_x(p_x · trust_x)` the player who
*sets* the ordering is whoever has the lowest product, which an untrusted player
achieves by being untrusted: their low trust drags the min down, the instance looks
hard, it gets deprioritised. Trust exists to discount an unreliable opinion, and
inside a `min` it amplifies it. Under `max_x(p_x · trust_x)` an overconfident
player's `p` is scaled down *before* it can win the max, which is the intended
behaviour.

**Prefer recalibration to a scalar trust.** `trust(x)` is to be estimated from the
historical log-likelihood of a player's predictions — which is exactly the data that
fits a per-player calibration map. One scalar cannot separate "overconfident
everywhere" from "overconfident only near `p ≈ 1`", and it is the second that
matters, because `p ≈ 1` is the region the schedule selects on. Fit Platt or
isotonic per player, order on the recalibrated `p̂`, and the trust factor is
absorbed. Keep a scalar only as the shrinkage prior for players with too few
observations to fit anything.

**Condition trust on the search budget.** The value head is not `p(solve)`. It
predicts the return under this environment's multi-objective cost with FiLM
conditioning on `log(A/B)`, and it is a *prior*: MCTS at 128 simulations solves what
the raw head does not, which is the largest effect the ladder has measured. So
calibration is a property of `(network, simulation budget, scramble depth)`, not of
a network. Measured at 16 simulations, `search-light` will look like a liar for
being under-searched. Store the budget alongside the prediction or trust is
meaningless.

**Beware the order statistic.** `min` and `max` over a population are order
statistics, so their bias grows with population size: adding a sixteenth player
shifts every ordering, and the curriculum changes silently when the roster does. A
trust-weighted high quantile behaves the same way at the top of the distribution and
is stable under roster changes. Use it if the population is not fixed.

**Retirement is not optional.** Ordering by predicted-easiest is the mechanism that
produces the self-calibration-toward-easy drift [01](01-game-design.md) cites
PopuLoRA for. An instance must leave the pool once every player solves it reliably,
and the block boundary is the natural place to apply that test.

### Approach C — frontier and disagreement

The same machinery, different objective: order by the `4p(1-p)` frontier term from
[01](01-game-design.md) with `p` estimated across the population rather than
measured, or by *disagreement* — the variance of `p̂_x` across players. Instances the
population has not agreed about are where the information is, and disagreement has
the useful property of being immune to a common bias: it does not care whether the
whole population is overconfident, only whether they differ.

**What would kill the direction.** Value heads that are well-calibrated only on what
they were trained on — the usual failure, which would show up in A as a correlation
that is strong within cleared rungs and absent beyond the frontier. That is exactly
the region a curriculum needs it to work in. A weaker version survives: use value
estimates to *re-order what is already known to be solvable*, and keep a fixed
frontier.

---

## 2. An arena between solvers, with recombination

**The idea.** Stop running candidates as parallel independent arms scored on
"highest rung reached", and run them as a population in a competitive arena, where
what survives reproduces with variation — genetic programming over agents rather
than a sweep over hyperparameters.

**Why here.** The strongest single result in this project is that the ranking of
arms **inverts** between structured and unstructured knots: `u1-puct` is the worst
arm where `u = g` and a greedy positive-braid strategy works, and the best where it
does not (see the handoff table in `pgx-mcts-bench`). A scalar leaderboard cannot
represent that, and the sweep currently in use collapses it into one number and
picks a winner that is only a winner on half the domain. An arena that scores on a
*distribution* of instances keeps the disagreement, which is the part that carries
information.

**What it needs.**
* A shared instance pool the arms compete on, with per-instance results retained
  rather than averaged — the same data structure direction 1 wants.
* A variation operator. The honest ordering is: **selection first, recombination
  second.** Selection over the existing arms needs no new machinery and would
  already answer whether the population beats the sweep.
* A budget rule. Arms are not equally expensive; `search-heavy` buys its rung with
  simulations. Compare at matched compute or the arena measures spend.

**Cheapest deciding experiment.** Re-score the existing arms per instance rather
than per rung, and ask whether any *pair* of arms covers the instance set better
than the best single arm. If the best single arm dominates instance-wise, there is
no population to have and the direction dies without a line of new code. The
inversion above says it probably does not dominate — but that is one seed, and
[everything in this project with one seed has a coin-flip prior](README.md).

### 2.1 Composing networks into larger creatures

**The idea.** If two arms are complementary, the next question is whether they can
be *combined* rather than merely selected between: a composite agent that routes,
or a network assembled from trained parts.

**Why here.** [12](12-serial-formulation.md) already found that the serial
formulation's components are separable — the policy head had to be positional, the
head register was a clean negative, the accumulator is the read side. Those are
parts with interfaces, which is the precondition for recombination meaning
anything. And [06](06-network-growth.md) established the growth operators while
measuring that capacity is not currently the constraint — a composite is a way to
add capacity that is *earned* rather than assumed.

**What it needs.** In increasing order of ambition, and each is a stopping point:
a mixture-of-arms with a learned router; a shared trunk with per-arm heads,
trained jointly; genuine recombination of trained blocks between architectures.

**Cheapest deciding experiment.** An *oracle* router: pick, per instance, the best
of two existing arms, using the answer. That upper-bounds every routing scheme by
construction. If the oracle router does not beat the better arm by a margin worth
the engineering, no learned router will.

**What would kill it.** Two arms that fail on the same instances for the same
reason — complementary on paper, correlated in practice. The per-instance re-score
in 2 measures exactly this and should be run first.

---

## 3. A challenge set that is actually unlabelled

**The idea.** Filter generated instances on their *invariants* rather than on a
shallow unknotting search, so the unlabelled half of the ladder is unlabelled.

**Why here.** It is not speculative; it is repair. The generator's filters are "one
component" and "a depth-4 search failed to unknot it", and
[../docs/rungs.md](../docs/rungs.md) shows what got through: a 22-letter word that
is the unknot, six connected sums, and five knots with published unknotting
numbers. 19 of the 23 rung knots have an exact `u`. The challenge set is mostly
calibration set wearing a disguise.

**What it needs.** `rf_knots.invariants` already computes the fingerprint and names
the knot against 2870 tabulated ones. The generator would reject on identification
and on decomposition, and keep what survives.

**Cheapest deciding experiment.** Generate a thousand candidate words at each
crossing count and count how many survive the invariant filter. If genuinely
unlabelled knots are common at 20+ letters, the fix is a filter. If they are rare,
the interesting question changes: the ladder should *attach* the published `u` and
grow the calibration set, which is the scarce resource anyway.

**What would kill it.** Nothing kills it — the measurement is worth having either
way. It is here because it is a fork in what the ladder is for, and that is a big
step even though the code is small.

---

## 4. Every player learns from the best solution anyone found

**The idea.** After each rung the population has, between them, a best solution for
each knot. Train every network against it, not only the one that found it. The
population then propagates forward together instead of each arm rediscovering the
same reduction.

**Why here.** The obvious objection to cross-agent imitation is that a bad teacher
poisons the student. That objection does not apply in this environment. Every move
preserves the knot type, so a shared solution is **verifiable, not trusted** — a
receiving player can check it by replaying it. The worst a bad teacher can do here
is be expensive, and expense is measurable. This is a property of a
machine-verifiable domain that most distillation schemes do not get for free, and
it is the same property the whole project is built on.

### The action-space problem mostly dissolves

The apparent difficulty is that players do not share an action space: the parallel
formulation's action is `(move kind, position)` over the whole word, while the
serial one is a head with `O(1)` actions relative to its position, a window width,
and a stride set. Mapping one action sequence to another pairwise is `n²` awkward
translations.

It is the wrong intermediate representation. **Every action is a deterministic
rewrite of the braid word**, so a solution is fully determined by the states it
passes through:

```
w_0 -> w_1 -> ... -> w_n
```

A receiving player needs no correspondence with the sender's action space. At each
step it searches *its own* legal actions for the one whose result is `w_{i+1}` —
which is `reference.apply` against the legality mask the environment already
computes, with `reference.equal_up_to_rotation` for moves that cross the seam. One
canonical form, `n` players, no pairwise table.

**The serial head is the one real cost.** A serial receiver can only act at the
head, so translation must insert head travel: a shortest path over
`serial_shift_strides`, exactly solvable by breadth-first search and cheap. But
those inserted moves are *charged*. The objective is
`A · crossing_changes + B · total_moves`, and [12](12-serial-formulation.md)
measured serial arms responding to `log(A/B)` by 5-6x precisely because head travel
costs them. So a parallel-optimal trajectory, translated, is **not optimal in the
serial player's own metric**, and imitating it teaches a target that is wrong by
construction.

The fix is to share the part that does not depend on the metric: **the crossing-change
decisions** — which crossing is changed, and at what point in the reduction. That is
the scarce resource, it is what `u(K)` counts, and it survives translation intact.
The moves between crossing changes should be re-derived by the receiver under its
own cost.

### Turn one global move into a short serial option

There is a more direct way to combine a strong parallel player such as
`u1-puct` with a more general serial player such as `s-gru128`, `s-fsa32`, or
`s-tape4`. Let `u1-puct` answer only the global question: **which braid
rewrite should happen next?** Then ask the serial controller to realise that
rewrite within a horizon of one, two, or three of its own actions:

```
serial preparation ... -> serial preparation -> teacher's braid rewrite
```

The final action is the move proposed by `u1-puct`. The zero, one, or two actions
before it are receiver-side preparation: shift the window, scan, or update memory
so that the proposed move becomes locally available. These preparation actions
do not alter the braid word. Thus a length-one option executes an already visible
teacher move, a length-two option needs one preparation action, and a length-three
option needs two. The label is not a foreign action ID; it is a short sequence in
the serial player's own action space whose final state is the teacher's proposed
next word.

This is an options formulation rather than policy averaging. The parallel expert
chooses a state-space subgoal `w -> w'`; the serial expert controls how to reach
it and pays for every shift and memory operation. At training time, breadth-first
search over depth at most three can certify the shortest valid serial macro and
provide policy targets for each prefix. At inference time the same construction
can be used either as distillation — the serial net eventually acts alone — or as
a fused agent in which the global expert proposes a fresh subgoal after every
completed or abandoned option.

Do not force a teacher move when no length-three macro exists. Record that as
`unreachable within horizon`, let the serial policy act normally, and ask the
global expert again after the window or memory state changes. Report coverage at
each horizon separately: the fractions executable in one, two, and three actions,
plus the charged move inflation. That measurement says whether the global expert
is useful guidance or merely an expensive oracle whose answers the serial player
usually cannot reach.

### Teacher first, fusion second

Do not begin by averaging weights or policy logits. `u1-puct` and the serial
players solve different control problems: one assigns probability to absolute
rewrites across the whole word, while the other assigns probability to local
rewrites, shifts, scans, and memory actions. Equal-sized tensors would not make
those meanings equal.

The first combined candidate should therefore use a **frozen `u1-puct` teacher**.
For every training state within the parallel player's supported length, project
its answer into serial semantics:

* copy probability mass for teacher moves already executable in the visible
  window;
* turn mass to the left or right of the window into targets for head travel;
* use the certified one-to-three-action options above when a short realisation
  exists;
* leave tape writes, register changes, and other private memory actions to the
  serial player's own search; and
* distil the teacher's solve probability and conditional crossing-change and
  move estimates into the corresponding auxiliary heads.

These are soft hints, not facts. A teacher prediction may be wrong; only a replayed
solution witness supplies the admissible bound described below. Keep the ordinary
search loss and the witness-bound loss, and give teacher imitation a tunable
coefficient so the student can contradict it when its own experience is better.

The decisive advantage of this stage is that the teacher disappears at inference.
`s-tape4`, `s-gru128`, or `s-fsa32` keeps its serial action space and can run on
words longer than the parallel teacher's training envelope. The experiment asks
whether privileged full-word supervision can teach better scanning and memory,
not whether the serial agent can call an oracle forever. Name and compare these
arms explicitly — for example `s-u1-distill-tape4`, `s-u1-distill-gru128`, and
`s-u1-distill-fsa32` — against both parents and against serial arms trained only
from shared witnesses.

Only if distillation helps should the global expert remain inside the deployed
model. A true dual-expert network has a full-word branch and a serial
window-and-memory branch, joined by a learned gated residual rather than an
average:

```
z = z_serial + sigmoid(g(state)) * project(z_global)
```

The global branch proposes a region or next-word subgoal; the serial branch
chooses the locally legal action and maintains memory. Initialise each branch
from its existing checkpoint, initialise the projection to zero so the fused
model begins as the serial parent, train only the projection and gate for a short
warm-up, then unfreeze both parents at a lower learning rate. The gate must be
able to go to zero when the word is outside the global branch's supported length
or when its advice conflicts with a verified witness.

The minimum ablation is four ways to use the same data: serial alone, witness
sharing only, frozen-teacher distillation, and permanent gated fusion. Report
solve rate and charged cost, option coverage at horizons one to three, performance
beyond the teacher's length range, and disagreement with `u1-puct`. A gain only
inside the teacher's range is useful acceleration; a gain beyond it is actual
knowledge transfer.

### Train on it as a bound, not as a target

A known solution gives an *admissible upper bound* on the cost-to-go from every word
on its path. Using it as an equality target for the value head asserts the path is
optimal, which is exactly what is not known — the ratchet exists because these are
upper bounds. A one-sided hinge, penalising only value estimates worse than the
known path implies, is correct even when the shared solution is beatable, and stays
correct as the ratchet improves. For the policy head the AlphaZero rule from
[06 Part III](06-network-growth.md) applies: distil the search-improved targets
`π ∝ N^(1/τ)` along the trajectory, not the raw action sequence.

### The cost is diversity, and it is not small

This is a homogenising force, and it runs directly against directions 1 and 2. The
schedule needs players to disagree about difficulty; the arena needs them to fail on
*different* instances; item difficulty is unestimable from correlated solvers
([01](01-game-design.md)). The strongest measured result in the project is that the
ranking of arms **inverts** between structured and unstructured knots — training
every net on the same best solutions after every rung is a machine for destroying
exactly that.

So: keep a subpopulation that never receives shared solutions. It is the control
that says whether sharing helped, and the reservoir that keeps the arena in 2 worth
running.

**Prerequisite, already on the books.** `bounds.jsonl` records the knot's defining
word and the crossing-change *count*, not the sequence — storing the unknotting
sequence as the witness was already an open item, motivated by wanting to
re-verify a claimed bound rather than trust it. This direction cannot start until
that exists. Same object, two independent reasons to build it.

**Cheapest deciding experiment.** Do not train anything. Build the translator and
measure it: replay each arm's best trajectory on cleared rungs, translate it into
every other arm's action space through the word sequence, and report two numbers —
the fraction that translates at all, and the move-count inflation when it does.
"Doable but complicated for some pairs" is a claim that can be measured before a
single gradient step. If parallel-to-serial inflates moves threefold, full-trajectory
sharing is dead on arrival and only the crossing-change decisions should be shared.

**What would kill it.** Arms whose best solutions are already near-identical, so
there is nothing to share; translation inflation large enough that the shared target
is worse than the receiver's own solution; or a measured collapse in instance-wise
diversity, which the coverage metric in direction 2 detects directly.

---

## What these four have in common

All four want the same missing object: **per-instance results, retained across runs
and across arms.** The schedule in 1 orders instances by what players believe about
them and needs `(player, instance, budget, predicted value, outcome)` rows to
estimate trust at all; the arena in 2 compares arms instance by instance; 3 decides
which instances should exist; and 4 needs the *witness* — the move sequence, not
just its length — attached to each best-known bound.

Today the ladder records a rung number and an average. Building that store is the
prerequisite for all four, it is small, and it should happen first.
