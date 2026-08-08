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
| 3 | [Every player learns from the best solution anyone found](#3-every-player-learns-from-the-best-solution-anyone-found) |

They are not independent: 3 homogenises the population that 1 and 2 need to
disagree, and all three need the same store, which is why the last section is about
that rather than about any of them.

[16-scientists-collaboration.md](16-scientists-collaboration.md) turns the three
directions into one controlled programme: a moving frontier of knot
representations, diverse rung-18 scientists, calibrated per-scientist scheduling,
verified best-witness exchange, periodic improvement attempts, and compute-matched
static, no-sharing, supervised, and single-scientist controls.

[00-glossary.md](00-glossary.md) defines the experiment vocabulary, objectives,
denominators, protocol versions, and scientist architectures used below.

[17-scalable-braid-raster.md](17-scalable-braid-raster.md) specifies a new
number-of-strands-agnostic direction: compile braid words into a lossless
strand-by-word raster, use a mathematically faithful cylinder rather than a
naive torus, share local and hierarchical blocks across all sizes, and select
the family through exact representation, extrapolation, curriculum, and paired
mixed-strand gates.  Its first controlled arm, `conv-window-128`, changes only
the input/trunk of `s-window-128` so the representation can be tested before the
action space is redesigned.

The seed-71 first-stage smoke rejected the initial joint-3x3 trunk (50% SR) but
not the representation: separate axial interactions and a four-repeat shared
axial block both matched `s-window-128` at 100% SR after two iterations, with
fewer parameters.  A paired five-rung gate is running; no mature-quality claim
is made from the elementary smoke.

The main raster experiment now has a verified toroidal action alphabet, `B*`.
Its seam letter is the Birman--Ko--Lee band `a_(1,k)`, compiled back to ordinary
Artin `B_k`; it is not an unchecked affine braid.  From-scratch
`s-window-128-bstar` retained 100% elementary solve rate after two iterations.
The axial and recurrent `B*` arms were at 50% after three iterations, but delayed
takeoff was real: axial promoted at 83.3% after iteration 4 and recurrent at 100%
after iteration 6.  All three fresh `B*` arms are now running the paired five-rung
comparison.  See
[the verified torus protocol](17-scalable-braid-raster.md#verified-toroidal-action-alphabet-b).

The intended 200-representation comparison is still gated. Historical v6--v10
sharing runs charged receiver-internal controller plies as solution `moves`, so
their objective comparisons do not answer the intended solver-independent
semantic-`L10` question. Protocol v11 now verifies a portable semantic witness,
charges only its braid edits, and reports receiver-native/internal plies only as
compute. Real rung-18 checkpoints migrate to the new remaining-budget input with
exactly identical outputs and actions. An `s-tape4` early-rung critic gate made all
10 curves budget-sensitive and monotone without reducing promoted-rung solve rate.

The first 25-representation v11 preflight used 12 routable superior donations,
six preservation canaries, and six identities untouched by donation, replay, or
preservation loss. At 16 evaluation simulations, sharing and control solved the
same single identity; sharing shortened it by six `L10` points but rescued no
target. A post-hoc 64-simulation sweep produced one sharing-only canary, but a
fresh 64-simulation/four-attempt run did not replicate it: both arms again solved
only `10_124`, and both lost the frozen `10_100` solve. Thus sharing remains
plausible at longer search horizons, but it has not passed a fresh preflight.
The three-seed confirmation, 30--50-item pilot, 200-item arms, and CPU-32 run
remain closed. The bank is not an all-`u=1` set: independent lower bounds certify
`u >= 2` for 87/200 BASE identities. See
[the semantic-cost v11 result](16-scientists-collaboration.md#semantic-cost-sharing-gate-v11-2026-08-07).

The corrected roster gate is also complete. On a source-disjoint mixed 3/4-strand
frontier at 256 simulations and four exactly paired stochastic attempts,
`s-window-128`, `s-cyclic-tape8-192`, and `s-head-128` solved 8/12, 7/12,
and 3/12 representations. Their union was only 8/12; cyclic and head added no
identity beyond window. Consequently no adaptive/static/sharing arm was started.
The next valid fork is either an explicitly 3-strand-only 100+ representation
methods benchmark, with harder strands reported separately, or forward training
of a genuinely complementary third scientist. See
[the paired roster-readiness result](16-scientists-collaboration.md#paired-roster-readiness-gate-2026-08-07).

The broader fork now has an executable replacement for `s-head-128`:
`s-strand-graph-128` compiles a compulsory full-representation scan into exact
crossing neighbours along both physical strands, applies five cyclic/strand graph
blocks, and scores head shifts from the edit sites they reach. The 794,676-parameter
prototype is locally trainable and costs 1.412 ms per batch-1 forward, but it has
not passed a forward curriculum or held-out complementarity gate. `s-head-128`
therefore becomes a historical control; the new model enters the long roster only
after those gates. See
[the replacement design](16-scientists-collaboration.md#replacement-third-scientist-s-strand-graph-128-2026-08-08).

The architecture screen now includes compact (64 x 3), balanced (96 x 5), wide
(160 x 8), and exact local-plus-global two-tower variants, with their real learning
rate, batch size, weight decay, and update count carried into the runner. A
calibrated 128-simulation check confirms that early learning can be delayed:
`s-head-128` rose from 75% at cycle 4 to 94.4% and promotion at cycle 6 on seed
71. This does not rescue its mature held-out 3/12 result or its 0/5 four-strand
coverage. The compact graph reached 88.9% after one cycle on the same seed; fresh
seed 73 confirmed at 100% after three cycles, while seed 72 remained unstable. The
balanced graph reached 80.6% after two cycles. Clean mixed-strand continuations
now enforce exact known-`u` promotion. A corrected single-seed five-rung test
shows that fixed `F_old=1` is too weak: per-rung adaptive rehearsal, using a
24-attempt retention certificate and doses `1 -> 2 -> 4 -> 8`, improved the
paired final result from 112/120 solves (fixed) and 115/120 (original baseline)
to 118/120, with exact `u` on every solved attempt. Wide and local-plus-global
admission gates remain open. See
[the capacity and optimizer screen](16-scientists-collaboration.md#capacity-and-optimizer-screen).

The fully scalable raster controller now has a full identity-column canvas, a
shared recurrent spatial action pointer, and a dynamic `B*` seam that wraps at
the state's active strand count rather than at tensor capacity. A six-run
negative control showed why the first factorized versions appeared broken:
using a random `p(solve)`/cost critic to steer MCTS immediately produced only
0--66.7% elementary SR. Protected stage-0 warmup keeps factorized heads in
shadow, detaches solve loss from the encoder, and samples four positions per
selected replay episode. With those corrections, the cylinder returned to 12/12
solves with zero crossing changes after two cycles on all three seeds. The torus
passed two seeds but failed crossing-change quality on seed 73, so the cylinder
alone advances to the adaptive six-stage mixed-strand gate. See
[the scalable full-canvas gate](17-scalable-braid-raster.md#full-canvas-scalable-controller-and-critic-warmup-2026-08-08).

The collaboration implementation has also advanced to protocol v7. It provides
bounded adaptive old-task rehearsal, permanent best-solution replay, paired
population retention with transactional rollback, and strict direct donation
only when receiver-native replay proves a lower semantic objective. The corrected
two-round sharing/no-sharing/solo smoke exactly matched qualification, full-search,
rehearsal, and retention simulations; sharing and no-sharing selected the same
two tasks and retained their portfolios. This validates plumbing only. The
100--200 representation pilot still waits for the third scientist's three-seed,
mixed-strand, and source-disjoint readiness gates. See
[the v7 runner smoke](16-scientists-collaboration.md#transactional-collaboration-runner-v7-2026-08-08).

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

A concrete window/scan/tape recombination attempt, including its pinned checkpoints
and evaluation protocol, is recorded separately in
[15-recombination-triad-attempt.md](15-recombination-triad-attempt.md).

---

## 3. Every player learns from the best solution anyone found

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

## What these three have in common

All three want the same missing object: **per-instance results, retained across runs
and across arms.** The schedule in 1 orders instances by what players believe about
them and needs `(player, instance, budget, predicted value, outcome)` rows to
estimate trust at all; the arena in 2 compares arms instance by instance; and 3
needs the *witness* — the move sequence, not just its length — attached to each
best-known bound.

Today the ladder records a rung number and an average. Building that store is the
prerequisite for all three, it is small, and it should happen first.
