# 13 — Directions: the big moves

[08-roadmap.md](08-roadmap.md) is the schedule: milestones, exit criteria, weeks.
This file is the other thing — moves large enough to change what the system *is*,
written down before they are scheduled so they can be argued with while they are
still cheap to abandon.

Each entry has the same shape: the idea, why it is plausible *here* rather than in
general, what would have to be built, **the cheapest experiment that would decide
it**, and what would kill it. An entry with no killing condition is not a
direction, it is an enthusiasm.

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
* A scheduling rule over those estimates. The obvious three, in increasing order of
  ambition: *easiest-first* (max over players of predicted value), *frontier* (the
  `4p(1-p)` term from [01](01-game-design.md), now with `p` estimated across the
  population rather than measured), and *disagreement* (highest variance across
  players — the instances the population has not agreed about are where the
  information is).
* Instance supply that is not a fixed list, so the schedule has something to choose
  from. The generator already produces more than the ladder uses.

**Cheapest deciding experiment.** Do not build a scheduler. Take the existing
checkpoints, score the 41 rungs with each value head, and correlate the predicted
ordering against the *measured* difficulty already in the ladder results (iterations
to promotion). If value heads do not rank known rungs in the order the ladder
actually found them hard, they cannot rank unknown ones either, and the direction
dies for the cost of a sweep. This uses runs that already exist.

**What would kill it.** Value heads that are well-calibrated only on what they were
trained on — which is the usual failure and would show up as the correlation above
being strong within cleared rungs and absent beyond the frontier. That is precisely
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

## What these three have in common

All three want the same missing object: **per-instance results, retained across
runs and across arms.** The schedule in 1 orders instances by what players believe
about them, the arena in 2 compares arms instance by instance, and 3 decides which
instances should exist at all. Today the ladder records a rung number and an
average. Building that store is the prerequisite for all three, it is small, and it
should probably happen first.
