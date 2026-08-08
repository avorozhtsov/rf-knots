# 00 — Glossary for the knot-learning experiments

This glossary defines the vocabulary used in the research notes and in the
`pgx-mcts-bench` experiment reports. It is deliberately more precise than the
informal discussion: many apparent contradictions came from using *solution*,
*cost*, *dose*, or *retention* for several different quantities.

The collaboration design is developed in
[16-scientists-collaboration.md](16-scientists-collaboration.md). The current
implemented sharing protocol is **v11**. It charges portable semantic witness
steps in `L_A:B` and records receiver-native/internal plies separately as compute.
Entries for v6--v10 explain historical artifact labels; their results must not be
silently pooled with v11 because both training rules and move accounting changed.

Contents: [notation](#quick-notation) ·
[problems and solutions](#problems-searches-and-solutions) ·
[sharing](#sharing-vocabulary) ·
[search and compute](#search-evaluation-and-compute) ·
[learning diagnostics](#learning-paths-and-diagnostics) ·
[protocol versions](#protocol-and-artifact-names) ·
[architectures](#architectures) ·
[reporting checklist](#reporting-checklist).

## Quick notation

| symbol | meaning |
|---|---|
| `x` | one concrete braid representation, not merely a knot type |
| `i` | a scientist, meaning a solver architecture plus checkpoint and search configuration |
| `cc` | number of crossing-change actions in a verified solution |
| `moves` | charged semantic braid actions in the verified witness |
| `L_A:B` | objective `A * cc + B * moves` |
| `L10` | shorthand for `L_10:1 = 10 * cc + moves` |
| `E` | evaluation attempts per representation |
| `T` | the registered set of training-witness identities |
| `S(pi)` | representations solved at least once by policy/checkpoint `pi` under the declared evaluation protocol |

The experiment must always state `A:B`, simulations per move, attempts per
representation, seeds, action budget, and failure penalty. An `L10` result
without those search conditions is incomplete.

The intended comparison uses a solver-independent semantic objective. Historical
v6--v10 sharing artifacts instead counted every receiver-native controller ply as
an objective move. That is a different metric and those artifacts must be rerun
before they are used as evidence about semantic `L10` sharing.

## Problems, searches, and solutions

### Braid representation

A concrete braid word together with its strand count. One mathematical knot can
have several braid representations with different word lengths and search
difficulty. The 200- and 2,700-item experiments schedule representations, so
their denominators are representations unless a report explicitly deduplicates
by knot identity.

### Scientist

A solver with a named architecture, a particular checkpoint, its native action
space, value and policy heads, and an MCTS configuration. Two checkpoints of
`s-window-128` are two scientist states even though they share an architecture.
The population arms compare learning systems, not just bare neural networks.

### Native action and semantic action

A **semantic action** edits the global braid and can be carried between
architectures. A **native action** is one action in a particular scientist's
game. For a serial scientist, native actions include head shifts, tape writes,
memory-state changes, and the eventual local braid edit. These controller-only
actions do not appear in the portable semantic witness and do not count as
`moves` in `L_A:B`. They still consume native episode/search limits and the
controller's internal-action allowance.

### Charged action

A **charged action** is one verified semantic braid action represented by a
`WitnessStep`. It changes the shared braid state and contributes one unit to
`moves`. A crossing change is both one charged action and one crossing change,
so under `L10` it contributes `10 + 1 = 11`.

Head shifts, scans, tape writes, register changes, and memory-state changes are
**uncharged internal controller actions** for the solution objective. They are
not free computation: count them separately as `native_plies` or
`internal_plies`, and enforce the declared controller and episode budgets. Do
not call those plies `moves` in an `L_A:B` result.

### Attempt, episode, trajectory, solution, and witness

- An **attempt** or **episode** is one complete stochastic search rollout for one
  representation under a fixed search budget. It may fail.
- A **trajectory** is its ordered native state/action record. It may also fail.
- A **solution** is a trajectory that ends in a verified one-strand empty braid.
  “Solved” always means within the declared action and search budgets; it does
  not claim the network could never solve the item with a larger budget.
- A **witness** is the machine-verifiable portable proof extracted from a
  solution: semantic braid edits, costs, and provenance. Its edits can be
  translated into another scientist's native controller.

“Unknotting with no `L` budget” still has finite implementation limits such as a
maximum episode length. It means that the objective channel does not prune the
search at a small `L_max`, not literally infinite computation.

### Paired attempt

Two attempts are **paired** when sharing and control evaluate the same concrete
representation at the same attempt index with the same registered random seed,
MCTS simulations per move, semantic objective cap, native episode limit, and
other search settings. The trained policies may choose different actions and
consume different wall time; pairing aligns the stochastic comparison rather
than forcing identical trajectories.

The pairing key should be stored explicitly, for example
`(representation_id, attempt_index, seed, protocol_hash)`. Solved-set and cost
comparisons must preserve this pairing before results are aggregated across
representations or seeds.

### Verified solution

A proposed trajectory whose actions are legal, whose final state is the unknot,
and whose reported `cc` and `moves` agree with replay. Collaboration never trusts
the donor's scalar prediction as a proof. Verification and cost recomputation
happen before a witness can train a receiver.

### Objective and charged semantic objective

For objective ratio `A:B`,

```text
L_A:B = A * cc + B * moves.
```

The current main comparison uses `A=10`, `B=1`. `moves` is the number of semantic
steps in the verified portable witness. Therefore a donated witness has the same
`cc`, `moves`, and `L_A:B` for every receiver. Translation may require a different
number of internal native plies, but that affects feasibility and computation,
not the solution objective.

### Capped loss

The panel-wide objective that makes solved-set coverage and solution quality
comparable in one number. If `b` is the preregistered failure penalty and
`best_L(pi,x)` is the best verified objective among the `E` attempts, then

```text
CappedLoss(pi, X) = sum over x in X of
                    (best_L(pi,x) if x was solved, else b).
```

Lower is better. In the current 17-item ratio-10 gate, the implementation uses
`b = 10 * 20 + 64 = 264`. The failure cap must be reported because changing it
changes the trade-off between solving another representation and shortening an
existing solution.

### Solved set and solve rate

`S(pi)` contains representation `x` if at least one of its `E` paired attempts
solves it. Panel solve rate is `|S(pi)| / |X|`. Attempt solve rate is a different
quantity: successful attempts divided by `E * |X|`. Reports should not call the
first number “SR” without its representation denominator.

### BASE and held-out sets

`BASE` is the fixed difficulty-ordered representation list used by the proposed
curriculum. `NEW_70` and smaller canary panels are held-out evaluations. A
representation is held out only relative to a specified source of training:
being absent from the current witness block does not prove that an old rung-18
checkpoint never encountered an equivalent knot or representation.

A canary whose state is used in an off-route preservation loss is **non-target**,
but not strictly held out. A generalization panel must be absent from donation,
replay, preservation losses, and model-selection decisions.

## Sharing vocabulary

### Donor

The **donor** is the scientist that authored a verified solution selected for
sharing, or the provenance-preserving archived replay of that scientist's
solution. In the collaboration loop this is normally the lowest-`L_A:B` verified
solution found so far for that representation among the evaluated scientists
and attempts. It is the best *observed* solution, not a proof of the globally
minimal `L_A:B`.

The neural network supplies the policy/value guidance and MCTS produces the
native trajectory. Verification then extracts the portable semantic witness;
only that verified witness is eligible for donation.

### Donation and receiver

The **donation** is the donor's verified portable semantic witness. The
**receiver** is the scientist offered that witness and potentially trained to
realise its semantic actions through its own controller.

Avoid the phrase **donated receiver**: the receiver is not donated. When an old
report says “donated receiver cost,” inspect the field definition: it may mean
the receiver-independent semantic `donated_L`, or the historical
`translated_native_plies`. New reports must name those quantities separately.

### Compact donor witness

A **compact donor witness** is the portable `UnknotWitness` extracted from the
donor's longer native trajectory. Consecutive native states with an unchanged
braid are collapsed, so head shifts, scans, tape writes, register changes, and
other controller-only work disappear. What remains is the initial braid plus the
complete verified sequence of semantic braid actions and resulting braid states.

“Compact” means **controller-elided**, not approximate and not necessarily
globally shortest. The compact witness preserves the donor solution's `cc`,
semantic `moves`, and `L_A:B`. Prefer the less ambiguous phrase **portable
semantic witness** in new reports.

### Donated receiver cost (also written “donated receive cost”)

This is a historical and potentially misleading term. Under the intended
semantic objective, the donation's solution cost is receiver-independent:

```text
donated_L = A * donated_cc + B * donated_semantic_moves.
```

The receiver may require additional head motion, tape writes, or memory changes;
report those separately as `translated_native_plies`. They determine whether the
receiver can realise the donation within its internal and episode budgets, but
do not alter `donated_L`. Historical v6--v10 code used the receiver-native ply
count here; that implementation does not match the intended objective.

### Frozen native cost

The best verified semantic objective found by the untouched starting checkpoint
during its frozen, paired pre-training evaluation:

```text
frozen_native_L(x) = min L among the E frozen attempts on x.
```

If all attempts fail, this finite cost is unknown; panel scoring uses the declared
failure penalty instead. Frozen native cost is a measurement under one seed block,
not the receiver's true optimum.

### Native incumbent

The best verified semantic objective archived from solutions the receiver found
natively for `(receiver, representation, A:B)`. “Native” describes who found the
solution; it does not mean that internal native plies enter `L`. The incumbent
begins with available frozen-evaluation solutions and can improve during native
refresh. Donation eligibility is checked against this persistent incumbent, not
only against the latest stochastic attempt.

### Superior, equal, inferior, and stale donations

A donation is **active/superior** for a receiver only when its verified semantic
objective is strictly smaller than the receiver's archived native semantic
incumbent. A rescue of an item with no native solution is also superior. Equal or
larger costs are **inferior** and cannot train the policy. Translation feasibility
is a separate requirement.

A previously superior donation becomes **stale** once native learning finds an
equal or better solution. Eligibility is rechecked before every v11 sharing block.
Stale records may retain safe one-sided critic information, but they are excluded
from imitation targets.

### Inferior donation

Specifically, a donated solution with

```text
donated_semantic_L >= best_archived_native_semantic_L.
```

Equality is intentionally rejected. Imitating an equal or worse foreign route can
replace a receiver's useful native method without improving the research
objective.

### Training witness

A certified donated witness selected as a potential policy-learning target for a
registered training identity. Three counts must be kept separate:

1. **translated witnesses**: translation into the receiver succeeded;
2. **training witnesses**: translated witnesses belonging to the registered
   target set `T`;
3. **active training witnesses**: training witnesses still strictly better than
   their receiver's native semantic incumbents at a particular sharing block.

The historical report field `trained_witnesses` records category 2. It does not
prove that every listed witness produced an adapter update.

### Receiver-unsolved witness

A donation for a representation the frozen receiver did not solve under the
declared screening attempts and seeds. This is evidence of a possible transfer
opportunity, not proof of absolute inability. Strong screens use several separated
seed blocks because a one-block “unsolved” label is noisy.

### Zero policy updates

A cycle or scheduled event in which no donated trajectory updates the option
policy adapter. Common reasons are:

- fewer than the required number of distinct active superior witnesses;
- the sharing block is not due yet;
- the current or donated witness became stale or inferior;
- no legal canonical route exists within the internal-action horizon.

This does **not** mean no learning occurred. Native self-play/replay updates can
still run, the control still receives its paired native update, and stale evidence
may still supervise a safe critic bound. In v11, the report field
`zero_policy_updates` refers only to the sharing policy path.

### Sharing arm and control arm

The **sharing arm** receives ordinary native RL plus eligible donated-solution
adapter updates. The paired **control arm** starts from the same checkpoint,
receives the same native-refresh opportunities, and gets extra native optimizer
work matched to the sharing work. “Without sharing” must not mean “with less
training compute.”

### Sharing effectiveness

There is no valid single unqualified “sharing effectiveness” percentage. The
primary v11 causal comparison is between paired sharing and compute-matched
control policies on the complete frozen panel:

```text
Delta_cap = CappedLoss(control) - CappedLoss(sharing).
```

Positive `Delta_cap` favours sharing. A valid report also gives:

- exact `sharing-only = S(sharing) - S(control)` identities;
- exact `control-only = S(control) - S(sharing)` identities;
- the intersection `S(sharing) intersect S(control)`;
- summed objectives for each arm on that intersection;
- target transfer rates with denominator `|T|`;
- compute consumed by each arm; and
- results across paired seeds.

The intended per-receiver primary condition is non-worsening semantic capped loss after
at least one real sharing block. The overall gate additionally requires at least
one sharing-only solved identity. Canonical-route loss and exact retention are
diagnostics, not definitions of effectiveness.

### Distillation

Supervised policy learning from a verified successful trajectory found elsewhere.
Here it teaches a receiver how to execute translated braid edits through its own
controller. It is distinct from native AlphaZero-style reinforcement learning,
where MCTS supplies search-improved policy targets from the receiver's own play.
Failed attempts are not “imitated”; they can contribute solve-probability or
one-sided value evidence, but not successful action targets.

### Sharing block (v10/v11)

A scheduled fixed-dose distillation event over the current set of active superior
witnesses. By default it requires at least ten distinct witness identities, occurs
every ten cycles, samples four canonical-route positions per witness, and applies
sixteen adapter optimizer steps. Every eligible representation contributes the
same number of sampled positions, so a long solution cannot dominate merely by
having more actions.

The threshold of ten is a minimum for starting a block, not a claim that every
witness is individually compatible with the receiver. Per-witness route-loss
regressions are allowed; external paired performance decides whether the block
was useful on average.

## Search, evaluation, and compute

### MCTS

Monte Carlo Tree Search. At each environment move it uses the neural policy/value
as a prior/evaluator, explores a search tree, and selects an action from the
search-improved distribution. MCTS is repeated at the next state; it is not one
tree built once for the whole episode.

### Simulations per move

The number of MCTS simulations scheduled before selecting each environment
action. `64 simulations` means up to 64 tree traversals/evaluations at *every
move*, not 64 episodes and not 64 total network calls for the representation.

Training and evaluation simulations are separate v10 manifest fields. A policy
may learn better with 64-simulation self-play yet be evaluated more reliably with
128 or 256 simulations. Reports also record scheduled network evaluations because
episode length and early termination make total compute differ even at a common
simulations-per-move setting.

### Evaluation attempt per representation

One independently seeded MCTS episode from the same initial braid representation
with the frozen evaluation policy and declared search/action budgets. With `E=8`,
the representation is counted as solved if any of its eight attempts succeeds,
and its `best_L` is the minimum objective among those successful attempts.

Increasing `E` raises the chance of observing a stochastic solve. Results with
different `E` are not directly comparable unless recomputed.

### Round

One newly selected representation in the long collaboration experiment. All
scientists evaluate that representation under the declared objective ratios,
the outcomes enter replay and the solution archives, and the processed
representation is replaced in the active task frontier. A round is a data-
collection unit; it does not necessarily perform an optimizer update.

### Training block

The consecutive rounds accumulated between scheduled training events. With the
current default `train_every=10`, one training block contains ten rounds and
therefore normally ten newly selected representations. Training batches may
combine those new outcomes with positive and negative examples from permanent
historical replay banks. An all-success or all-failure block is retained; the
missing outcome class is supplied from history rather than dropping the block.

### Cycle

The smallest repeated unit in the interleaved collaboration gate. In v11 a cycle:

1. selects the next registered target representation;
2. optionally collects paired native-refresh attempts;
3. adds verified outcomes to replay/incumbent archives;
4. performs one ordinary native replay update in both arms; and
5. if a sharing boundary is due and enough active witnesses exist, performs one
   sharing block and compute-matched extra native control updates.

Thus `16 cycles` is neither 16 MCTS simulations nor 16 optimizer steps. The exact
number of trajectories and updates depends on the manifest fields below.

In the long collaboration runner, prefer **round** and **training block** over
`cycle`: a round processes one new representation, while a default training
block processes ten. This avoids confusing the v11 gate's one-representation
cycle with a multi-round scheduled training event.

### Native-refresh attempts per cycle

The number of fresh native self-play attempts generated on the cycle's current
representation for each arm before learning. Sharing and control use identical
seed schedules, search budgets, and attempt counts, although their changed
policies can produce different trajectories. `native_refresh_games=4` therefore
means four attempts for sharing and four paired attempts for control per cycle.

Native refresh serves two roles: it supplies current-policy replay and it can
improve the native incumbent, which may make an old donation stale.

### Dose

The amount of one explicitly named intervention. This word is ambiguous and must
be qualified:

- **MCTS dose**: simulations per move;
- **evaluation dose**: simulations per move and attempts per representation;
- **native-refresh dose**: fresh attempts per cycle;
- **sharing dose**: sharing blocks, adapter steps per block, and routed state
  examples;
- **native-learning dose**: optimizer steps and state examples consumed.

An `adapter dose of 16` and a `search dose of 64` are unrelated units.

### Economical dose experiment

The fixed-checkpoint MCTS-dose sweep used to separate search at learning time
from search at evaluation time. Instead of retraining for every evaluation
budget, it re-evaluated the already trained 64- and 128-simulation checkpoint
pairs at 32, 64, 128, and 256 simulations with matched seeds.

It is “economical” because the expensive learning runs are reused. It estimates
whether an observed result is merely under-searched at evaluation; it does not
measure how retraining at each dose would change the network. The artifacts are
named `sharing-simulation-dose-v1-train64-*` and
`sharing-simulation-dose-v1-train128-*` in `pgx-mcts-bench/artifacts/`.

### Compute matching

Giving the control enough non-sharing training work that an advantage cannot be
explained merely by more optimizer computation. The current gate matches at least
adapter optimizer steps and routed/off-route state examples with extra native
updates. Reports retain optimizer-step counts, state-example counts, scheduled
network evaluations, and worker seconds because no one proxy captures all cost.

### Adapter state example

One routed canonical action target or one off-route preservation position consumed
by an adapter loss. It is the main training-compute unit used to size the control's
extra native updates. State examples are a better matching unit than raw optimizer
steps when sharing and native batches have different sizes.

### Paired sharing/control seeds; paired design

For every representation and attempt index, sharing and control use the same
registered seed schedule, starting checkpoint, evaluation panel, search budget,
action budget, and attempt count. Native-refresh seed streams are paired too.
This removes avoidable Monte Carlo variation from the treatment contrast.

Each corresponding sharing/control episode is a **paired attempt**, as defined
above. A paired seed is the larger block containing all of those attempt pairs
for one experimental seed.

“Paired” does not mean the two learned policies take identical actions or consume
identical wall time. After training diverges, the same random seed is applied to
different distributions. Analyze each seed as a pair: report intersections,
sharing-only and control-only identities, and common-success objective quality
before averaging across seeds.

## Learning paths and diagnostics

### Native RL update

An AlphaZero-style update of the scientist's ordinary policy/value network from
its replay: successful native trajectories provide policy/value targets, MCTS
provides search-improved policy information, and solve/cost heads use their
appropriate positive and negative supervision. In the sharing arm, the option
controller is bypassed during the isolated native update and stale gradients are
cleared before clipping and stepping native-owned parameters.

### Option-policy adapter

A separate residual policy module attached to a scientist. It sees the serial
observation with explicit head-cell and internal-budget features and learns
receiver-native routes for donated semantic edits. Its output projection starts
at zero, so attachment preserves the original policy exactly before training.
The base policy/value network remains frozen during an adapter update.

### Option-policy gate

A state-dependent applicability module that controls when the adapter residual
should influence the native policy. It is trained with route usefulness plus
off-route KL and activation penalties. Its purpose is selective reuse of a
foreign method, not uniform imitation everywhere.

### Adapter versus native optimizer updates

These are different interventions and must be reported separately.

- An **adapter update** changes only option-adapter/gate parameters. Base policy,
  value heads, BatchNorm statistics, and native optimizer state remain fixed.
- A **native update** changes the ordinary scientist from native replay while the
  option controller is bypassed. It does not step adapter/gate parameters.

The no-sharing control receives additional native updates to match sharing
compute. Equal numbers of updates do not imply equal wall time or equal state
examples, hence the separate compute fields.

### Canonical route

A deterministic receiver-native execution of one donated semantic braid edit:
the shortest legal neutral sequence that moves the serial head to a location
where the edit can be performed, followed by that local external edit. The route
is materialized without consulting current policy logits, so its target identity
does not jump when the policy changes.

The current horizon allows at most five internal controller moves. “Canonical”
means reproducible under this translator; it does not mean mathematically unique,
globally cheapest, or the receiver's preferred strategy.

### Canonical-route loss

Teacher-forced negative log-likelihood of the actions on canonical routes. For
route set `R`,

```text
route_loss = mean over r in R of
             sum over (state, action) in r of -log pi(action | state).
```

The implementation averages route sums, not individual action losses, so long
routes contribute larger loss within their route. The v11 macro diagnostic first
measures each witness separately to prevent long solutions from dominating the
cross-witness summary.

Route loss answers: “Does the receiver currently assign probability to this one
translated execution?” It does **not** measure `p(solve)`, eventual unknotting,
final `L10`, or whether the foreign method helps the receiver elsewhere. Total
adapter training can also include route-gate, off-route KL, and gate-suppression
terms, so one optimizer step need not decrease route loss monotonically.

In v11, mean, median, improvement/regression counts, and per-witness changes are
diagnostics. A particular witness may distract the receiver. No fixed percentage
decrease for every witness is a hard admission condition.

### Route learning

**Route learning** is the adapter learning to assign more probability to the
receiver-native canonical routes translated from donated semantic witnesses. Its
usual diagnostic is relative canonical-route-loss reduction:

```text
route_learning = (route_loss_before - route_loss_after) / route_loss_before.
```

A positive value means the donated, teacher-forced routes became more probable;
a negative value means they became less probable. When several witnesses are
used, report the per-witness reductions and their macro mean or median, so long
witnesses do not dominate merely by containing more actions.

Route learning is only an intermediate mechanism check. It does not establish
that the receiver can execute a complete witness closed-loop, solve the
representation under MCTS, improve semantic `L10`, generalize to another
representation, or outperform a compute-matched control. Those require separate
closed-loop route-completion and paired external evaluations. Consequently,
“0.8% route learning” means approximately 0.8% relative reduction in this
teacher-forced loss—not a 0.8% increase in solve rate or sharing effectiveness.

### Policy loss, value loss, and `p(solve)` loss

The **policy loss** teaches which action to take. Cost/value losses predict future
`cc`, moves, or scalar search value. The **solve loss** calibrates the probability
of solving within the declared remaining objective/action/search budget. A failed
episode supplies legitimate negative solve evidence but no successful policy
route to imitate. These losses may share features, so tests must check both
calibration and solver retention after critic training.

### Retention

Preservation of useful behaviour after learning. It must be qualified:

- **solved-set retention**: previously solved representations remain solved;
- **objective retention**: capped loss or best costs do not worsen beyond a
  declared tolerance;
- **policy/output retention**: logits and values remain equal, often used when a
  zero-initialized module is attached;
- **parameter retention**: protected tensors remain bit-for-bit unchanged.

Retention is evaluated on fixed canaries and paired seeds. It is not implied by a
falling training loss.

### Exact retention

In the v6--v11 sharing reports, **exact frozen-solve retention** means

```text
S(frozen starting checkpoint) is a subset of S(trained sharing checkpoint)
```

under the same declared finite evaluation attempts. No frozen solved identity may
be lost. It does not require identical objectives or attempt-by-attempt outcomes.
It is now a secondary stability criterion, not the v11 primary gate.

Do not confuse it with:

- **exact final-set match**: `S(sharing) == S(control)`;
- **bit-for-bit output retention** after a function-preserving migration; or
- retention of every possible solve under unlimited search.

### Canary

A representation solved by the frozen scientist and withheld from sharing targets,
used to detect collateral forgetting. Losing a canary is evidence of drift. It is
reported alongside gains, rather than automatically vetoing a much larger external
objective improvement in v10.

### Current-network portfolio and lifetime solution bank

The **current-network portfolio** evaluates one checkpoint on a frozen set of
representations with the registered simulations, attempts, action horizon, seeds,
and objective cap. It measures what the scientist can solve now. Its solved set
may exchange individual identities during continual learning.

The **lifetime solution bank** stores the best verified semantic solution ever
found for each representation, even if the current network later stops reproducing
that solution. It measures accumulated research output, not current policy recall.
Every report should keep these two quantities separate.

### Capped portfolio objective and block progress

For a frozen portfolio `P` and empirical cap `C`, the capped objective is

```text
sum over x in P of min(best verified L10(x), C), using C when x is unsolved.
```

`C` is fixed before learning from the maximum verified `L10` on the registered
calibration panel; it must not be retuned after observing treatment outcomes.
Both arms use exactly the same portfolio and cap.

A **block-progress** update compares the complete old-plus-seen portfolio before
and after a training block. The block may be retained when total solved count does
not fall and capped portfolio `L10` does not rise; the complete run must show at
least one strict improvement. Losing one canary or temporarily losing a newly
acquired solve is therefore allowed when the complete portfolio still improves.
If the block regresses, targeted recovery is attempted; if recovery fails, the
current network and optimizer return to the block-start state. The lifetime
solution bank remains intact.

### Common-success objective

The sum of each arm's best objectives only over
`S(sharing) intersect S(control)`. It asks whether sharing shortened or lengthened
solutions on tasks both arms can solve. This separates quality from the capped
loss benefit of covering additional tasks.

## Protocol and artifact names

### Gate

A preregistered bounded experiment with a frozen panel, checkpoints, budgets,
seeds, metrics, and pass/fail decision. A gate can fail while still providing
positive evidence about one mechanism. A **smoke test** checks plumbing cheaply;
it is not a gate and must not be promoted into paper evidence.

### Real v6 sharing gate

Informal historical name for the corrected, full three-seed v6 multi-witness
experiment, as opposed to a tiny smoke. It used the 17-representation
`s-tape4-h5` panel, eight training witnesses, held-out `12a_850`, seeds
20260950--20260952, 128 simulations, eight evaluation attempts per
representation, sixteen cycles, and compute-matched native controls.

Sharing solved 13, 17, and 15 representations versus 6, 6, and 4 for control and
won charged capped `L10` in every seed. It nevertheless failed the then-primary
exact-retention rule in two seeds. See
[the v6 record](16-scientists-collaboration.md#corrected-multi-witness-gate-v6-2026-08-06).
“Real” is not a schema field; cite the actual artifact and schema in formal work.

### Latest v7 smoke

An ambiguous, now-historical conversational label. The artifact usually meant is
`multi-witness-gate-v7-preflight-s-tape4-h5-seed20260960-20260806`: a six-item,
two-target, two-cycle plumbing preflight at eight simulations and two evaluation
attempts. It is useful for checking translation, optimizer isolation, and report
fields, but not sharing effectiveness.

It was followed by 17-item **v7 medium** runs at 128 and 64 simulations, so “latest
v7” is no longer unique and no v7 run is the current protocol. Always cite the
artifact directory, manifest schema, seed, training/evaluation simulations, and
attempt count instead of this phrase.

### v8 split-budget smoke

Historical plumbing check that separated learning simulations from final
evaluation simulations in the manifest. It established resumability and field
wiring, not a scientific comparison.

### v9 split-budget gate

Historical protocol that used 64 simulations for learning and 128 for evaluation
and made charged capped loss primary, but still required every active donation
event to achieve a fixed canonical-route-loss reduction. Seed 20260950 solved
14/17 with sharing versus 4/17 for control and won capped `L10` 2,392 to 3,721,
yet failed because one early witness did not reach 10% route-loss reduction.

That per-event veto was conceptually wrong: a foreign method may be locally
distracting, while a collection of ten or more better solutions helps on average.
v10 supersedes it; the v9 outcome must not be relabelled as v10.

### Implemented v10 block-balanced sharing gate

Schema `block-balanced-compute-matched-option-adapter-sharing-v10`. Its main
changes are:

- wait for at least ten distinct active superior donations by default;
- update at scheduled block boundaries rather than after each individual gift;
- draw the same number of route positions from every active witness;
- apply a fixed adapter-step dose and compute-match the control;
- report per-witness canonical-route-loss changes only as diagnostics; and
- decide primarily by paired external semantic capped loss, with at least one
  sharing-only identity required overall.

Exact retention, control-only identities, exact final-set equality,
common-success objective, and route-loss distributions remain mandatory reported
secondary criteria. This protocol implements the hypothesis “learn on average
from a collection of better solutions,” not “every foreign solution must
immediately reduce its own imitation loss.”

The first confirmatory three-seed run did not pass under the historical
native-ply objective. Every active witness's
canonical-route loss improved, but sharing's complete-panel capped-loss delta
`sharing - control` was +708, +780, and -270; its mean was +406 and its median
was +708. Non-target preservation-canary deltas were +431, +18, and -188; those
canaries are not a truly unseen test set. This shows that v10 learned its
teacher-forced routes under that run, but it does **not** decide effectiveness
under the intended semantic `L10`: witness selection, admission, budget features,
and reported objectives all used the wrong move count. See
[the v10 result](16-scientists-collaboration.md#block-balanced-sharing-gate-v10-2026-08-06).

### Semantic-cost v11 sharing gate

Schema `semantic-cost-block-balanced-option-adapter-sharing-v11`. It retains the
v10 block-balanced learner but corrects the scientific contract:

- `moves` is the number of verified portable semantic witness steps;
- translation must preserve donor `(cc, semantic_moves)` exactly;
- `receiver_native_plies` and `receiver_internal_plies` are compute diagnostics,
  not objective terms;
- sharing and control consume the remaining-semantic-`L` input channel;
- a registered generalization subset is excluded from donation, native replay,
  and off-route preservation; and
- the fresh zero-initialized option adapter uses its own `1e-3` optimizer rate,
  independent of the conservative native-controller learning rate.

The local 25-representation preflight had 13 registered targets, 12 routable
donations, six preservation canaries, and six untouched generalization items.
At 16 simulations it solved `10_124` in both arms with capped semantic `L10`
6,387 for sharing and 6,393 for control; no target or untouched item was solved.
At 64 simulations a post-hoc dose sweep added sharing-only `10_100`, but a fresh
64-simulation/four-attempt run tied at one solve and capped loss 6,387 while both
arms lost the frozen `10_100` solve. Therefore v11 is an accounting and plumbing
success, not yet an admitted sharing treatment.

## Architectures

### `s-*`

The `s` prefix means a **serial** formulation: a bounded moving head sees a local
window and must spend native actions to reposition or manipulate memory. It does
not mean “small.” Controller actions consume native episode and internal-action
budgets, but only the resulting semantic braid actions count toward `L_A:B`.

### `s-window-128`

A serial scientist that can act anywhere in a seven-cell visible window. The
`128` denotes the candidate's default MCTS simulations per move, not its hidden
width. Its controller is the function-preserving parent imported by
`s-cyclic-tape8-192`.

### `s-tape4-h5`

An independent serial head controller with a writable four-symbol aligned tape
and an internal-action horizon of five. It exposes remaining internal budget and
is the receiver used by the historical v6--v10 sharing gates. The first v11
semantic-cost preflight instead used budget-calibrated `s-tape4`. `h5` is the
controller horizon, not MCTS depth.

### `s-tape4` and `d-tape4-u1`

`s-tape4` is the independent serial four-symbol-tape scientist used by the
current K=3 collaboration roster. Its promoted stage-18 checkpoint was trained
through native rung reinforcement learning and has no distillation provenance.
It already uses the U1 exploration rule, so there is no separate
`s-tape4-u1` candidate name.

`d-tape4-u1` is a different historical student initialized by behavioural
distillation from the `u1-puct` teacher. The `d` prefix and `-u1` suffix identify
that provenance. Results or checkpoints for these two candidates must not be
silently interchanged even though both expose a four-symbol tape.

### `s-cyclic-tape8-192`

An experimental 306,214-parameter capacity scientist combining:

- the complete imported `s-window-128` local controller;
- a full-word cyclic residual tower of width 64;
- five dilated neighbour blocks at offsets 1, 2, 4, 8, and 16;
- a transported writable eight-symbol tape whose contents move consistently with
  braid shifts;
- seam-free occupied-word mean/max pooling; and
- zero-initialized residual policy, scalar-value, solve, and cost heads.

The fused representation has 192 components: 64 from the local controller and
`2 * 64` from global mean/max pooling. The suffix `192` should therefore not be
read as 192 convolution channels or 192 MCTS simulations. The global pooled
features are rotation-invariant; the local policy remains head-relative.

Zero-initialized residual heads and negligible initial probability for new
tape-write actions allow a parent checkpoint to be imported without immediately
destroying its established outputs. Exact-view contrastive pretraining showed
representational potential, and a local RL gate admitted it only as a K=4
sensitivity scientist—not as a replacement for the preregistered K=3 roster.
See [the capacity result](16-scientists-collaboration.md#experimental-cyclic-memory-scientist)
and [06-network-growth.md](06-network-growth.md).

### `s-strand-graph-128`

The proposed independent replacement for `s-head-128` in the collaboration
roster. Before every decision, a deterministic scan of the closed braid records
four neighbours for each crossing: previous and next crossings along each of its
two physical strands. A width-96, five-block residual graph encoder passes messages
both around the cyclic word and through these strand links. Its edit head acts at
the current head position; its routing head scores every fixed left/right shift
using the encoded position that shift would reach. `128` is the default MCTS
simulations per move, not the hidden width.

The scan is a compiled perception option with `K(x)=len(x)`, rebuilt after every
semantic edit. It consumes compute but is not a charged semantic action and is not
expanded as `K` branches inside MCTS. The candidate is implemented and passes
local wiring/trainability checks, but is not an admitted scientist until a forward
mixed-strand curriculum and a fresh held-out complementarity gate pass. See
[the replacement design](16-scientists-collaboration.md#replacement-third-scientist-s-strand-graph-128-2026-08-08).

### Remaining-`L` budget channel

An input feature giving the network the objective still available:

```text
remaining_L = L_max - L_spent_so_far.
```

It lets policy, values, and `p(solve)` condition on feasibility under the current
cap. It is an input channel—not a separate “remaining-budget head.” A network can
still be evaluated with a generous/no-practical objective cap to test general
unknotting ability.

### Internal-action budget channel

An input feature giving the remaining number of controller-only steps allowed
before the required external edit. It is distinct from remaining `L`: an internal
head shift spends controller horizon and native episode budget, but does not
spend semantic `L`. The two budgets answer different questions.

## Reporting checklist

Before interpreting a sharing number, verify that the report names all of the
following:

1. representation panel and target-witness set;
2. starting checkpoint hashes and architecture names;
3. objective ratio, move accounting, and failure penalty;
4. learning simulations, evaluation simulations, attempts per representation,
   action cap, cycles, and native-refresh attempts;
5. paired seeds and exact sharing/control solved sets;
6. capped loss and common-success objective for both arms;
7. translated, active, stale, and zero-policy-update witness counts;
8. adapter/native optimizer steps, state examples, search evaluations, and wall
   time;
9. retention and control-only identities as secondary safety evidence; and
10. manifest schema and artifact path.

Without those denominators and protocol identifiers, a result is a debugging
observation rather than evidence for or against sharing.
