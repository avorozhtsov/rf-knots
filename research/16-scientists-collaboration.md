# 16 — Scientists collaborating through verified solutions

This is a concrete programme combining all three directions in
[13](13-directions.md): an adaptive schedule, a population arena without network
recombination, and learning from the best verified solution anyone found. The
proposal is strong enough to become a paper experiment, but the first draft of the
loop mixes several quantities that have to be separated before its result can mean
anything.

The short verdict is:

* make comparison of approaches on a calibrated benchmark the primary study;
  treat record discovery as an optional later external-validity test;
* keep the moving frontier of roughly 100 representations;
* keep diverse rung-18 scientists and the common solution archive;
* schedule on calibrated probability of solving **and** conditional cost, not
  conditional cost alone;
* exchange semantic witnesses, not foreign action sequences;
* compare adaptive ordering and solution sharing as separate experimental factors;
* do not widen the networks until a capacity-bound regime is measured; and
* pre-train and calibrate the auxiliary heads on an independent, failure-bearing
  corpus before they are allowed to order tasks.

## 1. Objects and objective

A **knot identity** and a **representation** are different objects. Write

```text
x = (representation_id, knot_id, braid_word, strands, provenance)
```

and never use the braid word as the knot identity. One knot may contribute one to
six representations. Every split, confidence interval, and held-out test must be
grouped by `knot_id`; otherwise one diagram of a knot can train the model while a
second diagram of the same knot pretends to be independent test data.

A successful attempt produces a replayable semantic witness `w` with

```text
cc(w)     number of crossing changes
moves(w)  total charged moves in the author's action space
C10(w)    10 * cc(w) + moves(w).
```

`C10` is a legitimate fixed preference, but it is not lexicographic unknotting
quality: eleven saved moves can compensate for one extra crossing change. The
paper should therefore report `(cc, moves)` and `C10`, with `C10` fixed in advance.
Where exact `u(K)` is known, also report `cc - u(K)`. A learned solution proves an
upper bound; it does not prove that `C10` or even `cc` is optimal.

An unsuccessful fixed-budget attempt has no conditional `cc` or `moves`. Give it
a separate failure outcome and, where the scheduler needs one scalar, a capped
loss `C_fail` larger than every solved cost permitted by the move budget. Never
drop failures from scheduler-head training or from evaluation averages.

### Objective-budget search protocol

Treat an objective cap as part of the task, not as an undocumented MCTS limit.
For `L = A * crossing_changes + B * moves`, every observation may include
`clip((L_max - L_spent) / max(L_max, 1), -1, 1)`. Old checkpoints migrate by
appending a zero-initialised input weight, so enabling the channel initially
preserves their policy and value outputs.

Fresh tasks start with a learned attempt budget, initially
`max(L_floor, 2 * L_predicted)`, and geometrically double it after
`budget_exhausted` until the existing global move-derived cap only on a pinned
audit sample. Routine attempts stop at the first cap: an initial real 30-pair
smoke that restarted every failure increased scheduled inference by 81%, because
unsolved tasks repeatedly replayed doomed prefixes. A deterministic 10% audit
retains calibration and recovery evidence without paying that cost everywhere.
The first cap is not called an upper bound: exhaustion censors the unconstrained
crossing/move targets and must not create an invented equality label.

Shared witnesses create two distinct jobs. Translation keeps the author's
crossing-change count and permits only calibrated receiver-native move slack;
improvement uses the strict incumbent cap `L_best - 1`. After each action the
environment checks success before exhaustion, then terminates an unsolved path
when no objective budget remains. Search may mask an action whose immediate cost
already exceeds the remainder. `budget_exhausted` is persisted separately from
ordinary failure so replay, resume, evaluation and MCTS agree.

This optimisation is admitted to the collaboration experiment only if paired
regression recovers every baseline solve after restarts, witness replay and
resume remain exact, and network evaluations or wall time fall by at least 20%.
Every arm then receives exactly the same budget protocol.

The corrected 20-task, three-scientist ratio-10 gate at 16 simulations contained
60 paired attempts. Hard caps reduced scheduled network evaluations from 54,519
to 37,434 (31.3%) and measured wall time from 52.7 to 37.9 seconds. They censored
two individual baseline solves and one portfolio task, which is why exhaustion
is not an intrinsic failure label. Paired geometric audits at the global cap
recovered every baseline solve. A tape-channel ordering defect found during this
gate is fixed: the budget plane is appended after transported tape channels, and
all three roster networks now have exact function-preserving checkpoint tests.
The process-parallel runner also resumed exactly from round one to round two and
its held-out evaluator consumed the migrated checkpoint. The optimisation passes
the local admission gate with deterministic 10% recovery audits enabled equally
in every arm.

## 2. The 2,700-representation bank is a construction task

The current bundled table contains 2,870 knot identities, but only 1,639 of its
stored braids fit the present common envelope of at most five strands and 48
letters. It stores one representative braid per identity. More importantly,
`data/unknotting_numbers.json` currently imports exact `u` for only 17 compatible
table knots.

Consequently the requested bank cannot yet be made by taking 2,700 table rows and
sorting them by known `u`. With at most six representations per knot, 2,700 rows
require at least 450 distinct knot identities with exact `u`; the current imported
17 can supply at most 102 rows under that rule.

Build `BASE` as a pinned artifact rather than synthesising it inside the runner:

1. Audit the pinned KnotInfo snapshot and import every exact unknotting number that
   is actually supported there, retaining source URL, retrieval date, snapshot
   hash, and whether the published value is exact or only an interval.
2. Select knot identities before representations, enforcing the common strand,
   word-length, and move-budget envelope.
3. Generate one to six deterministic representations per identity using only
   verified isotopy/Markov rewrites, never crossing changes. Store generation seed
   and replayable generation path.
4. Verify every representation against its source identity, reject duplicate words
   and rotations, and cap each identity's weight so six easy diagrams do not count
   as six independent knots.
5. Freeze `BASE.json`, its SHA-256, and a manifest of the source snapshots before
   any experimental arm runs.

If the exact-`u` audit yields fewer than 450 compatible identities, either reduce
`N`, allow more than six representations per identity and admit the resulting
concentration, or stop requiring exact `u` for the main bank. Do not silently use a
lower bound or a missing database value as if it were exact `u`.

### Static ordering

For the exact-`u` subset, the proposed static score

```text
S_oracle(x) = 10 * u(knot_id) + len(braid_word)
```

is a useful privileged baseline. Here `len(braid_word)` is the number of crossings
in this representation, not the knot's minimal crossing number.

The main non-neural baseline should also have a score available for every bank
item, for example a frozen rank model over cheap features such as word length,
strand count, shallow simplification length, determinant, and certified lower
bounds. Fit its coefficients on a separate development corpus and then freeze
them. Evaluate correlation with realized `best_C10` by Spearman or Kendall rank
correlation and top-window recall. AUC is appropriate only after turning the target
into a binary event such as “solved within budget”; it is not a correlation measure
for continuous cost.

Keep both baselines:

* `S_oracle`, using exact `u`, asks whether a network can beat a strong privileged
  curriculum;
* `S_cheap`, using universally available cheap features, asks whether the adaptive
  scheduler earns its inference cost in a deployable setting.

## 3. Which rung-18 scientists

All five proposed candidates have `stage18-after.pt` snapshots in the finalized
rung-18 artifact. Their networks are small but not identical:

| scientist | distinctive state/model | parameters | search uses auxiliary critic? | rung-18 `10:1` solve rate / conditional `C10` |
|---|---|---:|---|---:|
| `s-window-128` | acts anywhere in a seven-cell window | 102,439 | no | 1.00 / 67.92 |
| `s-paint4` | head-only, four transported strand colours | 109,546 | no | 1.00 / 44.33 |
| `d-tape4-u1` | writable four-symbol tape, distilled from `u1-puct` | 105,829 | yes | 1.00 / 79.50 |
| `s-scan-gru` | forced full-word scan with GRU-128 | 177,831 | yes | 0.92 / 40.27 |
| `s-w11-128` | acts anywhere in an eleven-cell window | 102,439 | no | 1.00 / 34.08 |

These rung-18 numbers describe one representation and are not a roster ranking.
Choose the population by paired complementarity on a frozen development set:
measure per-representation successes, costs, and pairwise failure overlap, then
add the scientist with the largest marginal portfolio coverage. Do not choose five
arms merely because five names exist.

A reasonable provisional roster is:

* `K=4`: `s-window-128`, `s-paint4`, `d-tape4-u1`, `s-scan-gru`;
* `K=5`: add `s-w11-128` as the wider-local-view arm.

The fifth arm is likely the most redundant with `s-window-128`, which makes it a
useful test of whether an extra scientist adds coverage or only compute. Because
`d-tape4-u1` already inherited data from `u1-puct`, report a sensitivity roster
without it; otherwise “independent scientists” would be an inaccurate description.

Pin the five checkpoint paths, SHA-256 hashes, candidate specifications,
observation schemas, and search budgets in the experiment manifest. Use the exact
rung-18 snapshots for all five even though later snapshots exist, so curriculum
exposure is not silently unequal.

### 2026-08-02 challenge diagnostic

A deterministic low-budget check tested the five proposed families plus
`d-gru128-u1` on ten held-out continuation representations: the depth-0 and
depth-4 versions of `R(3,22)#0`, `R(5,22)#0`, `R(3,24)#0`, `R(5,24)#0`, and
`R(5,26)#0`. Each checkpoint received 128 simulations with no root noise at
ratios 10 and 1000. This is an engineering diagnostic, not a benchmark result.
It used the latest available continuation snapshot for each family
(`s-window-128` stage 22; `s-paint4` and `s-scan-gru` stage 18;
`d-tape4-u1` and `d-gru128-u1` stage 21; `s-w11-128` stage 19), so it is a
capacity/coverage probe rather than a curriculum-matched comparison.

| scientist | ratio 10 solved | conditional mean `C10` | ratio 1000 solved | conditional mean `C10` |
|---|---:|---:|---:|---:|
| `s-window-128` | 4/10 | 83.5 | 3/10 | 106.7 |
| `d-tape4-u1` | 3/10 | 141.7 | 3/10 | 142.7 |
| `s-w11-128` | 2/10 | 93.0 | 2/10 | 81.0 |
| `s-paint4` | 1/10 | 91.0 | 1/10 | 75.0 |
| `s-scan-gru` | 0/10 | - | 0/10 | - |
| `d-gru128-u1` | 0/10 | - | 0/10 | - |

The proposed five-scientist portfolio covered 4/10 representations at each
ratio. All four were the 3-strand representations; no checkpoint solved a
5-strand representation at this budget. At ratio 10, `s-window-128` supplied all
portfolio-best costs. At ratio 1000, the four best costs were split across four
architectures and `d-tape4-u1` uniquely supplied one success. Adding
`d-gru128-u1` changed neither coverage nor portfolio cost.

This result argues against parameter count as the first intervention: the largest
tested network, `s-scan-gru`, was not the strongest. It does show a floor effect
on these 5-strand cases at 128 simulations. The comparison pilot should therefore
use a newly assembled, frozen 200-representation set spanning easy, intermediate,
and hard cases, rather than simply reusing an old 200-item stream. Aim for roughly
20--80% initial portfolio coverage and stratify by strand count and cheap
difficulty features. Keep the hard 5-strand cases as a stress stratum, not the
whole benchmark.

## 4. The scheduler head must be calibrated first

The present auxiliary head already emits a four-member ensemble of

```text
p_solve, predicted crossing changes conditional on solving,
predicted moves conditional on solving.
```

All five rung-18 snapshots contain these head parameters. For
`s-window-128`, `s-paint4`, and `s-w11-128`, however, search still uses the legacy
scalar critic. More importantly, the independent five-seed test in
`pgx-mcts-bench/artifacts/factorized-critic-five-seed-r21-20260802` rejected the
current factorized critic as a search replacement: it solved fewer held-out cases
and had worse objective values at every tested ratio. The recorded likely defect
is inadequate training of `p_solve`, especially missing failed/censored outcomes.

Therefore pre-train the scheduling heads, but do not alter the policy trunks yet:

1. Build an offline corpus disjoint by `knot_id` from `BASE` and the final anchors.
   Include solved, failed, and capped attempts at the exact search and move budgets
   the collaboration experiment will use.
2. Freeze each policy encoder and train only its auxiliary ensemble. The current
   configuration already supports detached auxiliary training.
3. Calibrate `p_solve` per scientist and budget on a held-out calibration split.
   Fit conditional cost only on solved attempts, while retaining failures for the
   solve head.
4. Measure Brier score/calibration error for solve probability, MAE for conditional
   cost, and Spearman/Kendall ranking against realized capped cost.
5. Run the scheduler offline over historical attempts. If it cannot retrieve easy
   tasks better than `S_cheap`, the adaptive-ordering branch stops before RL.

For a representation `x`, use an expected capped loss such as

```text
q_i(x) = p_i(x) * (10 * cc_i(x) + moves_i(x))
         + (1 - p_i(x)) * C_fail.
```

Use a conservative ensemble quantile rather than an optimistic mean if calibration
shows substantial uncertainty. Sorting only by predicted conditional `C10` would
systematically prefer impossible tasks whose cost head happens to be small.

## 5. Corrected collaboration loop

Let `H=100` be the active frontier width. Keep a reveal cursor `r`, a processed
round counter `t`, and the active set `A`. This removes the original off-by-one
ambiguity around `BASE[:j+1]`.

```text
A = BASE[0:H]
r = H
t = 0

while A is not empty or r < len(BASE):
    if r < len(BASE):
        add BASE[r] to A
        r += 1

    each scientist proposes its lowest-q task in A
    each proposal receives the same fixed qualification search budget
    choose X by the smallest verified capped realized cost among proposals
    every scientist that has not yet attempted X receives the full task budget
    verify and store all attempts; select the best witness for X
    remove X from every task index
    t += 1

    after every 10 processed tasks: train on the new block
    after every 25 processed tasks: run the stale-solution improvement round
```

If every qualification attempt fails, select the proposal with the smallest
realized capped loss, let all scientists try it, and record either a winner or an
unresolved task. An unresolved task still supplies failure labels and moves to the
stale-task queue; it must not deadlock the frontier or disappear from the report.

### Task indexes, not magical heaps

Training changes every prediction, so all heap keys become stale after an update.
With only about 100 active items, a batched rescore and heap rebuild is cheaper and
clearer than a complicated decrease-key implementation. If an indexed priority
queue is still useful, represent it as:

```text
task_state[scientist_id][representation_id] =
    (priority, model_version, status)

heap entry = (priority, model_version, representation_id)
```

Arbitrary removal increments or tombstones the indexed version; `pop_min` discards
stale entries lazily. After any network update, increment `model_version`, batch
score the active set, and rebuild. Representation ID, never display name, is the
key.

### Attempt and solution store

The common store should be an append-only evidence log plus indexed projections,
not one heap overloaded with several jobs:

```text
Attempt = (round, representation_id, scientist_id, budget, seed,
           solved, cc, moves, witness_id, checkpoint_hash)

SolutionRecord = (representation_id, knot_id, attempts_by_scientist,
                  best_witness_id, best_C10, author_id,
                  first_solved_round, last_improved_round)
```

Maintain a reporting heap keyed by `best_C10` if useful. Improvement scheduling
needs a different, scientist-specific priority based on expected improvement; the
globally cheapest solutions are usually the least interesting ones to revisit.
Every witness must replay successfully under `rf-knots` reference semantics before
it can become `best_witness_id`.

## 6. Learning from a winner with heterogeneous action spaces

The proposed roster cannot copy raw replay records between scientists.
`s-window-128` can act at seven visible positions, `s-w11-128` at eleven, and the
head/tape/paint/scan agents have different internal actions and state. A foreign
action index or policy vector has no meaning in the receiver's action space.

Store the winning solution as a semantic word-state path

```text
w0 -> w1 -> ... -> wm
```

plus its verified crossing-change decisions. For each receiver:

* train the cost head with the witness as a one-sided upper bound;
* derive receiver-native routes between semantic states, charging all head, tape,
  colour, and scan actions;
* create policy targets only from receiver-native reanalysis or a certified local
  option that reaches the next semantic state; and
* fall back to value-only sharing when no route exists within the receiver's move
  budget.

This translator/reanalysis layer is a prerequisite for the experiment. The current
`braid-adaptive-scientists` runner only shares records among parallel agents with a
common action space and cannot run this roster correctly.

### Train every scientist, including the author

Do not exclude the author. The author already receives its own search-improved
record once through ordinary RL; do not duplicate it as a shared record. Every
peer receives the translated/reanalysed winner. This yields symmetric learning
without overweighting the author.

After every ten processed tasks, train on the **ten newly processed tasks**, not
the ten globally lowest-cost solutions. Repeatedly choosing the cheapest archive
entries would train almost entirely on low-`u` knots. Use a replay mixture with
explicit weights for own RL, new shared witnesses, older replay, and failure
examples, and log the sampled composition.

## 7. Improvement rounds

Every 25 processed tasks, consider records whose `last_improved_round < t - 20`.
For scientist `i`, rank them by calibrated expected improvement, for example

```text
EI_i(x) = p_i(x) * max(0, best_C10(x) - predicted_C10_i(x)).
```

Let each scientist attempt its top five eligible tasks under the same fixed budget.
An improvement is accepted only after witness verification. Append the event,
update `best_C10`, `author_id`, and `last_improved_round`, and broadcast the new
witness to the other scientists at the next training boundary.

Also retain a small random stale-task sample. Otherwise a miscalibrated head can
permanently hide exactly the tasks on which it underestimates its ability. Log
attempts that fail to beat the incumbent; improvement rate is otherwise inflated.

## 8. Experiments that identify the cause

The proposed paper is not one comparison. Adaptive ordering, population search,
RL, and sharing are separate causes. The minimum useful matrix is:

| arm | scientists | order | learning | shared winners |
|---|---:|---|---|---|
| `P-A-S` | K | adaptive within frontier | RL | yes |
| `P-A-NS` | K | adaptive within frontier | RL | no |
| `P-S-S` | K | frozen `S_cheap` or `S_oracle` | RL | yes |
| `1-A-RL` | 1 | adaptive | RL | n/a |
| `1-S-RL` | 1 | frozen static | RL | n/a |
| `1-S-BC` | 1 | same frozen static stream | offline best-witness training only | n/a |

`P-A-NS` is essential: without it, a positive result cannot say whether the gain
came from ordering or from witness exchange. Add `P-S-NS` if compute permits; it
completes the population `adaptive/static x sharing/no-sharing` factorial.

For `1-S-BC`, freeze a witness corpus before training and use receiver-native
targets only. If the corpus is generated online by the adaptive population, the
baseline has already inherited the method it is supposed to test.

### Compute matching

A K-scientist portfolio spends roughly K times the inference and search compute of
one scientist, plus collaboration searches. Report both:

* equal per-scientist budget, which measures the product as proposed; and
* equal total network evaluations, where the single scientist receives the
  portfolio's total search allowance.

Match qualification budgets, full-task budgets, move caps, training examples, and
optimizer updates. Simulations alone are not a perfect compute unit because the
architectures and action spaces differ; record network evaluations and wall time as
well.

## 9. Evaluation and paper claims

Freeze an external anchor set by knot identity before the experiment. No
representation of an anchor knot may appear in auxiliary-head pre-training,
`BASE`, proxy fitting, or witness-corpus construction.

Primary outcomes should include:

* held-out solve rate under a fixed search and move budget;
* conditional `cc`, `moves`, and `C10`, always shown beside solve rate;
* `cc-u` where exact `u` is independently sourced;
* capped expected cost including failures;
* cumulative number of verified solutions and improvements versus network
  evaluations;
* portfolio coverage and each scientist's marginal contribution; and
* scheduler calibration and rank regret against subsequently realized outcomes.

For a scalar primary endpoint, use final held-out **capped portfolio loss** on a
frozen `NEW_70` identity-disjoint set:

```text
Loss10(x) = min_i C10(i, x)                  if any scientist solves x
            C_fail                          otherwise

Primary = sum_{x in NEW_70} Loss10(x).
```

The uncapped `sum_x min_i C10(i,x)` over only solved representations is not a
valid primary measure: an arm can improve it by failing on difficult items. On
the 200-item training/development bank, report the same capped quantity over time
as an area under the learning curve, but do not use its final value as the main
generalisation claim. Report solve rate and conditional costs beside every capped
score. As a predeclared sensitivity analysis, replace `C10` by crossing changes
and use a crossing-change failure cap; do not select between these metrics after
seeing results.

Report both per-representation and knot-clustered estimates. Pair arms on bank,
initial checkpoint, generation seed, search seed, and compute budget. The project
has already produced misleading one-seed conclusions; use a small pilot only for
engineering, then preregister the full seed count and primary endpoint before the
2,700-item run. Eight seeds is the existing project standard for a scientific
claim, subject to a compute estimate made from the pilot.

The strongest defensible primary paper claim would be:

> Under matched compute, calibrated local adaptive scheduling and verified
> cross-scientist witness exchange improve held-out knot-level solve rate or
> cost over static curricula and single-scientist learning.

Do not claim new unknotting numbers from a low `C10` alone. Publish every witness,
its author and checkpoint hashes, and independent lower-bound provenance.
Discovery of a new upper bound may be reported if it happens, but it is neither a
success criterion nor part of the primary endpoint.

## 10. Are the networks deep enough?

There is no present evidence that parameter count is the bottleneck. The earlier
ladder comparison found that increasing the parallel network from about 48K to
372K parameters bought no measurable advantage over doubling search, while
starving search was fatal. The proposed serial scientists already range from about
102K to 178K parameters, roughly 50K of which belong to the four-member auxiliary
head.

For the windowed serial networks, two residual blocks already cover the complete
seven- or eleven-cell observation. More depth does not give them more visible braid
state. Their real representational differences are the window, transported colours,
aligned tape, and full-word GRU scan. “Enough bits” is therefore not well modelled
by parameter count: the state interface and move budget may bind before weight
capacity does.

Do **not** double the networks in the primary experiment. Horizontal concatenation
of two copies is not automatically function-preserving and would confound the
collaboration result with a growth intervention. First run a capacity gate:

1. compare training and knot-held-out auxiliary/policy loss;
2. compare extra MCTS budget with extra width at matched compute;
3. test whether performance improves when the replay corpus grows while the model
   is fixed; and
4. inspect whether failures are shared across architectures or disappear with
   search.

Only if validation loss plateaus above training loss and width beats search should
capacity growth enter a separate arm. Then use the function-preserving operators in
[06](06-network-growth.md): Net2Wider-style channel duplication with divided outgoing
weights, a ReZero residual append, or a frozen new column with adapters. Verify
logits and values are unchanged immediately after growth. Preserve the replay store;
it is more valuable than the surgery.

## 11. Execution gates

Do not commit the full 2,700-item budget at once.

1. **Corpus gate:** prove that the exact-`u`/representation constraints can produce
   the promised bank without identity leakage.
2. **Head gate:** on frozen historical attempts, each scheduling head must beat
   `S_cheap` in calibrated easy-task retrieval.
3. **Translation gate:** replay 100 witnesses through every ordered sender/receiver
   pair; report coverage and charged move inflation.
4. **Complementarity gate:** choose K from paired failures and marginal coverage,
   not ladder rank.
5. **Pilot:** 200 representations, three engineering seeds, fully resumable event
   store, no scientific claim.
6. **Full run:** freeze protocol, seed count, bank hash, checkpoints, primary
   endpoint, and compute budget before starting.

The run must be resumable after every attempt and training boundary. Heaps are
derived state and should be reconstructible from the append-only log. The existing
adaptive runner refuses an output directory once `schedule.jsonl` exists; that is
not adequate for a 2,700-round paper experiment.

### Implemented execution sequence, 2026-08-02

The corrected sequence is now:

1. run the legacy implementation for 10--20 tasks as a regression smoke;
2. build and gate the heterogeneous, resumable machinery;
3. run a new, stratified 200-representation four-arm factorial pilot;
4. admit an experimental higher-capacity scientist only after identity-disjoint
   value/solve gates; and
5. proceed to 2,700 only if scheduling, translation, sharing, resumability,
   held-out generalisation, and compute matching all pass.

Step 1 exposed a real interface regression: the fixed-knot wrapper did not forward
the newly added value-potential method. After repair, the unchanged legacy runner
completed 12/12 tasks, wrote 36 checkpoints, and shared a verified winner on eight
tasks.

The suspected old 200-knot preparation was not stranded on an unmerged branch.
Commit `33a4bbc` (`Advance adaptive knot research training`) introduced the legacy
runner and its documented 200-round command on local branch
`codex/adaptive-knot-research`; that commit is an ancestor of current `main` and
`origin/main`. It did not include a dedicated 200-round shell launcher. The runner
was present but had not been exercised against the later game interface.

The new `braid-collaborative-scientists` runner freezes checkpoint and bank hashes,
keeps `BASE` and `NEW_70` identity-disjoint, evaluates ratios 10 and 1000 together,
and gives adaptive and static arms equal qualification/full-search attempt counts.
Every round is an immutable transaction containing the event and complete network,
optimizer, replay, frontier, and cursor state. `schedule.jsonl` is derived state.
A changed protocol or checkpoint hash is rejected on resume.

The first local pilot exposed a storage gate: uncompressed cumulative replay
snapshots occupied 7.0 GiB by roughly round 88 across three arms. Future runs
therefore write the same atomic state as `state.pt.gz`; a measured 50 MiB round
compressed to 8.3 MiB at the fast compression level. Readers remain compatible
with the original uncompressed pilot. Storage consumption must be projected and
provisioned before CPU-32 and checked again before 2,700.

For heterogeneous sharing, a winner is replayed into global semantic braid edits.
Each receiver then routes those edits through its native head and memory actions.
Only a receiver-native trajectory that independently replays to the unknot within
the receiver's budget enters replay. The 13-round integration smoke obtained a
winner on six tasks and verified all 22/22 attempted translations. A subsequent
real resume committed round 13 without repeating earlier work.

The local pilot uses the complete population factorial:

| arm | order | shared winners |
|---|---|---|
| `adaptive-sharing` | auxiliary expected capped loss | yes |
| `adaptive-no-sharing` | auxiliary expected capped loss | no |
| `static-sharing` | frozen cheap score | yes |
| `static-no-sharing` | frozen cheap score | no |

All four use the same `BASE`, `NEW_70`, initial checkpoints, objective ratios,
search counts, and training boundaries. The run is an engineering seed; it does
not replace the preregistered multi-seed CPU-32 experiment.

The frozen 200-bank is not an unknotting-number-one bank. Independent KnotInfo
lower-bound data certify `u >= 2` for 87/200 identities: 58 have lower bound 2,
14 have 3, 14 have 4, and one has 5. Only two BASE rows are in the repository's
small published-exact-`u` snapshot (`7_5`, `u=2`; `10_124`, `u=4`), so most rows
must not be described as exact-`u` examples. `NEW_70` similarly contains 27/70
identities with certified lower bound at least 2 and includes `8_19`, exact
`u=3`. This makes the bank a valid mixed-difficulty engineering and causal pilot,
but not the final paper bank promised in Section 2: future bank artifacts persist
both exact `u` and certified lower-bound provenance per identity, and the final
static score may use `10*u+intersections` only where exact `u` is actually known.

The completed low-search engineering seed did **not** pass the efficacy gate. On
`NEW_70` at 16 simulations, the initial portfolio solved 5/70 at ratio 10 and
6/70 at ratio 1000. Adaptive sharing finished at 5/70 and 5/70; adaptive without
sharing at 6/70 and 4/70; static sharing at 6/70 and 6/70. Ratio-10 capped losses
were 17,449 initially, 17,515 for adaptive sharing, 17,284 without sharing, and
17,338 for static sharing. This is one engineering seed, not an ordering claim,
but it is enough to block immediate cloud rental.

The productivity trace explains the next gate: every arm found 27 winning tasks,
but the last winner occurred at round 60 for adaptive sharing, 53 without
sharing, and 36 for static sharing. After round 50 the adaptive-sharing
qualification searches produced only two solved attempts across all three
scientists. Continuing to train on roughly 140 failure-only rounds can damage the
critics and is not evidence for a useful adaptive curriculum. Run a 75-round local
gate at the exact CPU parameters (16 qualification and 128 full simulations)
before renting CPU-32. If higher search does not extend the productive frontier,
add a no-update-on-unproductive-round rule or an easier representation supply and
repeat the local gate; do not merely scale the failed schedule.

The 75-round high-search gate used the CPU parameters locally: 16 qualification
and 128 full simulations, 32 training steps, and the same frozen bank. It found
24 winner rounds with adaptive sharing, 26 without sharing, and 27 with static
sharing; every one of 196 translations replayed exactly. On `NEW_70` at 16
evaluation simulations, however, adaptive sharing merely matched the initial
ratio-10 result (5/70, capped loss 17,449) and worsened the ratio-1000 capped loss
from 1,314,177 to 1,325,147. Adaptive-no-sharing was slightly better than
adaptive-sharing at both objectives. More search establishes solver potential,
but does not rescue indiscriminate sharing.

The paired task identities make this negative result stronger than the counts
alone. All 24 training tasks won by adaptive sharing were also won without
sharing; no-sharing additionally won `11a_15` and `11n_119`. On held-out
`NEW_70` at ratio 10, the five adaptive-sharing solves were again a subset of
the six no-sharing solves, whose additional task was `12n_684`. At ratio 1000
the two arms solved exactly the same seven tasks, so their different capped
losses reflect solution quality or failure costs rather than coverage. Thus the
observed advantage of no-sharing is not an artefact of equal-sized but different
solved sets, and this single engineering seed contains no task solved uniquely
by adaptive sharing relative to its paired no-sharing control.

Inspection exposed a protocol error relative to the intended idea: the runner
inserted every translated population winner even when the receiver's own attempt
was equal or better. The corrected rule logs routing separately and admits a
shared trajectory only if its receiver-native objective is a strict improvement,
or if it rescues a receiver failure. A smoke routed 8/8 witnesses but admitted
only 6; the paired 75-round rerun routed 102/102 and admitted 90. Training-bank
winner rounds increased from 24 to 27.

Strict admission improved held-out ratio-10 coverage from 5/70 initially to 7/70
and capped loss from 17,449 to 17,316. It still lost to static sharing (6/70,
17,248) and adaptive-no-sharing (6/70, 17,281). At ratio 1000 it solved 6/70 and
scored 1,324,180, worse than the initial 1,314,177 and essentially tied with
adaptive-no-sharing. The identity comparison is nevertheless informative: at
ratio 10 strict sharing solved all six no-sharing tasks plus `12a_819`, whereas
at ratio 1000 it missed `12a_819`, which no-sharing solved. Strict sharing can
therefore change coverage, but this seed shows opposite effects at the two
objectives. Therefore CPU-32 remains blocked.

The result motivated a separation of **shared information** from **shared policy
imitation**. A translated witness is valid evidence that the state is solvable and
that its costs are upper bounds; it is not automatically a good policy target for
a receiver with different memory actions. The resulting
`adaptive-sharing-aux-only` arm masks shared positions
from policy loss and scalar-value equality loss while retaining solve supervision
and one-sided crossing/move bounds. In 75 paired rounds it found 26 winner tasks,
routed 102/102 witnesses, and admitted 94 strict improvements. On `NEW_70`, its
ratio-10 result was 6/70 with capped loss 17,259: better than adaptive-no-sharing
(17,281) and all full-trajectory sharing variants, but still 11 points worse than
static sharing (17,248). Its six-task set was not the no-sharing six-task set:
aux-only gained `12a_819` but lost `12n_684`. At ratio 1000 it regressed to 5/70
and capped loss 1,331,233. It therefore also fails the CPU-32 gate.

This changes the answer about simultaneous `10:1` and `1000:1` training. It was a
reasonable multi-task proposal, but the pilot now shows objective interference as
a live risk: the sharing rule that helps ratio 10 harms ratio 1000. The next
experiment should be ratio-10-only, because approach comparison rather than new
unknotting records is the primary goal. Treat ratio 1000 as a separate secondary
arm, or give it a ratio-specific policy/value adapter and replay stratum; do not
let it veto or distort the primary curriculum. A fair ratio-10-only gate must
rerun all three controls, not compare a single-task treatment against the existing
dual-task controls.

The first causal pilot deliberately disables the stale-solution improvement
round. Improvement search is another intervention and would obscure whether the
three-arm difference came from scheduling or sharing. After those two gates pass,
add improvement as a fourth factorial comparison and require calibrated expected
improvement plus a random stale-task audit before the 2,700 run.

### Experimental cyclic-memory scientist

`s-cyclic-tape8-192` is a 306,214-parameter capacity probe, not a wider copy of the
existing trunk. It imports the complete `s-window-128` local controller and adds a
five-scale cyclic residual encoder over the occupied full word plus a transported
eight-symbol tape. Seam-free cyclic pooling gives the global value representation
exact rotation invariance. New policy, value, and auxiliary paths are
zero-initialized; new tape-write actions begin with negligible probability.

At 32 simulations, the initialized child preserved its parent's 1/10 challenge
coverage at each objective. A short 40-task run improved challenge coverage to
2/10, but failed the more important identity-disjoint gate: ratio-10 coverage on
`NEW_20` fell from 3/20 to 2/20, while ratio-1000 coverage stayed 3/20 with worse
capped loss. That checkpoint is rejected from the main roster.

To test whether the global capacity can learn representation-independent features,
the next initialization used exact isotopy/Markov views of 400 identities, with
the pilot 200, `NEW_70`, and 50 calibration identities excluded. Contrastive
equivalence pretraining improved top-1 retrieval of a second representation among
the 50 unseen identities from 60% to 92%; mean different-knot cosine similarity
fell from 0.982 to 0.093. This establishes representational potential, not
unknotting performance. The subsequent paired 40-task RL run also passed the
untouched-anchor gate: coverage increased from 3/20 to 4/20 at both objectives;
ratio-10 capped loss fell from 4,653 to 4,462 (4.1%), and ratio-1000 capped loss
fell from 350,119 to 336,068 (4.0%). Admit this checkpoint only as a K=4
sensitivity scientist. Do not change the preregistered K=3 comparison after its
pilot has begun.

The CPU-32 launcher runs three paired seeds and three arms with three attempt
workers per arm, for 27 search workers. It has a 70-hour workload timeout and
transactional resume. At the current
[listed non-GPU rate](https://docs.nebius.com/compute/resources/pricing), a
32-vCPU/128-GiB VM is about `$0.7936/hour`, or `$55.55` for 70 hours before
storage and tax. A 200-GiB network SSD adds about `$1.36` for the same 70 hours at
the listed `$0.071/GiB-month` rate, making the projected subtotal about `$56.91`
before tax. The VM must still be stopped after the workload timeout to stop
compute billing; the disk continues billing until deleted.

The launcher defaults explicitly to the K=3 roster. The admitted cyclic-memory
model requires `ROSTER=k4` and a separate artifact root, so it cannot silently
change or resume into the primary comparison. Run that sensitivity only after the
K=3 gates, or reduce concurrency so a 32-vCPU host is not oversubscribed.

### Corrected ratio-10 four-arm pilot, 2026-08-03

The corrected budget-aware pilot completed 200 rounds for the full
`adaptive/static x sharing/no-sharing` population factorial. It used the K=3
roster, the same BASE and NEW70, four qualification and 16 full-search simulations,
and a ratio-10 objective only. This is a low-search engineering seed, not paper
evidence. On the frozen NEW70 evaluation the results were:

| arm | solved | capped `C10` loss | scheduled network evaluations |
|---|---:|---:|---:|
| initial | 5/70 | 17,449 | 218,994 |
| adaptive sharing | 7/70 | 17,091 | 219,181 |
| adaptive no-sharing | 6/70 | 17,303 | 219,657 |
| static sharing | 7/70 | **17,069** | 218,858 |
| static no-sharing | 6/70 | 17,272 | 217,141 |

The exact solved sets, rather than only their sizes, determine the causal reading.
At adaptive order, the six no-sharing successes were all shared successes and
sharing uniquely added `12a_819`; on the six common successes it also reduced
total objective by 22. At static order the pattern repeated: both arms shared the
same six successes, sharing uniquely added `12a_819`, and reduced common-success
objective by 13. Thus sharing has a positive signal in this seed under both
schedules.

Adaptive ordering did not help. With sharing, adaptive and static solved exactly
the same seven tasks, but adaptive had 22 more capped-loss points. Without sharing,
both solved exactly the same six tasks, but adaptive had 31 more points and 2,516
more scheduled evaluations. Static sharing is the best arm in this run. Therefore
the local gate passes only for a multi-seed confirmation of **sharing**; it fails
for an adaptive-scheduling claim and does not authorize a 2,700-item run. A CPU-32
run, if explicitly approved, should repeat the four-arm ratio-10 experiment over
three paired seeds and treat ordering as a possibly negative intervention. No
cloud resources were rented for this engineering run.

### Matched distillation-degradation gate, 2026-08-03

The poor final BASE200 retention required a direct intervention test. At the
round-48 static-sharing transaction, before the next scheduled update, each
scientist had at least ten admitted receiver-native witnesses. The checkpoint,
optimizer, and replay were forked under three minibatch seeds into no-update,
native-RL, one-witness, and ten-witness treatments. Witness arms used either full
policy imitation or auxiliary-only solve/upper-bound supervision. Every trained
fork received eight optimizer steps with batch size 32. Treatment batches replaced
three positions from the matched native batch; shared episodes were sampled
uniformly before positions so trajectory length did not determine weight.

At 16 evaluation simulations, native RL alone was worse than the untouched
checkpoint in all three seeds. BASE50 capped-loss deltas were +66, +399, and +203,
with zero, two, and one lost portfolio solves. NEW70 deltas were +22, +55, and +22,
with unchanged coverage. Thus the current native update is itself unsafe; sharing
is not the sole cause of the long-run regression.

One auxiliary-only witness was the best sharing treatment on transfer: relative
to its seed-matched native-RL control, NEW70 capped loss improved by 11, 11, and 16
with identical portfolio coverage. It nevertheless failed retention: BASE50
deltas were +157, -65, and +175, and two seeds lost one solve. Ten auxiliary-only
witnesses were clearly worse, losing `12n_684` on NEW70 in two seeds and producing
median BASE50 loss delta +167. Full-policy variants also failed the retention gate.
No distillation treatment passed the preregistered requirement of no coverage or
capped-loss regression on both splits in every seed.

**Decision:** stop the current RL and distillation update rule; keep CPU-32 and the
2,700 run blocked. Do not merely change ten shared witnesses to one and continue.
The next local gate should test a rollback-guarded candidate with one
auxiliary-only witness, one of 32 batch slots, success-balanced native rehearsal,
four optimizer steps, and learning rate 0.00025 instead of 0.001. Evaluate a
frozen BASE retention canary per scientist and keep the pre-update checkpoint when
the candidate is inferior. NEW70 must not participate in online selection or
rollback; it remains the final transfer endpoint.

### Corrected budget, held-out identities, and 128-simulation result

The follow-up found that the budget input had encoded
`remaining_budget / episode_cap`, so every fresh episode presented `1.0`
regardless of its absolute cap. Objective-cap exhaustion was also stored as an
ordinary failed policy/value target. Both are now corrected: budget is scaled by
the fixed global cap, capped trajectories are censored from policy/value
training, and collaboration replay samples episodes uniformly with balanced
native successes and a hard shared fraction.

An explicit-source audit identified `T(3,5)` as BASE item `10_124` and `T(3,4)`
as NEW70 item `8_19`. The corrected NEW70 replaces `8_19` with same-quartile
`12n_683`. The bundled table cannot identify `R(5,12)#0`, although its exact
braid word is absent, so this is disjoint from all *identified* ladder sources,
not a proof against every possible identity collision.

The old portfolios were reevaluated once per representation at 128 simulations:

| portfolio | BASE200 solved / capped loss | corrected NEW70 solved / capped loss |
|---|---:|---:|
| initial | **29 / 46,765** | **8 / 17,063** |
| static sharing final | 27 / 47,251 | 7 / 17,094 |
| static no-sharing final | 27 / 47,694 | 7 / 17,116 |

On corrected NEW70 the two final arms solved the exact same seven items; initial
also solved `11a_14`. On BASE, final sharing and no-sharing intersected on 25
items. Sharing-only successes were `11n_27`, `11n_46`; no-sharing-only successes
were `11n_76`, `11n_9`. Relative to initial, sharing lost `11a_24` and `11n_76`;
no-sharing gained `11n_9` but lost `11a_24`, `11n_27`, and `11n_46`. BASE is not
a clean endpoint because it contains ladder identity `10_124`. Deeper search
therefore confirms the old-training regression on the corrected transfer set.

The repaired 50-round static-sharing run was forked at round 48 into `pre`,
matched `RL0`, and one auxiliary-only witness. The fork used three minibatch
seeds, four steps, one shared slot of 32, and learning rate 0.00025. At 128
simulations D1-aux never changed portfolio coverage relative to its RL0 control,
but its capped-loss deltas were:

| split | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| BASE50 | +2 | +13 | -2 |
| corrected NEW70 | -11 | +22 | +11 |

It fails the all-seeds non-inferiority gate. RL0 versus untouched `pre` gained
two BASE50 successes in every seed and improved capped loss by 378--382, but its
NEW70 deltas were +22, -86 with one added solve, and +33. Native adaptation can
help the rehearsal distribution, but transfer is seed-unstable.

This also explains why rung-18 solve rates above 90% coexist with low one-shot
coverage. Each ladder iteration runs eight self-play episodes from one source
family and 96 optimizer steps; candidates generally stay roughly ten or more
iterations, then the reported solve rate pools 16 evaluation episodes per ratio.
The portfolio test gives each frozen scientist one attempt per heterogeneous
representation with no local update. These are intensive within-task adaptation
and one-shot transfer, respectively.

**Corrected decision:** keep CPU-32 and the 2,700 run blocked. The next local gate
is a rapid-adaptation test on a small identified-source-disjoint bank: create a
fresh disposable fork per task, run `F=5` current-representation iterations plus
`F_old=1` iteration on one distinct old BASE representation, and compare against
the `5+0` ablation and frozen deeper MCTS matched by measured network evaluations
and wall time. Start with ten BASE development tasks after removing identified
ladder identities and `s-window-128`; expand to 20 tasks and the three-scientist
portfolio only if `5+1` beats both controls.
Discard the fork after the task. A rollback rule may consult BASE retention only;
corrected NEW70 remains a terminal endpoint.

The first attempted 200-item rapid-adaptation expansion is invalid and must not
be cited. It loaded `s-window-128/stage22-after.pt`, an unpromoted checkpoint
whose embedded result was 0.0 solve rate after 100 capped iterations, rather
than the last promoted `stage21-after.pt`. A fresh check of the promoted
snapshot solved 12/12 held-out rung-21 instances at ratio 10 and 128
simulations. The runner now rejects unpromoted checkpoints and requires this
promoted-rung regression gate before starting. Passing that gate does not imply
high BASE coverage: the rung metric is measured on generated representations
of the rung source family, whereas BASE contains unrelated table-knot
identities and presentations. Therefore the corrected experiment returns to
the ten-task paired gate before any 200-item expansion.

The corrected paired gate used an outcome-blind, identified-source-disjoint
BASE20 bank: five 3-strand, five 4-strand, and ten 5-strand presentations. Each
of three seeds compared frozen compute-matched `5+1` search, trained `5+0`, and
trained `5+1` with aligned target RNG streams. Frozen search solved 7/20 in
every seed; both trained arms solved 6/20 in every seed. All trained solves were
already present in the first target iteration, so there were zero
post-training rescues. Frozen search alone found `11n_9` later in every seed.
Trained `5+1` worsened capped L10 versus frozen by +215, +206, and +235. Its
rehearsal retention was 1/20, 2/20, and 0/20 versus frozen 3/20, 2/20, and
0/20. This fails the learning gate and blocks the adaptive 200 experiment.

The original remaining-objective-budget path is not ready. The channel itself
is encoded correctly: seven requested caps map to inputs from 0.023 through
1.0. But the migrated stage-21 head produced exactly the same `p(solve)` at
every cap for every BASE20 item. The training artifact contains 1,077 eligible
solve-head positions, all positive and none negative. BASE20 calibration is
poor: mean `p(solve)` 0.612 versus observed 0.35, Brier 0.330, log loss 1.349.
Cap-exhausted attempts were censored out of solve loss, while auxiliary gradients
did not reach the zero-initialized encoder input. Therefore predicted-loss caps
and early stopping remain disabled for that checkpoint.

The first repaired implementation is deliberately limited to `s-window-128`.
Cap exhaustion is a negative label for `p(solve within remaining L)` but remains
masked from policy, scalar-value, `cc`, and move targets. Solve loss now trains
the shared body and encoder; cost losses stay detached. Four ensemble members
predict `cc` and moves first, the network constructs `L=A*cc+B*moves` exactly,
and a residual solve branch receives shared features, remaining budget, `cc`,
moves, and `L`. Zero-initialized skips carry remaining budget directly to the
shared body, scalar value, cost heads, and solve branch. Failed restart attempts
are retained and sampled across cap strata, and paired copies of the same state
enforce nondecreasing solve probability as budget increases.

Migration is function-preserving for policy, scalar value, solve probability,
`cc`, and moves. The original rung-21 checkpoint still solved 12/12 on its native
regression after the code change. In a five-knot easy-curriculum smoke test at
caps 4, 7, 14, 35, and 704, the first concentrated update exposed BatchNorm drift
and was rolled back. Freezing running statistics and reducing only budget
fine-tuning to learning rate 0.00025 made all five training knots monotone and
budget-sensitive while retaining 6/6 on the promoted rung; conditional cost there
changed from 5.00 crossings/18.00 moves to 4.67/17.83. This is not a held-out
admission result. Keep predicted caps disabled until source-disjoint calibration
and retention pass. If transfer from the promoted model fails that gate, train
the same `s-window-128` architecture from the earliest easy representations
before changing architecture or propagating the design to other scientists.

The source-disjoint calibration and retention gate has now passed on a fresh
decision split. The first ten held-out knots improved substantially (Brier 0.430
to 0.133, AUC 0.827 to 0.886, coverage 23/200 to 24/200, promoted rung 12/12),
but formally failed the preregistered `8/10` sensitivity condition: six knots had
no solve even at the global cap, so their correct curves stayed near zero. That
split was retained as a failed calibration run and not reused for admission.

Before seeing the next ten knots, the criterion was corrected: every knot with
both solved and failed attempts must be sensitive, and every never-solved knot
must have maximum predicted solve probability at most 0.1. On the untouched next
ten knots and 200 attempts, all 10 curves were monotone, all 7/7 informative knots
were sensitive, and all 3/3 never-solved knots had low probabilities. Brier
improved from 0.722 to 0.232, AUC from 0.672 to 0.824, and solves from 33/200 to
36/200. Attempt-level paired sets were 30 shared, 6 trained-only, and 3
baseline-only. The promoted rung remained 12/12; conditional cost improved from
4.75 crossings/19.83 moves to 3.58/16.83. Every corrected check passed.

**Decision:** admit a bounded search-savings ablation with paired full-budget and
cap-and-restart search. Do not yet enable predicted caps in collaboration, the
200-representation experiment, or the 2,700-representation experiment.

That ablation is complete. The direct proposal `L_max=2*L_predicted` plus
geometric restart preserved the exact solved set (20/80) and improved aggregate
L10 from 992 to 862, but failed the purpose of budgeting: 246 restarts increased
scheduled evaluations from 133,056 to 604,032 and wall time from 96 to 445
seconds. It used 4.54 times the full-budget compute and is rejected.

The failure mechanism was concentrated in never-solved tasks: each was replayed
four or five times before reaching the same global-budget failure. Using that
split only for development, the next rule was frozen before another untouched
20-knot slice:

```text
if p_solve(global budget) < 0.04:
    run one attempt with L_max = 2 * L_predicted; accept failure without restart
else:
    run one global-budget attempt
```

On 80 fresh paired attempts this solve-gated rule retained the exact 11-attempt
solved set and the same aggregate L10=573. Scheduled evaluations fell from
149,985 to 101,178 (32.5% savings), and wall time fell from 108.7 to 72.5
seconds. Sixty attempts used the bounded probe and twenty went directly to the
global budget. Every gate passed: identical final solved set, non-inferior
objective, and at least 20% compute savings.

**Decision:** the solve-gated budget rule may be an optional arm when the
separate learning and scheduling gates unblock the corrected 200 pilot. It does
not override the failed rapid-adaptation gate and does not yet authorize the
collaborative 200 or 2,700 runs.

### K=3 network repair and fast-learning admission, 2026-08-04

The budget-aware repair was propagated from `s-window-128` to the actual K=3
roster, `s-window-128`, `d-tape4-u1`, and `s-w11-128`. The shared design predicts
`cc` and moves, constructs `L=A*cc+B*moves` exactly, and conditions `p(solve)` on
those values, the shared body, and remaining budget. Solve loss reaches the shared
encoder; capped failures remain excluded from policy, scalar-value, and cost
targets. A frozen-teacher policy/value trust region and promoted-rung rehearsal
prevent the narrow critic curriculum from silently erasing an established
controller. The distilled tape network additionally needs controller LR `5e-5`,
auxiliary LR `1e-3`, and a stronger preservation penalty; a single LR either
collapsed its policy or left its old all-positive solve head saturated.

Two roster defaults were invalid: window stage 22 and wide-window stage 19 are
unpromoted capped snapshots. The corrected defaults are stages 21 and 18, and
launcher preflight now rejects unpromoted or mismatched checkpoints. Predicted
objective budgets default off.

All three networks learned across 65 identities with only eight updates per
identity. Every one of 65 training curves was monotone, all empirically
informative identities were sensitive, and promoted-rung retention passed. On a
fresh 75--84 slice (200 paired attempts per scientist), all three sharply improved
Brier score and AUC while retaining their native rung. Only `d-tape4-u1` passed
the complete cap gate: its paired solved set stayed exactly 16/16, Brier improved
from 0.353 to 0.0002, and AUC from 0.826 to 1.000. `s-window-128` improved Brier
0.506 to 0.043 and AUC 0.826 to 0.995 but lost one paired global-cap solve,
`10_123@704#1`. `s-w11-128` improved coverage 10 to 11, Brier 0.446 to 0.049,
and AUC 0.805 to 0.980, but remained overconfident on two never-solved identities.

**Decision:** all three critic architectures are repaired and learn quickly; only
`d-tape4-u1` may currently use predicted caps or budget-based early failure.
Window and wide-window start controlled full-budget arms from their promoted
source checkpoints and keep the critic in shadow mode. This gate does not show
that all three solver policies learn quickly--paired coverage changed by -1, 0,
and +1--and does not yet prove persistent 200-task RL, adaptive scheduling, or
sharing.

## What would kill the programme

Stop or reduce the claim if any of these occur:

* fewer than 450 compatible exact-`u` identities exist under the requested bank
  constraints and the paper requires all 2,700 rows to have exact `u`;
* calibrated heads do not rank unseen tasks better than the frozen cheap proxy;
* sender-to-receiver witness translation has low coverage or prohibitive move
  inflation;
* the chosen scientists fail on the same representations, so K adds compute but
  not coverage;
* adaptive ordering wins only without total-compute matching; or
* sharing improves the training bank but reduces held-out knot-level performance
  by homogenising the population.

Any of these is still a useful result, but it supports a narrower paper than the
full collaboration claim.
