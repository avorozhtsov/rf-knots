# 16 — Scientists collaborating through verified solutions

This is a concrete programme combining all three directions in
[13](13-directions.md): an adaptive schedule, a population arena without network
recombination, and learning from the best verified solution anyone found. The
proposal is strong enough to become a paper experiment, but the first draft of the
loop mixes several quantities that have to be separated before its result can mean
anything.

Definitions and reporting denominators are collected in
[00-glossary.md](00-glossary.md).

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

Fresh collaboration tasks never use one scientist's predicted cost as an attempt
cap. Local serial scientists see only the current window at the initial state,
so `2 * L_predicted` can be arbitrarily wrong about an unseen tail and would give
different scientists different solving opportunities. For observed braid-word
length `c`, ratio `A:B` normalized to `A/B`, and common native action horizon
`H`, every scientist instead starts at the representation-derived tier

```text
min((A/B + 1) * H, (A/B) * ceil(c / 2) + H).
```

This first tier is not called an upper bound. Every `budget_exhausted` failure is
repeated with the same seed at the common global cap `(A/B + 1) * H`; both records
enter replay with their actual encoded budgets, while only the final attempt
decides task success. Ordinary action-horizon failures are not repeated because
the objective cap did not censor them. The old predicted-cap multiplier and 10%
audit are retained only in historical artifacts, not in collaboration schema
`collaborative-scientists-v5-common-structural-budget`.

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

A historical provisional roster was:

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

The current independent K=3 roster supersedes that provisional list:
`s-window-128`, `s-tape4`, and `s-w11-128`. The tape scientist now starts from
its own rung-ladder checkpoint rather than weights distilled from `u1-puct`.

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

This makes objective interference a live risk, not a settled conclusion about
simultaneous `10:1` and `1000:1` training. The final target is now explicitly
`1000:1`: `10:1` is valuable only insofar as its exploration discovers witnesses
that re-score better under `L1000`. A short gate can check plumbing or reject a
catastrophic regression, but it cannot decide this multi-task question. The
decisive compute-matched ablation must use at least 100 knot-identity-disjoint
representations and compare `1000:1`-only, equal dual-objective, and target-biased
dual-objective training. Every witness found by either ratio is re-scored under
`L1000`, and final evaluation uses only `1000:1`. Report the exact paired solved
sets, capped `L1000`, crossing changes on common successes, and how often the
winning `L1000` witness was originally discovered by the `10:1` explorer.

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

### Independent tape migration and internal-budget ablation, 2026-08-04

The primary K=3 roster now replaces distilled `d-tape4-u1` with independent
`s-tape4`, loaded from its promoted rung-18 checkpoint. Checkpoint migration was
generalised from exactly one new observation channel to any number of appended
channels. Every old input weight is copied unchanged and every new input column
is zero, so new budget features are initially ignored. On the real checkpoint,
policy, scalar value, solve probability, predicted crossing changes, and predicted
moves were bit-for-bit identical at objective caps 4, 12, 100, and 704. The
source checkpoint SHA-256 is
`0176e975e9c9d5e616c9ca8b074c211a301b03cefb4b95f52ab0c24b2fa180df`.

A five-identity varied-budget admission used caps 4, 7, 14, 35, and 704, two
self-play attempts per cap, 64 simulations, 32 updates per identity, eight native
rung rehearsal games, learning rate `2.5e-4`, and a frozen-teacher preservation
loss. All five learned curves were monotone and budget-sensitive. Both identities
with mixed solved/failed observations were sensitive. Promoted-rung retention
remained 8/8; conditional crossing changes/moves improved from 1.375/18.875 to
1.25/18.75. This passes the easy-curriculum expansion gate, not the source-disjoint
collaboration gate.

`s-tape4-h5` is a separate candidate. It begins from the same independent
`s-tape4` weights, adds both the objective-remaining and internal-remaining budget
planes by the same zero-padding migration, and limits consecutive head/tape
operations to five. Historical distilled models retain their old fraction-spent
encoding. The new arm exposes `1 - internal_steps/5`, decreasing from 1.0 to 0.8
after one internal action. Do not substitute this arm into K=3; compare it as a
capacity/credit-assignment ablation only after ordinary `s-tape4` passes the
source-disjoint solver-learning gate.

### Source-disjoint critic admission and fixed-F solver gate, 2026-08-04

A new outcome-blind critic corpus excludes all 270 BASE200/corrected-NEW70
identities and every identified ladder identity, then fixes 60 training, 20
validation, and 20 decision identities with 3/4/5-strand quotas 15/15/30,
5/5/10, and 5/5/10. Split SHA-256 hashes are respectively
`2ae022596872a3e39796a63feb8aabbf5feaa1d1946f83553aa73078c481f752`,
`6cda775a1e0c8dd8807c4976b06e9ed80c945d1d70e7b3ce7c94fcc92c76404f`,
and `89f39a3d31eef8b0008c00389b4b80b080c78d8a912d5843dded667daf5d880a`.
Each candidate saw five objective caps, two 64-simulation attempts per cap, eight
updates per identity, native-rung rehearsal, and a frozen-teacher preservation
loss. Positive-scale Platt calibration used validation outcomes and changed only
checkpoint metadata, not network weights.

Two critics passed the untouched decision split:

| scientist | paired solves baseline -> trained | AUC | Brier skill | ECE-5 | rung retention |
|---|---:|---:|---:|---:|---:|
| `s-window-128` | 17 -> 19 / 400 | 0.987 | 0.543 | 0.012 | 12/12 -> 12/12 |
| `s-w11-128` | 22 -> 23 / 400 | 0.928 | 0.727 | 0.024 | 12/12 -> 12/12 |

Both were monotone on all 20 identities and sensitive on every empirically
informative identity. They may drive adaptive ordering only at the calibrated
64-simulation population budget. `s-tape4` did not pass. Its conservative training
run was monotone on 60/60 training identities and retained its rung in the training
seed, but paired validation produced zero trained successes versus one baseline
success, no positive label for AUC, and rung retention 11/12 versus 12/12. Keep
its critic in shadow mode; the independent tape can still solve in static,
full-budget arms.

The next solver-learning bank excludes BASE200, NEW70, the 100 critic identities,
and ladder identities. The fixed paired treatment was `F=8`, `F_old=1`; controls
were frozen `8+1` search and trained `8+0`, with 64 simulations, eight self-play
games per iteration, and 96 optimizer steps. The gate was stopped after two
complete seeds because failure was already irreversible:

| seed | frozen solved / capped C10 | trained 8+0 | trained 8+1 | post-training rescues from 8+1 |
|---|---:|---:|---:|---:|
| 20261360 | 9 / 3,366 | 9 / 3,366 | 9 / 3,366 | 0 |
| 20261361 | 9 / 3,366 | 8 / 3,578 | 8 / 3,578 | 0 |

All three seed-0 solved sets were identical. In seed 1 both trained arms lost
`11a_33`, so `8+1` was one solve and 212 capped-loss points worse than frozen
search. Rehearsal did not create a rescue: it only matched `8+0` on target tasks.
The third seed could not restore the required all-seeds non-inferiority and was
terminated to save compute.

**Decision:** the critic-pretraining gate succeeds for both window scientists but
the native solver-learning gate fails even at `F=8`. Do not run the sharing gate,
the four-arm BASE200 factorial, CPU-32, or the 2,700 experiment. Sharing cannot be
interpreted while the receiver's native update is itself non-beneficial. The next
research problem is the policy-learning update, not scheduling or distillation.

### Policy-update diagnosis, 2026-08-05

The failed rapid learner had two distinct defects. First, its configuration set a
policy/value preservation weight but never attached the frozen teacher, so the
reported preservation loss was identically zero. Second, every task iteration
applied 96 optimizer steps to eight highly correlated games even when all eight
searches failed. Across `F=8`, `F_old=1`, this is 864 updates including the
rehearsal iteration. Ordinary position-uniform replay then imitates unsuccessful
MCTS visit distributions before the task has supplied any positive policy target.

A five-arm paired diagnostic used the same promoted `s-window-128` checkpoint,
`11a_33`, 64 simulations, eight games per iteration, the same rehearsal rule,
and three seeds. It compared frozen search, the legacy 96-step update, 96 and 24
steps with a real frozen teacher, and a 24-step arm that both used the teacher and
refused all task-local updates until replay contained a genuine uncensored native
solve. Once admitted, the last arm used episode-uniform, success-balanced replay.
The historical task-local seed family gave:

| arm | optimizer steps | self-play solves / 192 | solves by current ordinal | post probes / 24 |
|---|---:|---:|---|---:|
| frozen | 0 | 20 | `0,4,0,3,6,0,3,4` | 0 |
| legacy 96 | 2,592 | 0 | `0,0,0,0,0,0,0,0` | 0 |
| guarded 96 | 2,592 | 71 | `0,0,3,7,9,14,20,18` | 24 |
| guarded 24 | 648 | 14 | `0,0,2,4,0,2,3,3` | 0 |
| success-gated 24 | 576 | 132 | `0,4,8,24,24,24,24,24` | 24 |

This reproduces the erased solve deterministically. On the first historical seed,
both frozen and trained arms began with 0/8. The frozen network then solved 2/8
on the identical ordinal-1 seeds; after 96 failure-only updates the legacy arm
solved 0/8 and never recovered. The success-gated arm preserved that 2/8 batch,
then converted the following sequence to `1,8,8,8,8,8`. The same qualitative
result held for all three seeds. A second disjoint seed family was consistent:
legacy made 2,592 updates with zero self-play solves, guarded 96 reached 16/24
post probes, guarded 24 reached 8/24, and success-gated 24 reached 24/24 using
only 480 updates.

The artifact is
`pgx-mcts-bench/artifacts/policy-update-diagnostic-swindow-lost-solve-20260805/report.json`.
It freezes checkpoint and bank hashes, per-seed rows, exact solve sets, update
counts, and preservation losses. The diagnostic runner also verifies that all
arms are bit-for-bit identical before their first update.

**Decision:** the main failure is destructive imitation of failure-only search,
amplified by the missing teacher; it is not too few optimizer steps. Preserve the
controller with a real frozen teacher and do not update policy/value from a new
task until at least one verified solution exists. The 24-step success-gated arm
is the preferred efficient repair. Before a 20- or 200-representation flight,
split the optimization masks so failed/censored attempts may still train
`p(solve)` while policy, scalar value, and conditional cost remain protected;
then repeat this causal gate on several representations, including tasks with no
frozen solve. Sharing and adaptive scheduling remain closed until that
multi-representation gate shows rescues without loss of the frozen solved set.

### Small split-loss gate, 2026-08-05

The split was implemented directly in the AlphaZero update. Failed and censored
attempts remain eligible for the budget-conditioned `p(solve)` loss and its shared
encoder gradient. Policy and scalar-value targets are restricted to verified
successful native trajectories; conditional crossing-change and move targets
were already success-only. A real frozen teacher constrains policy/value drift.
After the first success, replay is episode-uniform and success-balanced.

The preregistered small gate selected three retention identities,
`12a_146`, `11a_26`, and borderline `11a_33`, plus `11n_107`, `10_71`, and
`10_137`. Each of the latter had zero solves in 128 frozen attempts over the
previous two-seed gate. Frozen and split-success-24 arms used `F=8`, `F_old=1`,
64 simulations, eight games per iteration, and matched seeds. Admission required
no frozen-only solved identity, non-worse capped objective in every seed, and the
same historical never-solved identity rescued in at least two seeds.

The first two seeds completed as follows:

| seed | frozen solved set | split solved set | frozen -> split self-play solves / 384 | capped C10 |
|---|---|---|---:|---:|
| 20261420 | retention 3/3, never 0/3 | retention 3/3, never 0/3 | 92 -> 176 | 948 -> 948 |
| 20261421 | retention 3/3, never 0/3 | retention 3/3, never 0/3 | 91 -> 177 | 948 -> 948 |

There were no treatment-only or control-only identities. Split learning made 528
and 648 optimizer steps whose sampled batch had no policy/value target, confirming
that the failure critic was trained without imitating failed policies. Nevertheless,
all three zero-positive tasks remained unsolved in both seeds. Because only one of
three declared seeds remained, a rescue replicated in two seeds was impossible;
the third seed was stopped to save compute. The frozen solved-set and objective
non-inferiority gates passed, but the rescue gate failed.

The frozen artifact is
`pgx-mcts-bench/artifacts/split-loss-gate-swindow-20260805/early-stop-report.json`.
It includes checkpoint and bank hashes, the fixed strata, exact paired sets,
per-item self-play counts, critic-only update counts, and the irreversible-stop
reason.

**Decision:** admit split-success-24 as a safe and strong *consolidation* update,
not as a discovery mechanism. It nearly doubled success frequency on tasks where
search supplied positive examples, while preserving established coverage. It did
not invent a solution on any zero-positive task, which is the expected limitation
of success-only policy learning. Keep the 20-, 200-, sharing-, and adaptive-order
gates closed. The next discovery gate must first supply a verified positive seed
for a frozen-never task--from deeper/diversified search, another scientist, or an
alternate representation--and only then test whether split-success-24 reliably
consolidates and transfers that witness without losing the frozen solved set.

### Readiness repair and five-arm protocol, 2026-08-05

“Solved” must not mean “the search respected one arbitrary objective cap.”  An
admitted solution is an exactly replayed trajectory whose final braid word is
empty and whose strand count is one.  For the readiness and comparison runs the
learned objective cap and crossing-change cap are disabled unless the manifest
explicitly enables the budget experiment.  Search still has a finite native
action horizon, currently 64.  Therefore a failure is censored beyond 64 native
actions; it is not evidence that the network cannot unknot the representation at
a larger horizon.  Completeness-oriented evaluation must repeat failures under a
declared increasing horizon sequence rather than call the 64-action result
“unsolvable.”

Three update regimes must remain separate:

1. Fresh curriculum training uses the historical rung-18 AlphaZero update on
   wins and losses.  Negative trajectories are necessary to train a critic from
   random initialization.
2. Task-local adaptation of a promoted scientist uses split-success learning:
   failures may train the conditional solve head, but policy and scalar value
   imitate verified successes only, with a frozen starting-network teacher.
3. Sharing uses bounded-option distillation.  For every certified semantic braid
   edit, a serial receiver may choose any legal sequence of at most five internal
   actions--including shifts, tape writes, and controller-state changes--and must
   then perform that edit.  Shared trajectories supply solve/cost upper bounds,
   but do not force one hand-written navigation route into the policy.

Applying regime 2 during training from scratch is itself a protocol bug.  A
strict success-only rung-0 ablation needed ten iterations and still learned a
worse crossing-change policy where the historical update promoted in two.  This
does not contradict the task-local diagnostic: it establishes that the safe
pretrained update is not a replacement for the successful fresh curriculum.

The long comparison is fixed to five arms:

1. four scientists, adaptive schedule, bounded-option sharing;
2. four scientists, adaptive schedule, no sharing;
3. four scientists, static schedule, bounded-option sharing;
4. four scientists, static schedule, no sharing; and
5. one best scientist, static schedule, compute matched.

For arm 5, qualification simulations, full simulations, and optimizer updates
are multiplied by `K`; scheduled network evaluations and wall time are reported
as measured checks, not inferred from the multiplier.  Sharing and no-sharing
arms receive the same number of optimizer steps: one bounded-option update is
matched by one additional native-control update.  Each run freezes checkpoint,
bank, anchor, and protocol hashes, commits every round transactionally, and can
resume only when the protocol hash matches.

Readiness remains sequential.  First reproduce the first ten rungs with
`s-window-128`, 128 simulations, eight games and 96 optimizer steps per
iteration, twelve held-out games per ratio, and the historical 100-iteration
cap.  Every rung must promote with held-out solve rate at least 0.80.  Then run a
matched multi-representation bounded-option admission gate, including witnesses
the receiver did not find itself, and require target realization plus no loss of
the frozen solved set.  Only after those gates pass run the transactional
five-arm smoke test, followed by a small multi-seed population pilot.  CPU-32 and
the 200-representation comparison remain closed until all four scientists pass
critic calibration and paired retention; at present the fourth adaptive-order
scientist is not admitted.

### Replay contract for the collaboration arms

The collaboration runner now retains complete attempts for the most recently
used 100 representation identities.  Each attempt records its representation,
termination reason, objective cap, action horizon, residual word length, best
residual length, MCTS root value and visit count, position index, and persistent
episode/position exposure counts.  These counters are part of the transactional
checkpoint, so resuming a run does not reset which evidence has already been
reused heavily.

Each collaboration minibatch allocates episode slots as 25% current
representation, 25% structurally similar retained representations, and 50%
global retained history.  The initial similarity key is a deterministic vector
of braid length, strands, crossings, writhe-like totals, signs, sign changes and
generator frequencies; cosine similarity chooses neighbours.  This is an
explicit cheap baseline, not a claim that it is a learned knot embedding.  A
learned encoder embedding may replace it only as a separately tested protocol.

Within the requested representation group, replay first balances uncensored
native success and native failure 50:50 when both exist.  It then divides the
negative half between ordinary and objective-censored failures when both exist,
and caps shared witnesses at the arm's declared fraction.  Representation
identities are sampled uniformly; attempts within one identity are sampled
inversely to their prior episode exposure.  Thus long attempts and frequently
replayed easy examples do not automatically dominate.

A selected attempt supplies four states rather than throwing away all but one:
the initial state, the final recorded state, the highest-policy-entropy interior
state, and an inverse-exposure state.  Very short attempts reuse states as
needed.  Negative and budget-censored attempts train the conditional `p(solve)`
head, but task-local policy/value/cost targets remain success-gated.  A failed
search is therefore useful critic evidence without becoming a policy imitation
target.  Verified shared solutions still enter through the separate
bounded-option target.

The 25/25/50 mixture is a training distribution, not an estimate of the natural
success probability.  Calibration metrics must therefore be computed on an
unrebalanced held-out stream or corrected for the sampling propensity.  Merely
lowering the objective budget may create useful conditional failures, but those
failures never become policy or solved-cost targets.  Increasing the native
action horizon and learned representation retrieval remain separate experiments;
retrieving a similar knot does not turn its solution into a valid target until
exact translation and replay verify it on the current representation.

### Replay-v3 readiness results, 2026-08-05

Replay-v3 passed its real-trajectory integrity gate.  `s-window-128` generated
36 attempts and 1,461 positions on six fixed identities, including three native
successes and twelve objective-censored failures.  Save/resume preserved every
exposure counter.  A 512-position audit batch allocated exactly 128 positions to
the current-representation quota, 128 to similar history and 256 to global
history; it contained 256 native-success targets, 128 ordinary failures and 128
budget-censored failures.  No failure received a policy/value target.  The
artifact is `pgx-mcts-bench/artifacts/replay-v3-integrity-swindow-20260805/report.json`.

The paired learning gate compared old success-balanced replay with replay-v3 on
the same six identities, two seeds, 64 simulations, eight self-play games, and
24 matched optimizer steps.  Starting diagnostics were bit-for-bit identical.
In both seeds old replay finished with `{11a_26, 11a_33}`, while replay-v3 also
solved `12a_146` and lost no old-only identity.  Capped loss improved by 1,720
and 2,232; current-task self-play solves improved from 147 to 164 and from 137
to 158.  Replay-v3 is therefore admitted.  The artifact is
`pgx-mcts-bench/artifacts/replay-v3-learning-gate-swindow-20260805/report.json`.

The unrebalanced, identity-disjoint `s-window-128` critic gate also passes:
AUC 0.947, calibrated Brier score 0.0266, ECE 0.0166, 20/20 budget-monotone
items, 15/20 budget-sensitive items, no baseline-only held-out solve, and no
promoted-rung regression.  This admits the critic for adaptive ordering but not
for predicted early-stop caps.  The current decision artifact is
`pgx-mcts-bench/artifacts/critic-pretrain-independent-k3-20260804/s-window-128-readiness.json`.
The historical first-ten-rung reproduction had already passed all promotions
with held-out solve rates from 0.806 to 1.0, so fresh-curriculum parity remains
admitted.

Bounded-option sharing is **not admitted**.  Two real `s-window-128` witnesses,
`11a_26` and `12a_146`, translated exactly into both `s-tape4` and
`s-w11-128`, but pure option training produced no new solved identity and made
`s-w11-128` lose `12a_146`.  Architecture-specific preservation weights and a
smaller eight-update dose did not repair the regression.  A compute-matched
interleaved gate then compared native-plus-option against native-plus-native
updates.  Native-only learned `11a_26`; sharing did not.  Training a complete
receiver-unsolved witness instead of one sampled position also failed.  Finally,
one complete `11a_26` witness, one option update, and an option learning rate of
0.1 times the native rate still made sharing lose `12a_146`, while native-only
retained it; full option loss rose from 2.35 to 6.96.  The decisive artifact is
`pgx-mcts-bench/artifacts/interleaved-minimal-sharing-w11-20260805/report.json`.

**Decision:** admit replay-v3, the calibrated `s-window-128` ordering critic,
and historical fresh-rung training.  Keep both sharing arms, roster finalization,
the five-arm smoke, CPU-32, and the 200-representation experiment closed.  The
next sharing repair must improve actual option transfer while retaining the
isolation achieved by the zero-initialized adapter described below, and must
pass the same native-plus-native paired control before any arm comparison
resumes.  Increasing preservation weight, reducing dose, sampling one position,
training one whole witness, and merely lowering the learning rate are now
rejected repairs.

### Zero-initialized option-policy adapter gate, 2026-08-05

The sharing path now attaches a separate width-32 policy residual with two
normalized residual blocks and a zero-initialized output projection.  Attaching
it is bit-for-bit function preserving.  It reads the complete serial observation
and, for h5 scientists, the decreasing internal-action budget; historical
rung-18 scientists without that channel receive a constant full-budget input.
Bounded-option updates optimize only this adapter.  The original policy body,
all value heads, and BatchNorm running statistics remain fixed during the
sharing update.  Native reinforcement-learning updates continue to use the
original optimizer.  Adapter parameters and their optimizer state are included
in transactional checkpoints and worker/evaluation checkpoint migration.

The exact one-cycle reproduction on `s-w11-128` established the safety half of
the repair.  Before training, sharing, and native-only control all solved only
`12a_146`; after one paired cycle both arms still solved exactly `12a_146`, with
no baseline loss.  Sharing capped loss was 1,392 versus 1,406 for native-only.
The artifact is
`pgx-mcts-bench/artifacts/interleaved-option-adapter-minimal-w11-v2-20260805/report.json`.

The transfer half did not pass.  Across eight paired cycles the adapter arm
retained `12a_146`, but native-only additionally learned `11a_26`; the adapter
arm did not.  Capped loss was 1,424 versus 1,269.  Aggregate option loss rose
from 2.355 to 2.727 while the per-cycle loss measured after each native update
also rose, although an isolated one-step regression test confirms that an
adapter update descends its fixed-position loss and changes no base parameter or
value prediction.  This points to interference between changing native policy
logits, discrete option-beam membership, and the adapter objective rather than
the old shared-encoder degradation.  The artifact is
`pgx-mcts-bench/artifacts/interleaved-option-adapter-w11-8cycle-20260805/report.json`.

**Decision:** the adapter implementation is admitted as a safety mechanism, but
sharing remains closed.  The next gate should measure option loss immediately
before and after every adapter step on a frozen witness set, stabilize route
selection independently of the current policy beam, and require both transfer
of a receiver-unsolved witness and paired solved-set noninferiority.

That next repair is now implemented.  Adapter training no longer chooses a
policy-dependent beam.  For each certified semantic edit it deterministically
selects the shortest legal head route, followed by the corresponding local
external edit, and teacher-forces that complete option.  The legacy beam loss is
retained only for historical diagnostics.  Every sharing update records loss on
the exact same frozen positions immediately before and after the optimizer
step, including route-target identity and position count.

On the eight-cycle `s-w11-128` paired gate every adapter step genuinely reduced
its fixed-route loss.  At 0.1 times the native learning rate the eight immediate
deltas ranged from -0.00056 to -0.00079.  At the declared 1.0 times rate they
grew from -0.0051 to -0.0297.  Thus the old increasing-loss report was not an
optimizer-sign bug: intervening native updates move the base logits faster than
the conservative adapter update corrects them.  Nevertheless neither dose
transferred `11a_26`; native-only learned it, while both adapter doses retained
the baseline `12a_146`.  At 1.0 times, sharing capped loss was 1,424 versus
1,269 for native-only.  The decisive artifact is
`pgx-mcts-bench/artifacts/interleaved-stable-option-adapter-w11-8cycle-lr1-20260805/report.json`.

**Updated decision:** deterministic route stability and per-step measurement
are admitted, but useful sharing is still not demonstrated.  Do not open the
long sharing arms.  The next bounded investigation is dose matching: multiple
adapter steps per native update or a target-loss decrease threshold, with the
same paired solved-set retention gate and explicit adapter compute accounting.

The dose-matched investigation is now complete.  After every native update the
adapter trains on one frozen translated witness until its canonical-route loss
falls by 10%, or until sixteen adapter steps.  The native-only control receives
at least the same optimizer-step count and at least the same number of training
state examples.  Because the control updates the full scientist while sharing
back-propagates only through the adapter, this is conservative in the control's
favour.  The report also records wall time rather than pretending the two kinds
of update have identical implementation overhead.

Across seeds 20260854--20260856, sharing learned the targeted receiver-unsolved
`11a_26` in all three runs and retained the baseline `12a_146` in all three.
This is the first repeatable evidence that translated-solution learning actually
transfers an option.  Native-only also learned `11a_26` in two of three seeds.
The paired final sets were:

| seed | sharing | native-only | capped loss sharing / native |
|---:|---|---|---:|
| 20260854 | `11a_26`, `12a_146` | `11a_33`, `12a_146` | 1,193 / 1,216 |
| 20260855 | `11a_26`, `12a_146` | `11a_26`, `12a_146` | 1,169 / 1,173 |
| 20260856 | `11a_26`, `12a_146` | `11a_26`, `11a_33`, `12a_146` | 1,177 / 1,009 |

Thus sharing beat capped loss in two seeds but lost badly in one; mean capped
loss was 1,180 for sharing versus 1,133 for native-only.  One seed had a
sharing-only identity and one had a control-only identity, so the required
same-final-solved-set/noninferiority condition does not pass.  Adapter training
used 33--46 optimizer steps and 363--575 routed state examples; controls used
the same optimizer-step counts and 528--736 native state examples.  The primary
artifact is
`pgx-mcts-bench/artifacts/interleaved-threshold-compute-matched-w11-v2-20260805/report.json`,
with the two replications under the corresponding `seed20260855` and
`seed20260856` artifact directories.

**Current decision:** threshold dosing is admitted as evidence that option
sharing can transfer a specified solution without erasing the baseline solved
set.  It is not evidence that collaboration beats compute-matched native
learning.  Keep the long sharing arms closed until this advantage replicates on
multiple receiver-unsolved witnesses and held-out identities, with paired final
sets reported exactly as above.

### Multi-witness sharing gate, 2026-08-05

The next gate mined solutions from the committed transactional state of the
old static-sharing run rather than trusting summary rows.  A witness was admitted
only when its complete stored action sequence replayed to the empty one-strand
braid and its crossing-change and move counts matched.  This recovered 25
distinct certified representations, with author, episode seed, source round,
source-manifest hash, and bank hash.  The resumable screen stores one atomic row
per representation and can use several widely separated evaluation seed blocks.

The first `s-w11-128` screen found only four receiver-unsolved witnesses at 128
simulations and 16 attempts: `11a_231`, `11n_119`, `12a_1215`, and `12a_722`.
On the gate's different seed block, however, the frozen network solved the latter
two in all eight attempts.  Sharing and compute-matched native learning then
finished with exactly the same seven solved identities; capped loss was 403
versus 460.  This is safe parity, not evidence for sharing, and demonstrates why
a single seed block is not a robust definition of “receiver-unsolved.”  The
artifact is
`pgx-mcts-bench/artifacts/multi-witness-gate-s-w11-seed20260910-20260805/report.json`.

The stronger `s-tape4-h5` screen evaluated every candidate on three separated
seed blocks, eight games per block, and 128 simulations.  Nine witnesses remained
unsolved in all 24 attempts.  Eight structurally varied identities formed the
training panel: `10_126`, `11a_15`, `12a_1215`, `11a_231`, `12a_1222`,
`11n_119`, `12a_1225`, and `12a_1235`; `12a_850` was held out.  Eight easy
identities were included as retention canaries in the gate.  The screen artifact
is
`pgx-mcts-bench/artifacts/multi-witness-screen-s-tape4-h5-strong-20260805/report.json`.

The three-arm gate compared the frozen checkpoint, native learning plus
threshold-dosed canonical-route adapter sharing, and a native-only control.  Both
trained arms used 16 cycles and four fresh native games per cycle.  The control
consumed at least as many optimizer state examples as the adapter.  A protocol
audit found that the first implementation gave the two arms different native
refresh seed streams; those runs are diagnostic only.  Version 4 uses identical
native refresh seeds, and all final evaluations are paired on the same seeds.
At 128 simulations and eight final games per identity, the corrected results
were:

| seed | target solves sharing / control | sharing-only | control-only | frozen identities lost by sharing | capped loss sharing / control |
|---:|---:|---|---|---|---:|
| 20260950 | 6/8 / 6/8 | `10_159`, `11n_119`, `12a_1199` | `11a_231` | `11n_46` | 2,241 / 2,092 |
| 20260951 | 7/8 / 5/8 | `11a_231`, `12a_1222` | none | none | 1,359 / 1,665 |
| 20260952 | 7/8 / 8/8 | none | `11n_119`, `11n_46`, `12a_1199` | `11n_46`, `12a_1199` | 1,996 / 1,106 |

Here sharing-only and control-only are computed over the complete 17-identity
evaluation set, not just the eight training targets.  Both arms solved the held-
out `12a_850` in every seed.  Mean target transfer was 83.3% for sharing and
79.2% for native-only, but mean capped loss was worse, 1,865.3 versus 1,621.0.
Only one of three seeds passed exact paired non-inferiority.  Option loss fell in
every seed (from 6.827 to 3.918, 3.880, and 3.827), so the failure is not an
adapter-optimization failure: it is unstable downstream policy interaction and
retention.

**Decision:** the multi-witness machinery, robust screening, paired refresh
seeds, and the fact that sharing can add identities are admitted.  Collaboration
itself is not admitted: two seeds lost either a paired identity, a frozen solved
identity, or objective quality.  Keep the two long sharing arms and CPU-32 run
closed.  The next repair should reduce adapter influence when it conflicts with
native policy--for example by a learned or validated option gate--and must repeat
this exact panel without losing any frozen or control-only identity.  The two
no-sharing arms can proceed to a small transactional smoke test independently;
they do not depend on passing the sharing gate.

### Adapter counterfactual, gated repair, and no-sharing schedule gates

The final ungated sharing checkpoints were evaluated again with the learned
adapter either active or bypassed, using the exact final-evaluation seeds.  This
does not restore the native-only control, because the base network has already
learned under adapter-influenced self-play; it isolates the adapter's direct
contribution at evaluation time.  On failed seeds 20260950 and 20260952,
bypassing the adapter worsened capped loss from 2,241 to 3,118 and from 1,996 to
3,323.  The active adapter supplied seven and nine identities absent when it was
bypassed.  In seed 20260952 bypassing recovered only `12a_1199`.  At the initial
states, the ungated residual changed the legal-policy argmax on 15/17 and 13/17
identities; policy KL commonly lay between 3 and 5.7 and residual L2 norms were
about 40.  Thus the adapter is both useful and globally dominant.  Removing or
uniformly shrinking it is not the correct repair.

The next implementation therefore added an optional state-dependent gate.  It
is a separate module, so old adapter checkpoints retain their exact behavior.
New gated runs start with a 0.1 applicability probability and train three terms:
canonical-route likelihood, KL from the native policy on off-route states, and
an off-route gate penalty.  Every retention and held-out initial state is an
anchor; fresh native states from the current representation are sampled
deterministically.  The compute-matched control consumes at least the same total
route-plus-preservation state examples.

The three-seed development gate did not pass:

| seed | target solves sharing / control | sharing-only | control-only | frozen identities lost by sharing | capped loss sharing / control |
|---:|---:|---|---|---|---:|
| 20260950 | 6/8 / 7/8 | `11a_15` | `10_126`, `11n_119`, `12a_1199` | `11n_46`, `12a_1199` | 1,978 / 1,544 |
| 20260951 | 8/8 / 1/8 | ten identities | none | none | 1,513 / 3,084 |
| 20260952 | 7/8 / 0/8 | eleven identities | none | `12a_1199` | 1,453 / 3,645 |

Only seed 20260951 passed exact non-inferiority.  Sharing had much better mean
capped loss than the heavily compute-matched control, 1,648 versus 2,757.7, but
retention failed in two seeds.  Mean squared off-route gate activation remained
0.41--0.46, so the state gate still activated too broadly.  The large control
regression also warns that matching every adapter preservation example with a
full-network native update can over-train the control; this comparison is
conservative but not a desirable long-run update schedule.  No untouched
sharing-validation panel was opened.

The independent no-sharing path then ran transactional adaptive, static, and
compute-matched-solo arms.  A 20-round technical smoke used a fresh outcome-blind
20-identity bank.  Every initial and final arm solved the exact same six
identities: `10_152`, `11a_160`, `12a_1203`, `12a_1255`, `12a_722`, and
`12n_749`.  Capped L10 changed from 4,025 initially to 4,018 adaptive and 4,036
static; solo changed from 4,049 to 4,016.  Transactional execution, arm filtering,
evaluation, exact-set reporting, and compute accounting therefore passed.

The 50-round pilot used a new outcome-blind 50-identity bank.  It revealed a
real learning regression:

| arm/state | solved | identities lost from its initial comparator | capped L10 |
|---|---:|---|---:|
| three-scientist initial | 15/50 | -- | 10,111 |
| adaptive final | 13/50 | `11a_231`, `12a_1215` | 10,565 |
| static final | 14/50 | `12a_1222` | 10,507 |
| compute-matched solo initial | 15/50 | -- | 10,053 |
| compute-matched solo final | 15/50 | none | 10,064 |

Adaptive added no new identity and was two solves and 454 capped-loss points
worse than its initial portfolio.  Static added none and was one solve and 396
points worse.  Solo retained the exact set but worsened by 11 points.  The
training runner generated one new episode per scientist and representation,
then trained every five rounds.  Final replay contained all 50 identities, but
only one episode per identity; individual positions were reused as many as
92--283 times.  This is the one-shot heterogeneous schedule, not the intensive
within-task rung recipe.  It reproduces the previously diagnosed failure of
task-local/native policy learning and must not be interpreted as evidence
against adaptive ordering itself.

**Updated decision:** all 200-representation and CPU-32 arms remain closed.
Sharing remains unsafe, and the no-sharing learner loses frozen coverage before
schedule quality can be interpreted.  Do not add another adapter dose or simply
increase `F`: earlier source-disjoint `F=8`, `F_old=1` gates already showed no
post-training rescues.  The next valid target is again the native policy-learning
update: it must demonstrate rescues and exact retention on an independent
multi-representation gate before scheduling, sharing, or a fourth scientist is
tested at scale.

### Transactional native-learning gate

The next implementation isolated native solution discovery from scheduling and
sharing.  It used one `s-window-128` checkpoint and a fixed 12-representation
development panel: three historical retention identities, six transition
identities, and the three identities that the preceding split-loss experiment
never solved in 128 frozen attempts.  Search began with four 64-simulation
attempts.  Promising failures could receive two 128-simulation attempts and one
256-simulation attempt.  Attempts varied root seeds, PUCT strength, and cyclic
conjugates of the braid word.  Only trajectories passing exact witness replay
were policy targets.  Replay sampled up to four positions per episode, balanced
success and failure when both existed, and capped each position at 64 uses.
The final v2 run enforces this cap strictly within a batch; it reproduced the
same solved sets and objectives as the initial eligibility-capped run.

Every optimizer update was transactional.  The candidate was evaluated on the
same fixed seeds as its pre-update parent.  It was accepted only if it lost no
previously solved identity, did not worsen capped L10, and either rescued its
target or strictly improved capped L10.  Otherwise both network and optimizer
state were restored.  The preregistered multi-seed gate required two distinct
representations to become post-training rescues in at least two of three seeds,
with exact retention and non-worsening capped L10 in every seed.

| seed | initial solved | final solved | gained | lost | capped L10 initial / final | accepted / attempted updates |
|---:|---:|---:|---|---|---:|---:|
| 20261520 | 6/12 | 8/12 | `10_149`, `11a_33` | none | 1,913 / 1,520 | 1/6 |
| 20261521 | 6/12 | 8/12 | `10_149`, `11a_33` | none | 1,913 / 1,542 | 1/6 |
| 20261522 | 6/12 | 9/12 | `10_149`, `11a_33`, `12a_146` | none | 1,913 / 1,285 | 2/6 |

Rollback was necessary rather than ceremonial: 14 of 18 attempted updates were
rejected, usually because they erased `12a_981` or another fixed-seed solve.
`10_149` was a reproducible post-training rescue in all three seeds.  `11a_33`
also became solved in every seed, but it is a retention identity rather than one
of the declared discovery identities.  No second declared discovery identity
was rescued.  The hard frontier `11n_107`, `10_71`, and `10_137` produced no
certified positive trajectory; even 256 simulations did not solve `11n_107` in
the two seeds whose residual progress admitted that tier.

**Decision:** transactional admission and exact retention pass, and the result
is materially better than the destructive one-shot learner.  The native-
discovery gate nevertheless fails because only one declared identity replicated.
Do not run the 20/50 schedule progression, sharing validation, 200-
representation comparison, fourth scientist, or CPU-32 experiment yet.  The
next repair belongs in positive-trajectory discovery for the hard frontier, not
in another policy dose: broaden exact representation-preserving search or add a
separately measured search escalation, then repeat this development gate before
opening an untouched panel.

Artifact:
`pgx-mcts-bench/artifacts/native-learning-gate-swindow-v2-20260806/report.json`.

### Sharing implementation audit repair, 2026-08-06

A line-by-line audit found defects that prevent the old sharing runs from being
used as evidence for the intended long protocol.

1. Receiver cost was computed from the compact portable witness.  That witness
   intentionally removes serial states that only shift the head or modify
   controller memory.  The solver charged those plies, but sharing admission and
   reporting did not.  The repaired implementation verifies the same portable
   proof and uses the complete receiver record length for `moves`.
2. The native optimizer was constructed before the option adapter was attached.
   Native policy loss nevertheless flowed through the adapter; its gradients
   accumulated because native `zero_grad` did not own them, and global gradient
   clipping included them.  Native updates now bypass a separately optimized
   adapter, clear every network gradient, and clip only parameters owned by the
   native optimizer.  Option updates similarly clear stale base gradients.
3. The long `braid-collaborative-scientists` runner attached the adapter but not
   the state-dependent gate used by the later sharing gate.  It now attaches the
   gate at probability 0.1 and trains route-gate, off-route KL, and off-route
   activation penalties.
4. The adapter and gate pooled the serial sequence only by mean and maximum,
   although the policy decision is head-relative.  Both now include the encoded
   head cell explicitly.  Old adapter checkpoints are migrated with the new
   columns zeroed, preserving their outputs exactly.
5. The long runner gave sharing one adapter update per training event.  It now
   uses `ceil(train_steps / 4)` sharing updates, and no-sharing arms receive the
   same number of extra native optimizer steps.  The frozen manifest records the
   dose and uses schema `collaborative-scientists-v3`.
6. Distillation admission compared a donation only with the receiver's current
   stochastic attempt.  If that attempt failed, an older, better receiver-native
   solution was forgotten.  Worse, the long runner computed a filtered option
   set but the trainer silently sampled from the entire replay, so a donation
   could still train policy after becoming stale.  Replay now keeps a persistent
   best native objective by representation and A:B ratio.  Only the best donated
   trajectory whose fully charged objective is *strictly* below that incumbent
   is an active policy target; equal and worse solutions are excluded and the
   check is repeated at every training event.  A stale trajectory remains only
   as a one-sided upper-bound label for the critic.

The verified historical solution identities are still useful: the proof replay
was correct.  Historical native move counts, capped objectives, and any strict
sharing admission that depended on the undercounted objective are not valid for
the corrected experiment.  Historical learned checkpoints describe the old
algorithm, whose native/share gradient paths were entangled; retain them only as
diagnostics.  Do not use their objective comparisons to accept or reject the
fixed sharing hypothesis.

The repair has regression coverage for a receiver route with an extra charged
head shift, native/option optimizer separation, positional head conditioning,
old-checkpoint migration, gated option isolation, and transactional resume.  It
does not yet implement target-conditioned option programs: the target remains a
deterministic shortest neutral head route followed by one certified edit, not an
arbitrary five-step tape/register/memory program.  That is a later architecture
experiment.

**Decision:** the code is ready for a corrected bounded multi-representation
gate, not for the 200-representation or CPU-32 experiment.  Recompute paired
receiver-native objectives and strict admissions, repeat the multi-witness gate
with exact solved-set intersections, and open the long arms only if the repaired
sharing treatment is non-inferior to its compute-matched native control.

### Corrected multi-witness gate v6, 2026-08-06

The repaired gate reused the frozen 17-identity `s-tape4-h5` panel, eight donated
training witnesses, held-out `12a_850`, seeds 20260950--20260952, 128 simulations,
eight final attempts per identity, sixteen update cycles, and the compute-matched
native control.  The only protocol changes were the audited implementation
repairs: every receiver ply is charged, native and option gradients are isolated,
the option controller has an explicit head-cell feature, and manifests identify
the run as v6.

| seed | sharing / control solved | sharing-only | control-only | lost from frozen before | charged capped L10 sharing / control | gate |
|---:|---:|---|---|---|---:|---|
| 20260950 | 13 / 6 | 8 identities | `12a_1199` | `11n_46`, `12a_1199` | 2,200 / 3,407 | fail |
| 20260951 | 17 / 6 | 11 identities | none | none | 1,634 / 3,364 | pass |
| 20260952 | 15 / 4 | 11 identities | none | `11n_46`, `12a_1199` | 1,998 / 3,757 | fail |

Mean target-witness transfer was 91.7% for sharing and 4.2% for native-only.
Mean final coverage was 15.0/17 versus 5.33/17, and mean charged capped L10 was
1,944 versus 3,509.3.  The held-out receiver-unsolved identity `12a_850` was
solved by sharing in all three seeds and by control in none.  Thus corrected
sharing produces a large, repeated transfer signal and improves the aggregate
objective; this is no longer a “sharing does nothing” result.

The preregistered gate nevertheless fails because exact retention is mandatory.
In seeds 20260950 and 20260952 the sharing scientist lost both frozen canaries
`11n_46` and `12a_1199`.  Seed 20260950 additionally had `12a_1199` as a
control-only final solve.  The failures are deterministic within the eight final
attempts: each lost identity was 8/8 before training and 0/8 after sharing.

A paired counterfactual evaluated the two lost canaries from both failed seeds
with the final option adapter enabled and bypassed.  Both modes solved 0/2 and
had capped loss 528 in both seeds.  Therefore the final residual is not directly
masking a retained base solution.  The base scientist itself has lost the
canaries, plausibly through native updates trained on adapter-influenced search
data.  Raising only the off-route gate penalty cannot repair this failure.

**Decision:** sharing transfer is demonstrated, but the long sharing arms and
CPU-32 remain closed because only one of three seeds retained the complete frozen
solved set.  The next repair should protect native base updates inside the
sharing learner with explicit canary rehearsal or transactional acceptance and
rollback, then repeat this exact frozen three-seed gate.  Do not redesign the
adapter or reduce its dose before isolating base retention.

Artifacts:

* `pgx-mcts-bench/artifacts/multi-witness-gate-v6-s-tape4-h5-20260806-summary/report.json`;
* per-seed directories `multi-witness-gate-v6-s-tape4-h5-seed20260950-20260806`
  through `seed20260952-20260806`; and
* failed-seed counterfactual directories
  `multi-witness-gate-v6-counterfactual-s-tape4-h5-seed20260950-20260806` and
  `seed20260952-20260806`.

### Superseding v9 gate rule: quality primary, exact retention secondary

The v6 decision above was intentionally conservative, but exact equality of a
stochastic neural policy's solved set is too strong as the primary research
criterion.  A scientist that loses two old fixed-seed solves but gains nine hard
solutions and substantially improves total capped L10 should not automatically
fail.  Exact retention is still valuable evidence of stability, so v9 reports
it, paired control-only identities, and exact final-set equality as secondary
criteria.

The v9 primary gate is:

1. for every receiver and seed, charged capped L10 across the complete frozen
   panel is no worse than its compute-matched native control; and
2. every active-witness distillation event reaches its registered canonical-route
   loss reduction before the hard optimizer-step cap. Individual steps may rise
   because the same update also protects off-route native behaviour.

The overall gate also requires at least one representation solved only by the
sharing treatment.  Thus quality and demonstrated transfer decide admission;
exact NN retention measures collateral drift and remains in every report.  The
stale/worse-solution repair is part of this protocol. Training and final
evaluation simulations are frozen separately. Historical v6--v8 numbers cannot
be relabelled as v9 because both the policy-target set and admission rule changed.

### Expanded stale-aware preflight and simulation-dose result

The corrected preflight expanded to 17 representations: eight transfer targets,
held-out `12a_850`, five canaries, and deliberately stale `12a_1199`, `10_124`,
and `12a_1203`. Native refresh also made `10_126` and `11n_119` stale. All five
stale donations produced zero adapter updates, confirming that policy training
uses only a strictly better donated receiver-native objective.

At 128 learning simulations, sharing tied control at 11/17 but lost charged
capped L10 3,280 to 2,512. At 64 learning simulations it solved 10/17 versus
4/17 and narrowly won 3,656 to 3,671. A fixed-checkpoint evaluation-dose sweep
then separated learning search from evaluation search:

| learning simulations | evaluation simulations | sharing / control solved | capped L10 sharing / control |
|---:|---:|---:|---:|
| 64 | 32 | 12 / 4 | 3,377 / 3,794 |
| 64 | 64 | 10 / 4 | 3,656 / 3,671 |
| 64 | 128 | 14 / 6 | 3,263 / 3,412 |
| 64 | 256 | 17 / 7 | 2,767 / 3,261 |
| 128 | 32 | 9 / 10 | 3,762 / 2,771 |
| 128 | 64 | 10 / 10 | 3,452 / 2,716 |
| 128 | 128 | 11 / 11 | 3,280 / 2,512 |
| 128 | 256 | 11 / 13 | 3,178 / 2,164 |

The 64-trained sharing checkpoint therefore has a repeatable coverage advantage
at all four evaluation doses. It is not simply an under-searched checkpoint.
However, its summed L10 on the identities both arms solve is worse at every dose;
sharing wins capped loss by covering more tasks, not by shortening common
solutions. The 128-trained sharing checkpoint loses at every evaluation dose.
The full gate consequently froze 64 simulations for learning and 128 for final
evaluation.

### v9 split-budget seed and stop decision

The clean v9 rerun of seed 20260950 started again from the untouched rung-18
`s-tape4-h5` checkpoint. It used eight final attempts, sixteen update cycles,
and four CPU evaluation workers. Sharing solved 14/17 versus control 4/17 and
won charged capped L10 2,392 to 3,721. The exact intersection was `10_100`,
`10_124`, `10_152`, and `12a_1203`; sharing-only contained ten identities and
control-only was empty. On the intersection, however, sharing's summed objective
was 465 versus control's 289. Sharing also lost frozen solve `12a_1199`.

Stale filtering remained correct. `10_126` was rejected at donated/native
objectives 68/66, while `11n_119` was rejected at 162/110 and later 162/85.
Eleven of twelve active distillation events reached their registered 10% route-
loss reduction. The first `11a_15` event exhausted all sixteen adapter steps,
moving only from 6.9719 to 6.8420 instead of the target 6.2747. The seed therefore
fails the preregistered primary gate even though its aggregate coverage is
promising.

Stop the sequence at this failure: do not average it away with seeds 20260951
and 20260952. The 30--50-representation pilot, capacity-growth branch, 200-item
five-arm comparison, and CPU-32 rental remain closed. The next local experiment
should tune adapter learning rate and maximum steps under an explicit matched-
compute budget, and must show target attainment for every active event without
solved-set degradation before this same three-seed gate is reopened.

### Block-balanced sharing gate v10, 2026-08-06

The v9 event-level stopping rule was the wrong scientific test. Canonical-route
loss only measures probability assigned to one translated controller route; it
does not measure eventual solve probability or solution cost. A receiver may be
distracted by an individual foreign method even when a collection of better
methods helps on average. Requiring every donation to reduce route loss by 10%
therefore vetoed the hypothesis before measuring it.

Protocol v10 uses schema
`block-balanced-compute-matched-option-adapter-sharing-v10` and makes these
changes:

1. wait for at least ten distinct active, strictly superior receiver-native
   witnesses;
2. train at a scheduled ten-cycle boundary, not after every donation;
3. sample four canonical-route positions per witness on every adapter step, so
   long routes cannot dominate the block;
4. apply a fixed 16-step adapter dose and give control enough extra native
   batches to match or exceed the adapter's state-example count; and
5. use paired complete-panel capped receiver-native-ply objective as primary.
   Per-witness route loss, exact retention, solved-set differences, and
   common-success cost are reported diagnostics, not individual vetoes.

A tiny two-witness plumbing smoke and a 13-witness economical preflight verified
block scheduling, balanced sampling, stale-witness rechecking, and compute
matching. The full confirmatory run then used 25 representations, 19 registered
training targets, six non-target retention canaries, 64 training simulations,
128 evaluation simulations, eight final attempts per representation, four
native-refresh attempts per cycle, and fresh paired seeds 20261000--20261002.
Every seed completed one real sharing block. The canary start states participated
in the off-route preservation loss, so this subset is not an unseen held-out set.

The table reports the historical `delta = capped loss(sharing) - capped
loss(control)`, where the move term included receiver-native plies. Negative is
better for sharing, but these values are not semantic `L10`.

| seed | sharing / control solved | sharing-only / control-only | capped-loss delta | mean route-loss reduction |
|---:|---:|---:|---:|---:|
| 20261000 | 16 / 21 | 2 / 7 | +708 | 0.90% |
| 20261001 | 12 / 17 | 0 / 5 | +780 | 1.07% |
| 20261002 | 16 / 16 | 5 / 5 | -270 | 0.84% |

All active witnesses improved canonical-route loss in all three seeds. That is
useful evidence that the fixed-dose block optimizes its teacher-forced target.
It is also the decisive counterexample to using route loss as the gate: external
quality still lost in two of three paired seeds. Mean complete-panel delta was
+406 and median delta was +708. The result was not confined to training targets:
their deltas were +277, +762, and -82, while non-target canary deltas were +431,
+18, and -188. Thus both subsets favoured control in two of three seeds. A future
generalization claim still needs a separate identity-disjoint panel unused by
donation, replay, preservation loss, and model selection.

Sharing did produce real alternative solutions: the union of sharing-only
identities was `10_152`, `10_159`, `11a_231`, `12a_1199`, and `12a_1255`.
However, those gains were outweighed by control-only and lost native solves.
This is not evidence that all knowledge sharing is impossible; it is evidence
that the current unconditioned option-route adapter does not improve expected
external performance under a balanced 16-step dose.

**Decision:** v10 fails its preregistered multi-seed gate. Keep the 30--50-item,
200-item, and paid CPU-32 sharing runs closed. Do not rescue the result with a
larger dose merely because training route loss falls. The next bounded research
question is selective applicability: predict when a donated route helps this
receiver and representation, or detect conflicting native/share gradients,
while retaining the same external paired endpoint and compute matching.

Artifacts:

* `pgx-mcts-bench/artifacts/multi-witness-gate-v10-confirm-summary-20260806/report.json`;
* per-seed directories `multi-witness-gate-v10-confirm-seed20261000-20260806`
  through `multi-witness-gate-v10-confirm-seed20261002-20260806`; and
* economical dose preflights
  `multi-witness-gate-v10-block10-preflight-seed20260981-20260806` and
  `multi-witness-gate-v10-block10-dose16-seed20260981-20260806`.

### Semantic-cost sharing gate v11, 2026-08-07

The solver-independent objective is now enforced in code. A solution's `moves`
is the number of verified portable semantic braid edits. Head travel, tape and
register operations, scanning, and memory-state changes remain real receiver
compute: they consume native episode/internal limits and are reported as
`native_plies` and `internal_plies`, but do not enter `L_A:B`. Translation now
hard-fails unless donor and receiver agree exactly on `(cc, semantic_moves)`.
The witness bank was rebuilt under schema
`certified-semantic-collaboration-witness-bank-v2`; it contains 25 certified
witnesses and has hash
`e76b6b7911690b2e4b6bab9c3ef4883dcb21d3a7cac2e3754d9c1efcb81117a5`.

The remaining-semantic-`L` input is a function-preserving migration. Across the
real rung-18 `s-window-128`, `s-tape4`, and `s-w11-128` checkpoints, the maximum
absolute difference in policy logits, scalar value, solve logits, crossing head,
and moves head was exactly zero on three representations and three ratios.
Paired eight-simulation action sequences were also identical. The new `s-tape4`
critic was then trained on deterministic instances from actual ladder rungs
0--9 rather than unrelated table knots. Eight of ten identities supplied both
positive and censored outcomes; all ten final curves were monotone and
budget-sensitive. Promoted-rung solve rate changed from 7/8 to 8/8, so the
rollback guard accepted the checkpoint.

Two implementation defects surfaced before interpreting the preflight:

1. one valid witness, `11n_119`, could not be routed within the receiver's
   five-internal-action option cap, but the cycle scheduler still indexed it as
   trainable. Scheduling now uses only successfully translated targets and
   reports all unroutable targets;
2. the fresh option adapter inherited `s-tape4`'s conservative native rate
   `5e-5`. Its optimizer is now independent at `1e-3`. Under the same seed and
   16-step dose, mean canonical-route loss reduction rose from 0.006% to 0.757%;
   all 12 routable witnesses improved and none regressed.

The corrected economical preflight used 25 representations: 13 registered
targets, six non-target preservation canaries, and six generalization identities
excluded from donation, native replay, and preservation loss. It used eight
training simulations, 16 evaluation simulations, two paired attempts, ten
cycles, two native-refresh games per cycle, and one balanced 16-step sharing
block. Sharing and control both solved only `10_124`; sharing found cost 51 and
control 57, so complete-panel capped semantic `L10` was 6,387 versus 6,393.
Neither arm solved a target or untouched generalization identity. Exact retention
and final solved sets matched, but the overall gate failed because there was no
sharing-only identity.

A fixed-checkpoint 16/32/64 simulation sweep tested whether the adapter needed
more search. At 64 simulations and a new evaluation seed, sharing solved
`10_100` and `10_124`, control only `10_124`, and capped loss was 6,173 versus
6,387. `10_100` was a preservation canary, not a donation target or untouched
generalization item. Because 64 simulations were selected after this observation,
that row is exploratory. A fresh 64-simulation/four-attempt preflight did not
replicate it: both arms solved only `10_124`, tied capped loss at 6,387, and both
lost the frozen `10_100` solve. Route loss still improved for all 12 routable
witnesses by a mean 0.849%, but no target transfer appeared.

**Decision:** do not run the three confirmatory seeds, 30--50-item pilot, 200-item
arms, or paid CPU-32 sharing experiment. The result does not prove that sharing
cannot work over a longer horizon: it shows one post-hoc high-search rescue that
failed a fresh-seed replication. The next sharing experiment must change a
registered learning variable—longer interleaving, repeated blocks, or a
target-conditioned option policy—rather than merely increasing final MCTS after
seeing the outcome. Static/adaptive no-sharing engineering can proceed
independently.

Artifacts:

* `pgx-mcts-bench/artifacts/semantic-budget-migration-v11-k3-seed20261130-20260807/report.json`;
* `pgx-mcts-bench/artifacts/semantic-critic-v11-s-tape4-early10-seed20261120-20260807/report.json`;
* `pgx-mcts-bench/artifacts/semantic-sharing-v11-preflight-s-tape4-seed20261150-lr-fixed-20260807/report.json`;
* `pgx-mcts-bench/artifacts/semantic-sharing-v11-dose-s16-32-64-seed20261160-20260807/report.json`; and
* `pgx-mcts-bench/artifacts/semantic-sharing-v11-fresh64-preflight-s-tape4-seed20261170-20260807/report.json`.

### Common structural-budget equivalence audit, 2026-08-07

The prediction-independent v5 budget protocol was audited separately from
learning. The panel contained 100 distinct braid representations generated from
the mastered scrambled prefix of the ladder, with a disjoint 20-representation
calibration split. The three scientists were the promoted independent K=3
roster: `s-window-128`, independent `s-tape4`, and `s-w11-128`. Every
representation received two paired attempts under `L1000`. Calibration selected
the lowest registered dose, 32 simulations, after portfolio coverage reached
20/20.

Direct-global search and structural-first/global-restart search both solved
exactly 92/100 representations. Their solved sets were identical and their
capped `L1000` was exactly 639,569 in both arms. All 600 paired final outcomes
and every solved cost matched. Native action sequences matched for 598/600
pairs; the two differences were unsuccessful paths with the same final failure
classification, so action equality remains a diagnostic rather than a scientific
endpoint.

The structural probe was not economical. Thirty-six capped failures restarted,
raising scheduled network evaluations from 578,424 to 617,133, a 6.69% increase.
Thus the removal of `2 * L_predicted` succeeds as a fairness/equivalence repair,
but the replacement cap is not admitted as a compute optimization. Keep
objective budgeting opt-in and disabled in the first five-arm comparison. The
artifact is
`pgx-mcts-bench/artifacts/structural-budget-equivalence-k3-100-seed20261600-20260807/report.json`.

### Transactional positive-acquisition gate v3, 2026-08-07

The approved next gate tested whether hard-frontier failure was caused by search
breadth or by the 64-action episode clock. It used `L1000`, the same fixed
12-representation development panel and three new seeds. The promoted
`s-window-128` checkpoint first reproduced its rung at 12/12. Discovery then used
registered `(simulations, native action horizon)` tiers `(64,64)`, `(128,96)`,
`(256,128)`, and `(512,128)`. Residual progress was required after the first
tier; the final two tiers additionally required the best residual to cross a
75%-of-initial near-solve threshold. Training remained certified-success-only,
exposure-capped, and transactional.

| seed | initial solved | final solved | gained | lost | capped `L1000` initial / final | accepted updates |
|---:|---:|---:|---|---|---:|---:|
| 20261620 | 4/12 | 6/12 | `11a_33`, `12a_981` | none | 179,560 / 149,459 | 1 |
| 20261621 | 4/12 | 5/12 | `11a_33` | none | 179,560 / 162,509 | 1 |
| 20261622 | 4/12 | 6/12 | `11a_33`, `12a_981` | none | 179,560 / 147,457 | 2 |

Exact retention and non-inferior capped objective passed in every seed, but only
`12a_981` replicated as a declared discovery rescue; two were required. Final
panel solve rates were 50.0%, 41.7%, and 50.0%, below the preregistered 70%
minimum.

The detailed transactions locate the remaining failure. `10_149` already
supplied four certified positive trajectories in every seed at `(64,64)`, but
each attempted 24-step consolidation was rejected because it lost `11a_26` or
worsened capped `L1000`. More search cannot fix that case. On the hard frontier,
`11n_107` stayed at residual length 11 under both `(64,64)` and `(128,96)`, just
outside its registered near-solve threshold 10; `10_71` and `10_137` made no
residual progress. Consequently no representation qualified for the 128-action
tiers.

**Decision:** positive acquisition v3 fails. Keep the K=4 no-sharing smoke,
sharing gate, objective-mixture ablation, 200-representation comparison, and
cloud run closed. The next valid repair is inside success consolidation—for
example transactional selection across 1/2/4/8/... optimizer steps or an
explicit canary-preserving update—not an unregistered relaxation of the search
threshold. The artifact is
`pgx-mcts-bench/artifacts/native-learning-gate-swindow-v3-horizon-L1000-20260807/report.json`.

### Continual portfolio-progress criterion, 2026-08-07

Exact identity retention is no longer the primary invariant for continual
learning. Training on new representations can degrade old tasks, while rehearsal
can erase a recent gain. That exchange is acceptable when the complete portfolio
improves. Reports now separate the current network from the permanent solution
bank: the former measures what the policy reproduces under a fixed evaluation
dose, while the latter keeps the best verified semantic solution ever found for
each representation.

The preceding `L1000` diagnostic illustrates the distinction. Its continual arm
had 20 identities in the permanent native-solution archive but reproduced 18/20
at final evaluation. It gained `11a_288` and `12a_878`, lost current reproduction
of `12n_820` and rehearsal identity `9_1`, and finished at 13/20 on the new panel.
The paired transactional arm also finished at 13/20. Both were below the required
70%, so that diagnostic remains non-decisive and does not open a longer run.

The corrected smoke uses semantic `L10`. A single empirical failure cap is frozen
before learning as the maximum verified `L10` on the registered calibration
panel. Both arms evaluate exactly the same complete old-plus-seen portfolio.
Every ten rounds, the current network may retain the block when total solved count
does not fall and capped portfolio `L10` does not rise; at least one block across
the run must improve strictly. A regressing block receives targeted recovery. If
recovery fails, the network and optimizer return to the block-start state, but
the permanent best-solution bank is preserved.

Calibration selected 64 simulations at 8/10 coverage and froze the empirical
cap at `L10=85`. The paired local smoke uses four evaluation attempts,
`F_new=5`, `F_old=1`, 24 optimizer steps per iteration, and 10-round blocks. Its
registered artifact is
`pgx-mcts-bench/artifacts/portfolio-progress-smoke-swindow-seed20261720-20260807`.

The smoke completed in 70 minutes. Both block-progress decisions passed without
recovery. Block 1 kept 15/16 current solves and reduced capped `L10` from 807 to
785. Block 2 moved from 17/26 to 18/26 and reduced capped `L10` from 1,579 to
1,537. Replay exposure was balanced at 45,713 positive and 46,080 negative
positions; 2,304/46,080 failures (5.0%) were budget-censored.

This validates the aggregate retention mechanism, but not the learner for a
longer flight. Final block-progress coverage was 12/20 on NEW, 6/6 on old
rehearsal, 3/10 on held-out, and 0/4 on hard stress. It retained the exact initial
NEW solved set and improved NEW capped `L10` by 35, but found no net new current
solve. The transactional diagnostic reached 13/20 NEW by adding `11a_288`. On the
identical 26-item current-network portfolio, block-progress/transactional scored
18/19 solves and capped `L10` 1,537/1,523. Their lifetime banks scored 19/20 and
1,424/1,412. Block-progress won held-out capped `L10` by 22, but lost the sole
initial hard-stress solve; both primary NEW solve rates remained below 70%.

**Decision:** the portfolio criterion replaces exact canary retention, but this
training configuration fails its readiness gate. Keep the 50-representation,
five-arm, 200-representation, and cloud runs closed. Before a fresh gate, use a
frontier-matched calibration panel to select evaluation dose and the empirical
cap, then diagnose whether 128 or 256 simulations exposes latent coverage in the
saved final checkpoints. Any such dose sweep is exploratory; confirmation must
use fresh paired seeds and a frozen panel.

### Joint budget-aware pretraining and rewind audit, 2026-08-07

The proposed replacement `s-window-128` keeps the measured two-residual-block,
width-32 controller rather than changing capacity and curriculum at once. It
adds remaining semantic `L` as a nineteenth observation channel. The four
width-64 auxiliary members predict `p(solve)`, crossing changes, and semantic
moves; the solve loss may train the shared encoder/body, while cost regression
uses the shared features without pushing its noisy gradients back into the
controller. MCTS remains on the established scalar value until the factorized
critic passes a separate calibration gate. The H5 ablation adds a twentieth
channel for the remaining five-step internal-action budget.

Migration from the independent rung checkpoint was exactly function preserving:
the maximum absolute change was zero for policy logits, scalar value, solve
logits, crossing predictions, and move predictions on three representations and
both `L10` and `L1000`. A first aggressive ablation balanced failures inside the
ladder and also let cc/moves regression reshape the encoder. It still promoted
on rungs 0--9, but required 14 iterations at `T(2,5)+8` and ended with two 3/4
retrospective cells. This is a useful warning: successful rung promotion alone
does not establish continual retention.

The conservative v2 restored the historical natural replay distribution and
detached cc/moves loss from the encoder. It cleared all ten rungs at 100% held-out
solve rate in two iterations each, with every retrospective cell at 100%. A
two-objective capped block then produced 20/20 monotone training curves and
20/20 monotone disjoint curves. Its paired easy-prefix panel stayed at 80/80
solves and improved capped `L10 + L1000` from 57,050 to 57,036. The earlier
single-`L10` block is rejected: despite monotone curves, it worsened the paired
objective by teaching unnecessary crossing changes on unknots.

The untouched gate overturned the apparent success. Rewinding the rung-21
network through rungs 0--9 reduced its original promoted-rung solve rate from
12/12 to 2/12 before capped calibration; calibration recovered only 3/12. On
the fixed 400-attempt `L10` validation grid, the original checkpoint solved 16
attempts and the new network only 3. The latter had monotone curves and AUC
0.955, but no trained-only solve, Brier skill below zero, and ECE 0.268. Under
`L1000` it solved 6 attempts against the baseline's 14. Thus an apparently good
critic can coexist with a badly degraded solver, and an easy-prefix retention
panel cannot protect a later mastered frontier.

The controls locate the cause. Random initialization learned all first five
rungs, needing ten iterations only on `unknot+6`; it later forgot that rung in a
four-attempt retrospective. The warm H5 model cleared all five in two iterations.
So neither network depth nor the added channels prevents fast learning. The bug
is the training trajectory: a promoted rung-21 checkpoint must not be restarted
at rung 0, because the new mixture then excludes rungs 10--21. Code now refuses
that rewind unless it is explicitly requested as an ablation, and an internally
passing checkpoint is not called admitted until an untouched source-disjoint
gate passes.

**Decision:** reject both rewind-trained checkpoints. For the immediate roster,
retain the existing independent calibrated `s-window-128`; its 60-identity
critic curriculum used promoted-rung rehearsal and a policy/value preservation
teacher and already passed the source-disjoint gate with AUC 0.947, Brier 0.0266,
ECE 0.0166, no held-out solve loss, and no rung regression. A genuinely new
network should be initialized randomly and climb the whole ladder forward once
with the new channels, plus explicit old-rung rehearsal. Do not add residual
blocks or width until that forward curriculum reaches a measured capacity wall.

The second-objective confirmation strengthens the immediate-roster decision.
On the fixed 400-attempt `L1000` panel, the independent calibrated checkpoint
solved 18 attempts against 14 for the original rung-18 checkpoint. There were
four calibrated-only solves and no original-only solve. Both reproduced the
promoted rung at 12/12 with identical measured cost. The calibrated critic had
AUC 0.964, Brier 0.0323, Brier skill 0.248, ECE 0.0273, and monotone curves on
all 20 held-out representations. Thus this checkpoint is admitted for both
`L10` and `L1000`; this does not admit the rewind-trained replacement.

The valid random-init follow-up used eight frontier games and exactly one fresh,
pinned rehearsal game from every cleared rung per iteration (`F_old=1`). It
cleared rungs 0--4, but `unknot+6` required 12 iterations and only 80.6% pooled
solve rate. It was temporarily 2/4 immediately after rung 2, recovered to 4/4
after rung 3, and ended at 3/4 after rung 4. The final five-rung two-objective
panel had 34/40 solved attempts before budget calibration. Capped training kept
that count at 34/40 but worsened capped `L10 + L1000` from 205,540 to 208,571;
the budget update was rolled back. This is a negative result for fixed
`F_old=1`, not evidence of insufficient depth. The next from-scratch curriculum
must use the already implemented aggregate portfolio guard: retain a block only
when total solved attempts do not fall and capped objective does not rise;
target recovery at regressions and restore the block start if recovery fails.
Exact per-rung retention remains a reported secondary measure.

Artifacts:

* `pgx-mcts-bench/artifacts/joint-pretrain-warm-v2-prefix10-20260807/report.json`;
* `pgx-mcts-bench/artifacts/joint-pretrain-controls-prefix5-20260807/report.json`;
* `pgx-mcts-bench/artifacts/joint-pretrain-warm-v2-prefix10-20260807/heldout-L10.json`;
* `pgx-mcts-bench/artifacts/joint-pretrain-warm-v2-prefix10-20260807/heldout-L1000.json`;
* `pgx-mcts-bench/artifacts/joint-pretrain-warm-v2-prefix10-20260807/existing-calibrated-heldout-L1000.json`;
* `pgx-mcts-bench/artifacts/joint-pretrain-scratch-f-old1-prefix5-20260807/report.json`.

### Paired roster-readiness gate, 2026-08-07

The approved sequence stopped before the adaptive/static/sharing arms, at the
roster gate. The implementation now separates the remaining-`L` observation
feature from hard objective censoring: the feature is present, no predicted
objective cap is used, and the only termination limit is the common 128-action
horizon. Evaluation uses four stochastic MCTS attempts per representation with
Dirichlet root noise and temperature zero. A representation-attempt uses exactly
the same seed for every scientist and simulation dose. Final evaluation also
restores the frozen action horizon and solve calibration from the run manifest.

Two defects were found before the final gate. First, the original "frontier"
selector sampled the complete 3--12-crossing table and produced a stress panel on
which all four scientists were at or below 1/12 at 64 simulations. The selector
now takes a frozen structurally easiest eligible band before forming four strata.
Second, the first four-attempt implementation was deterministic, so every knot
was either 0/4 or 4/4; those labels were pseudoreplicates. Preliminary roster
screens remain useful only for candidate selection, not as critic evidence.
Nested remaining-budget migration for the cyclic and triad architectures was
also repaired and checked as a zero-column, function-preserving migration.

The final source-disjoint K=3 roster was `s-window-128`,
`s-cyclic-tape8-192`, and independent rung-18 `s-head-128`. The 12-item
calibration panel was drawn from the easiest 96 eligible table representations
after excluding the historical BASE/NEW70, critic/solver banks, and all
outcome-observed screening panels. It included seven 3-strand and five 4-strand
representations. The decisive check used `L1000`, 256 simulations, four paired
stochastic attempts, no objective cap, and a 128-action horizon.

| scientist | representations solved | attempts solved | raw AUC | raw Brier | raw ECE |
|---|---:|---:|---:|---:|---:|
| `s-window-128` | 8/12 (66.7%) | 29/48 (60.4%) | 0.881 | 0.434 | 0.485 |
| `s-cyclic-tape8-192` | 7/12 (58.3%) | 28/48 (58.3%) | 0.886 | 0.229 | 0.283 |
| `s-head-128` | 3/12 (25.0%) | 7/48 (14.6%) | 0.981 | 0.600 | 0.672 |

These are pre-recalibration critic metrics. Calibration and the disjoint
24-representation confirmation were correctly skipped because no scientist met
the required 70% representation coverage. The portfolio union was only 8/12 and
the intersection was 3/12. `s-window-128` alone supplied the sole marginal
identity, `9_45`; both other solved sets were subsets of its solved set. Thus this
is not merely a one-item miss by the portfolio: the proposed population has no
measured marginal coverage at the maximum dose.

The run scheduled 2,973,490 network evaluations (4,737,024 allocated upper-bound
evaluations). Screening also rejected the original independent `s-tape4` and
`s-w11-128` at 3/12 each at 128 simulations, `s-paint4` at 4/12,
`s-scan-gru` at 0/12, and the migrated triad at 4/12 on the development panel.
Those screen rates are engineering diagnostics because candidate selection saw
their outcomes.

**Decision:** the 30--50-representation no-sharing arms, sharing admission,
200-representation comparison, and cloud run remain closed. No training arm was
started. More MCTS is not the next repair: window and cyclic were flat from 128
to 256 on the preceding stochastic sweep, while the exactly paired maximum-dose
confirmation still failed.

There are now two honest forks. For the method-comparison paper, preregister a
source-disjoint 3-strand-only benchmark of at least 100 representations and keep
4/5-strand knots as a separately reported stress endpoint; do not call it a
general knot frontier. This still needs a third scientist with measured marginal
coverage. For the broader programme, forward-train a new independent third
controller through the full ladder with the soft remaining-budget feature,
portfolio guard, and explicit 4-strand curriculum, then repeat this exact gate.
Do not spend the CPU-32 budget or widen/deepen networks until that forward
curriculum distinguishes a capacity wall from missing training coverage.

### Replacement third scientist: `s-strand-graph-128`, 2026-08-08

`s-head-128` is retired from the intended collaboration roster. It remains a
historical control, but its 3/12 result and zero marginal identities do not justify
paying for it in a long arm. The replacement candidate is
`s-strand-graph-128`. It is implemented in `pgx-mcts-bench`, but it is not yet an
admitted scientist: the architecture and training plumbing are verified; forward
curriculum and held-out coverage are still unmeasured.

Simply forcing a scan is not the new idea. `s-scan-gru` already performs a virtual
full-necklace scan of centred local windows before every decision and scored 0/12
on the development panel. Its whole word is compressed into one final GRU vector;
it discards the feature at every potential edit site, and its shift logits are a
generic readout rather than scores of the positions those shifts reach.

The replacement compiles one deterministic scan of the current closed braid into
a crossing graph. Every occupied word position gets four pointers: previous and
next crossing along each of the two physical strands passing through it. Closure
arcs are followed exactly, so a strand ending at bottom height `h` continues at
top height `h`. The learned controller then has:

1. width 96 token features over the head-relative complete word;
2. five residual graph blocks, at cyclic word dilations 1, 2, 4, 8, and 16, that
   also exchange messages through both pairs of physical-strand neighbours;
3. a local edit head at the current serial head position;
4. a routing head whose logit for each left/right power-of-two shift reads the
   encoded position that shift would actually reach; and
5. shared-body `p(solve)`, conditional crossing-change, and conditional semantic
   move heads with the soft remaining-`L` input, solve-loss backpropagation, and
   budget-monotonic regularisation.

This is aimed specifically at four or more strands. Width is not the important
change: the network is told which distant crossings lie on the same physical
thread, while generator order and sign remain learned inputs rather than a supplied
knot invariant. Message passing preserves a distinct feature per crossing, so the
policy can coordinate several generator families without compressing the entire
braid into one vector before choosing where to work.

The proposed `K` forced shifts are implemented as a compiled perception option,
not as `K` MCTS actions. Here `K(x)=len(x)`: the adapter scans every crossing once
to build the four graph pointers, then the NN/search makes its first choice. The
scan is rebuilt after every semantic edit. Literal shifts only at episode start
would leave stale memory after the first rewrite; literal shifts inside the tree
would consume native horizon and repeat the same deterministic path in every
simulation. The compiled scan is still compute and must be reported in forward
latency/network evaluations, but it is neither a semantic move nor a search
branch. A literal embodied-scan version can remain a later ablation.

Local implementation measurements on the same four-strand word were:

| candidate | parameters | batch-1 forward |
|---|---:|---:|
| `s-head-128` | 140,308 | 0.331 ms |
| `s-scan-gru` | 215,700 | 0.816 ms |
| `s-cyclic-tape8-192` | 344,083 | 0.958 ms |
| `s-strand-graph-128` | 794,676 | 1.412 ms |

The first per-strand recurrent-slot prototype was rejected before retention: it
needed 1,200,276 parameters and 10.086 ms per batch-1 forward. Compiling the scan
therefore retains the structural bias at 14% of that latency. On a fixed batch of
four four-strand examples the retained network reduced policy cross-entropy from
3.236 to 0.000038 in 30 Adam steps, and an end-to-end one-game/eight-simulation
ladder smoke completed one optimizer iteration and promoted rung 0. These checks
establish wiring and fast trainability, not knot generalisation.

The forward-training decision gate is:

1. start independently rather than importing `s-head-128` policy weights;
2. interleave 2-, 3-, and 4-strand examples from the first training block, with
   simple torus/positive cases retained as rehearsal;
3. require the historical early-ladder behaviour to reappear: at least 80% solve
   rate on each promoted block, escalating simulations/iterations rather than
   accepting a negative-only stream;
4. evaluate on a new, frozen, source-disjoint mixed-strand panel, because the
   12-item panel above influenced this design and is now development data; and
5. replace `s-head-128` in the long roster only if the new scientist reaches at
   least 70% representation coverage and adds held-out identities beyond both
   `s-window-128` and `s-cyclic-tape8-192` at paired compute.

The word “capable” remains a hypothesis until steps 2--5 pass. The implementation
is now suitable for that test; it is not evidence that hard four-strand knots are
already solved.

#### Capacity and optimizer screen

The replacement is being screened as a family rather than as one arbitrarily
large network. Optimizer settings are now part of the persisted candidate
specification and are used by the ladder runner; previously the runner silently
constructed every AdamW optimizer at learning rate `1e-3`, weight decay `1e-4`,
and trained with batch size 32, so a nominal hyperparameter experiment would not
have changed the actual update.

| candidate | graph width x blocks | towers | parameters | LR / batch / steps |
|---|---:|---|---:|---:|
| `s-head-128` | -- | local | 140,308 | `1e-3 / 32 / 96` |
| `s-strand-graph-compact-128` | 64 x 3 | strand graph | 293,652 | `2e-3 / 32 / 96` |
| `s-strand-graph-128` | 96 x 5 | strand graph | 794,676 | `1e-3 / 32 / 96` |
| `s-strand-graph-local-128` | 128 x 6 | local + strand graph | 1,709,219 | `7.5e-4 / 64 / 128` |
| `s-strand-graph-wide-128` | 160 x 8 | strand graph | 2,971,348 | `5e-4 / 64 / 160` |

Every graph block is pre-normalised with `LayerNorm`, residual, and alternates
cyclic dilations `1,2,4,8,16` while also following the exact physical-strand
edges. The local-plus-graph arm is the learnability safeguard for a large model:
its local policy/value tower is an ordinary two-block serial controller, and its
graph contributions to policy and value have zero-initialised scalar gates.
Consequently its initial policy and value are exactly the local tower's outputs;
the auxiliary losses can train global features before the graph is allowed to
change behaviour. This tests progressive capacity without assuming that a
three-million-parameter controller can discover the elementary curriculum as
quickly as a small one.

All four graph sizes can fit a fixed batch of four-strand policy targets: under
their configured optimizers they reduced policy cross-entropy below `0.05` in
12--17 updates. That is only an optimizer/wiring test. The first 32-simulation
reinforcement-learning pilot was rejected because even `s-head-128` was below the
pre-registered 70% solve-rate floor; no architecture conclusion is drawn from it.

The calibrated admission test uses 128 simulations, eight self-play games per
cycle, 96--160 optimizer updates according to the table, and 12 independent
evaluation attempts for each of the three historical objective ratios. On seed
71, `s-head-128` had 75% aggregate solve rate after the initial four-cycle cap,
but an extended run promoted at cycle 6 with 94.4% aggregate solve rate. Thus
delayed takeoff on an elementary stage is real. It does not explain the mature
rung-18 checkpoint's 3/12 source-disjoint coverage: that network had already
trained through 18 rungs, and its three held-out successes were all three-strand
while it solved none of five four-strand representations.

The compact graph arm promoted seed 71 after one cycle at 88.9%; seed 73 confirmed
at 100% after three cycles. Seed 72 is a useful instability check rather than a
hidden success: it recovered from 36.1% at cycle 2 to 77.8% at the four-cycle cap,
but failed the collapse/objective checks (`1:10` was 41.7% and mean `1000:1`
crossing changes were 0.333 against the allowed 0.25). The balanced
96 x 5 graph promoted seed 71 after two cycles at 80.6%, including 100% on the
`1000:1` slice and mean 0.25 crossing changes. Wide and local-plus-graph arms are
still running. None is admitted to the collaboration experiment until a clean
mixed six-stage artifact closes. Large arms are also charged for wall time and
network evaluations: capacity that requires several times the compute to clear
the elementary stage is a later growth target, not a fair replacement at the same
experiment budget.

The continuation also closes a historical promotion loophole. The general ladder
may leave a known-`u` rung when crossing-change quality has plateaued, because
moving forward can teach it more efficiently than grinding one rung. That is not
a valid architecture certificate: a controller that always solves a trefoil with
two crossing changes has not passed a one-crossing objective. The architecture
gate therefore disables plateau promotion whenever exact `u` is known; it still
permits plateau promotion for unknown-`u` sources. Evaluation is performed every
two training cycles, with 12 attempts per objective at each evaluation.

The first mixed continuation exposed a second protocol mismatch and is retained
only as a diagnostic. It had old successful episodes in replay but generated zero
fresh old-stage attempts, although the approved schedule sets `F_old=1`. By cycle
8 it still solved the `1000:1` trefoil slice, but used two crossing changes and 50
moves, while the `1:10` slice fell to 0%. Clean compact and balanced continuations
therefore restart from their promoted stage-0 checkpoints and generate one fresh
MCTS attempt for every cleared stage in every cycle. Adding rehearsal only after
degradation would not be a valid repair experiment.

The clean `F_old=1` compact continuation also shows that rehearsal alone is not
the complete answer. Its first `T(2,3)` evaluation at cycle 2 solved 12/12 for
`10:1` but 0/12 for both endpoint objectives. The long balanced-graph process
later capped the trefoil stage after 12 cycles at only 33.3% aggregate SR and
mean 25 crossing changes. That process had loaded the runner before the final
success-only policy update, so it is retained as a superseded diagnostic rather
than an admission result. It provides no reason to prefer the graph model over
the cylinder for the next corrected broad-table curriculum.

Primary artifacts:

* `pgx-mcts-bench/artifacts/strand-architecture-first-stage128-seed71-20260808/`;
* `pgx-mcts-bench/artifacts/strand-architecture-first-stage128-compact-confirm-20260808/`;
* `pgx-mcts-bench/artifacts/strand-architecture-first-stage128-shead-long-seed71-20260808/`;
* `pgx-mcts-bench/artifacts/strand-architecture-mixed128-compact-rehearsal-seed73-20260808/`;
* `pgx-mcts-bench/artifacts/strand-architecture-mixed128-balanced-rehearsal-seed71-20260808/`;
* `pgx-mcts-bench/artifacts/frontier-roster-k3-paired-max256-source-disjoint-L1000-seed20261880-20260807/report.json`;
* `pgx-mcts-bench/artifacts/frontier-roster-k3-stochastic-source-disjoint-L1000-seed20261880-20260807/report.json`; and
* `pgx-mcts-bench/artifacts/frontier-roster-triad-stochastic-band96-L1000-seed20261840-20260807/report.json`.

### Transactional collaboration runner v7, 2026-08-08

The next three-arm runner is implemented and has passed an engineering smoke;
this is not yet evidence that sharing improves knot solving.

Every ten-round training block now has a bounded adaptive `F_old` dose over the
entire block. Half of the dose prioritizes representations degraded by the last
certificate, one quarter is recent, and one quarter is inverse-exposure sampled.
For each scheduled old task, replay receives its permanent best native solution
when one exists plus one fresh MCTS attempt. A deterministic rotating portfolio
is evaluated with identical seeds immediately before and after training. The
update is accepted only when portfolio solves do not decrease and complete
capped `L1000` does not increase; otherwise network and optimizer state roll back,
while replay and the best-solution bank remain available for the next attempt.

The direct-sharing arm has no separate policy adapter. A translated donation
enters native successful replay only when exact receiver replay proves that it is
strictly better than the receiver's best archived native objective for that
representation and ratio. Stale, equal, or inferior donations cannot train the
policy. Because a batch of 32 with four positions per episode has only eight
episode slots, a positive sharing fraction now receives at least one shared
episode when eligible; the scientific run should use batch 64, where one slot is
6.25%, close to the intended 5% dose.

The retention probe now repeats paired attempts per scientist/task/ratio until
the declared minimum attempt count is actually met. This matters during the
first blocks, when the solo arm has seen fewer distinct tasks than its nominal
panel size. The corrected two-round `v7b` smoke exactly matched all three arms:

| compute category | sharing | no sharing | solo |
|---|---:|---:|---:|
| qualification simulations | 12 | 12 | 12 |
| full-search simulations | 24 | 24 | 24 |
| rehearsal simulations | 24 | 24 | 24 |
| retention simulations | 24 | 24 | 24 |

Sharing and no-sharing selected the same final two representations
(`12a_16`, `11a_330`) and both portfolio guards accepted. The compute-matched
solo arm used three times the search simulations and three times the optimizer
steps in its single network. An earlier two-round wiring smoke also exercised a
real superior donation: `s-window-128` donated a receiver-valid witness that
improved `s-tape4` from `L1000=3009` to `2008`; an equal-cost donation to the
cyclic scientist was correctly rejected.

Primary artifacts:

* `pgx-mcts-bench/artifacts/collaboration-v7b-engineering-smoke-sharing-direct-seed20262011/`;
* `pgx-mcts-bench/artifacts/collaboration-v7b-engineering-smoke-no-sharing-seed20262011/`;
* `pgx-mcts-bench/artifacts/collaboration-v7b-engineering-smoke-solo-matched-seed20262011/`; and
* `pgx-mcts-bench/artifacts/collaboration-v7-engineering-smoke-sharing-direct-seed20262010/`.

The scalable cylinder candidate has now passed the first two of the three
scientist-admission conditions. After one verified `P(4,5)#0` witness was
success-only distilled into each independently trained seed, seeds 71--73 all
completed the six-stage 2/3/4-strand curriculum. Every stage was 12/12 at 256
simulations and attained the exact known crossing number; all three also learned
the unseen `P(4,7)#0` frontier natively without another donation. The witness
extractor itself was corrected to reproduce evaluation RNG state and to use
temperature-zero route search, so a reported evaluation solve is replayable as
a donor trajectory.

The source-disjoint comparison is now complete and negative. On a fresh panel
of eight 3-strand and four 4-strand table representations, window and cyclic
each solved 9/12 at both 256 and 512 simulations and had identical solved sets.
The cylinder improved from 3/12 at 256 to 6/12 at 512, but every cylinder solve
was already in that common established set. Its raw solve AUC was 0.325 and
0.318, versus 1.000 for cyclic; monotone scale calibration cannot repair that
ordering failure. Thus the cylinder misses the 70% coverage floor, adds no
marginal identity, has an unusable adaptive-scheduling critic, and costs much
more wall time on full-horizon failures.

The 100--200 representation pilot, sharing comparison, and paid run remain
closed. The next valid repair is forward training on a broad table-knot
curriculum with verified positive acquisition and separate held-out critic
labels. Increasing MCTS again or widening the network does not address the
observed generalisation failure.

Primary artifact:

* `pgx-mcts-bench/artifacts/frontier-roster-conv-cylinder-band64-L1000-seed20262200-20260808/`.

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
