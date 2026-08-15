# Scientist collaboration: active protocol

## Status

This is the clean protocol for the next investigation. No historical checkpoint
is admitted as a vNext scientist, and no old leaderboard is evidence for this
protocol. The detailed pre-reset history remains in
[`../archive/pre-semantic-moves-v1/research/16-scientists-collaboration-full.md`](../archive/pre-semantic-moves-v1/research/16-scientists-collaboration-full.md).

The scientific objective is always

`L_AB = A * crossing_changes + B * semantic_moves`.

Head shifts, tape writes, finite-state changes, and other controller-local actions
consume an internal-step budget but are never charged as semantic moves.

## Selected three-scientist roster

| Name | Information path | Main hypothesis |
|---|---|---|
| `raster-axial` | local braid raster with shared masked axial blocks | Better strand geometry without fixed-strand alphabets |
| `cyclic-memory` | cyclic scan, persistent tape, and global objective conditioning | Can accumulate information outside a local window |
| `strand-graph` | compact shared strand/message blocks with a routed action head | Test strand transfer cheaply, then grow only after a capacity gate |

Names describe architecture only. Objective ratio, simulations per move, seed,
training dose, and checkpoint are explicit run fields.

`raster-routed` was tested as a possible replacement but was not admitted. It
solved 10/10 first-rung evaluations after six iterations while averaging 0.3
unnecessary crossing changes. An exact-state `F=8` continuation retained 10/10
feasibility but worsened mean crossing changes to 0.6. In the same matched gate,
`raster-axial` cleared the four-strand stage at 10/10 and optimal cost. Therefore
the first big experiment keeps `raster-axial`; `raster-routed` remains a separately
named capacity research candidate. “Raster-certified” means verifier-checkable
encoding, transitions, and witnesses—not a correctness claim about the network.

Every scientist receives the requested `log(A/B)`, remaining semantic objective
budget, remaining internal-step budget, and its architecture-specific observation.
The shared body feeds policy and three auxiliary outputs: crossing changes,
semantic moves, and `p(solve)`. The objective value is constructed from the first
two using `A` and `B`. Budget features reach the shared body and all heads through
skip/conditioning paths so `p(solve)` can genuinely depend on budget.

## From-scratch pretraining

All candidate scientists began from independent seeded initializations. The curriculum
starts with unknots, small torus knots, and simple controlled scrambles, then adds
source-disjoint mixed-sign braids and more strands. It samples exactly L10 and
L1000 at a 1:1 ratio. Foundation pretraining is deliberately neutral; target-
biased objective sampling is a later declared experimental variable.

Admission requires, on held-out representations:

1. at least 70% solve rate at a declared search allocation;
2. monotone `p(solve)` response to decreasing remaining budget in aggregate;
3. calibrated crossing-change and semantic-move heads;
4. successful replay after save/resume;
5. no portfolio regression after a rehearsal block; and
6. successful distillation of multiple strictly better donations without a paired
   solved-set or capped-objective regression.

## Adaptive compute and rehearsal

Adapt only at ten-round training-block boundaries.

- Start `F_native = 5`; if recent held-out acquisition is below 0.80, move through
  `8, 12, 16`. Do not silently change it within a paired block.
- Start `F_old = 1`; if the retention solved rate is below 0.80 or capped objective
  worsens, move through `2, 4, 8`. Select rehearsal tasks from lost solves and the
  worst objective regressions before uniform replay.
- Start at 64 simulations per move; if paired solve rate is below 0.70, move through
  `128, 256, 512`. Report compute at every level. Do not encode this number in the
  architecture name.
- Increase donation dose only after at least several eligible donations accumulate
  and a paired block does not regress. Reduce it immediately after regression.

No scientific upper bound is `2 * L_predicted`. Initial objective budgets come
from a representation-level empirical policy based on cheap global features and
the current verified solution bank, with a generous fallback cap. Local networks
may use predictions to order tasks, but not to censor the search that evaluates
them.

## Semantic-v2 sequential curriculum

The previous 24-representation readiness run was a frozen zero-shot transfer
test. It was useful diagnostically but was too early to be an admission gate:
even archived rung-18 `s-window-128` and `s-tape4` solve only 8/24 and 7/24
representations under its two-attempt, 64-simulation protocol. SV2 therefore
makes **one fixed representation one rung** and permits declared task-local
learning before evaluation.

The first prefix contains `6 + 6 + 3 + 3 + 3 + 3 = 24` representations:

1. six simple source representations;
2. the same six sources after two deterministic scramble moves;
3. three different simple sources after one scramble move;
4. three different simple sources after two moves;
5. three different simple sources after three moves; and
6. three different simple sources after four moves.

The `6+6+3+3+3+3` construction is provenance, not an ordering barrier. Static
order globally sorts all 24 representations by the auditable, outcome-blind score

`ACS = 10 * strands + 5 * exact_u + presentation_crossings`.

For a braid word, `presentation_crossings = len(word)`: every generator letter
is one crossing in that diagram. This is not necessarily the knot type's minimal
crossing number. All R24 sources have exact `u`; later groups retain exact or
certified-bound provenance explicitly.

All four arms use identical R24 native compute: 64 simulations per move,
`F_native=10`, eight self-play games and 96 optimizer steps per iteration, and
four evaluation attempts per objective. Only `F_old` is adaptive on R24. It
starts at one total rehearsal iteration per ten-rung block and moves through
`1, 2, 4, 8` when paired retention solve rate is below 0.80 or rehearsal worsens
complete capped cost. Failed and expensive retained tasks are rehearsed before
exposure-balanced tasks. The 200-representation group is the first group where
native iterations and simulations may adapt.

Evaluation-protocol correction (2026-08-12): the deployed R24 and initial R200
runner used temperature zero without root noise. Because resetting a fixed word
does not use the attempt seed, its four nominal attempts were identical
deterministic trajectories. Those runs therefore contain one effective attempt
per scientist, representation, and objective repeated four times; their
fractional aggregate coverage remains descriptive, but per-cell `EV4` solve
rates and adaptive thresholds must not be interpreted as four-sample estimates.
Protocol v3 uses four paired seeds with independent Dirichlet root noise and
batches those four searches. The manifest freezes
`evaluation_attempt_protocol`; the legacy deterministic mode remains available
only for exact reproduction.

The frozen groups are `24`, then `200`, then `400`, followed by further groups
of `400`. Group identity is outcome-blind and common to every arm. Static order
is recomputed separately inside each group. Adaptive arms reorder only inside
the current group; they never pull a task from a later group. Any learned
replacement for ACS must be trained on a separate scheduler-development stream
and frozen for every arm, not fitted from one treatment arm's outcomes.

The frozen R200 group retains the original ACS coefficients. Because its table
knots do not all have known exact unknotting number, its declared `u` feature is
the current certified unknotting-number upper bound:

`ACS_R200 = 10 * strands + 5 * certified_u_upper_bound + len(braid_word)`.

Here again `len(braid_word)` is the number of intersections in the selected
presentation. The source table's minimal crossing number is retained separately
and is not substituted for presentation length.

The post-R200 stream is now frozen at 2,700 total BASE representations: 1,639
compatible canonical table braids and 1,061 deterministic type-preserving
Markov variants. After removing the existing R200 identities, it forms six
R400 groups and a final R100 tail. Static ACS ordering is recomputed inside each
group, and cumulative prior banks retain R24 plus every completed earlier group.
Exact provenance, hashes, and the clean optimized four-arm R200 restart are in
[`../paper_partials/2026-08-12-optimized-four-arm-r200-and-large-banks.md`](../paper_partials/2026-08-12-optimized-four-arm-r200-and-large-banks.md).

A separate five-scientist oracle family tests whether explicit classical,
Alexander, and Jones information improves the raster controller, and whether
late, FiLM, or dual-tower fusion is best. It is an ablation of human knowledge,
not a replacement for the invariant-free arms. Its protocol is in
[`../paper_partials/2026-08-12-human-invariant-oracle-family.md`](../paper_partials/2026-08-12-human-invariant-oracle-family.md).

The dated interim paper evidence, including paired innovation versus portfolio
metrics, exact solved-set checks, capacity mutations, durable native-event
logging, and the planned causal L1000 objective-mixture ablation, is in
[`../paper_partials/2026-08-13-interim-r200-innovation-and-objective-ablation.md`](../paper_partials/2026-08-13-interim-r200-innovation-and-objective-ablation.md).

## Replay and sharing

Maintain permanent positive and negative episode banks with per-example usage
counts. Each loss requests its own batch:

- policy and cost heads use successful native or eligible donated routes;
- `p(solve)` uses both successful and failed attempts, stratified by ordinary and
  budget-censored failure;
- rehearsal samples representations approximately uniformly before sampling a
  useful position from the selected episode.

A donated witness is trained only when verifier replay proves it has lower
semantic `L_AB` than the receiver's own incumbent for that representation and
objective. Equal or worse donations are recorded as inferior and cause zero policy
updates. Sharing trains the ordinary policy; there is no permanent sharing-only
adapter or split policy brain.

At each block boundary, save a pre-update checkpoint, apply the declared native,
rehearsal, and donation doses, then run paired portfolio evaluation. Accept the
block only if solved-set size does not decrease and capped objective does not
worsen. Otherwise roll back and raise rehearsal/search according to the adaptive
schedule. Exact logit retention is reported, but is not a hard blocker.

Donation dose is an exact count rather than a replay fraction. `D=1,2,3` means
that every still-eligible donated witness receives one, two, or three controlled
optimizer exposures. Eligibility is rechecked before every exposure: the
translated witness must verify and must still have strictly lower semantic cost
than the receiver's best native solution for the same representation and
objective. Native/rehearsal updates are checkpointed before a separate donation
phase, so a donation-only regression can be rolled back without erasing native
learning.

Start `D=1`. Hold it until a block contains at least ten distinct eligible
donations. Raise it by one only after two consecutive donation blocks preserve
portfolio solved-set size and do not worsen either capped L10 or capped L1000.
Any paired regression lowers `D` immediately and resets the healthy-block count.
Canonical-route loss is diagnostic and never changes the dose by itself.

## Experiments

The sequential-learning comparison contains four arms:

1. three independent scientists, static ACS order, no sharing;
2. three independent scientists, adaptive evidence-backed order, no sharing;
3. arm 2 plus strictly-better verified donations; and
4. arm 1 plus strictly-better verified donations.

The adaptive scheduler ranks the remaining representations in the current group
by the minimum predicted L10 among the scientists. The proposing scientist must
supply a declared qualification attempt; a failed qualification is evidence too
and receives the failure cap. Scheduling coordination is not solution sharing:
in arms 1 and 2, no trajectory crosses a scientist boundary.

The synchronized implementation is `pgx-mcts-bench braid-sv2-coordinated`.
Adaptive qualification solutions remain native evidence owned by the proposer;
they are not discarded after task selection. Sharing updates are deferred to the
block boundary, where every active strictly better donation receives exactly
`D` ordinary-policy optimizer exposures. A paired donation-only transaction is
accepted only when portfolio coverage and capped cost are separately
noninferior for both L10 and L1000. Rejection restores network and optimizer
state while retaining native learning and replay provenance.

The production runner uses one persistent process per scientist. A paired
three-rung counterfactual established exact equivalence with the sequential
reference: selected order, event/controller state, and every final model tensor
matched. At rungs 10, 20, and 24, the runner emits a compact block certificate
covering native/evaluation work, rehearsal retention and dose, translation and
admission counts, and the full donation-only transaction. The CPU-32 launch
script assigns the three remaining arms disjoint CPU sets and records source,
bank, and checkpoint hashes before any learning starts.

The three remaining arm modes passed sequential real low-compute smokes on
2026-08-11 from the exact Arm-1 checkpoint hashes. Adaptive/no-sharing exercised
different scientist proposals and evidence-backed selection. Static/sharing
admitted three strictly better donations and completed three exact exposures.
Adaptive/sharing admitted four and completed four exact exposures. Every paired
donation transaction was noninferior under both objectives, and a completed-run
resume restored the synchronized roster successfully. These are mechanism
checks, not R24 treatment results.

Pair initial weights, representation-specific self-play/evaluation seeds,
budgets, and group membership. Seeds are keyed by representation and scientist,
not round index, so reordering does not change task randomness. Adaptive
controllers may consume different compute; report both quality-versus-compute
curves and compute-matched truncation. Primary quality is complete capped L10
and L1000 on every completed group. Also report solve rate, solved-set
intersections and arm-only identities, acquisition curves, rehearsal/donation
dose, network evaluations, and wall-clock.

Only after the 100+ gate shows a stable advantage do we open a 1,000+
representation run and a separate hard-knot upper-bound campaign.

## Recommended nine-step plan

1. **Freeze the semantic contract and data splits.** Pin charged semantic moves,
   verifier version, objective ratios, failure caps, curriculum identities,
   source-disjoint assessor/pilot/test sets, seeds, and code provenance.
2. **Select the three architectures.** Use `strand-graph` seed 71,
   `raster-axial` seed 71, and `cyclic-memory` seed 73. `window-local` remains an
   engineering baseline because only one of three seeds reached the last stage
   and its aggregate promotion/retention result was weaker. Test replacements
   under separate names and admit one only after it beats the corresponding gate.
3. **Run neutral foundation pretraining.** Start every scientist and seed from
   random weights; use the 1:1 L10/L1000 mixture, native self-play, balanced
   replay, and adaptive `F_native`, `F_old`, and simulations. No donations and no
   adaptive task ordering are allowed in this stage.
4. **Certify each resulting checkpoint.** Require at least 70% paired held-out
   solve rate at the declared allocation, calibrated `p(solve)`, useful
   conditional crossing/move heads, aggregate budget monotonicity, exact
   save/resume, and a non-regressing rehearsal block.
5. **Pass the adaptive-assessor gate.** Train and calibrate checkpoint-bound task
   assessors on a separate set of at least 100 representations. Only certified
   assessors may populate the scientist-specific indexed heaps used by adaptive
   scheduling.
6. **Pass the sharing gate independently.** Distil multiple verifier-confirmed,
   strictly better receiver-native donations into the ordinary policy. Compare
   paired sharing/control checkpoints and require no solved-set or capped-cost
   regression; inferior donations perform zero policy updates.
7. **Run the sequential 24+200 four-arm pilot.** Compare static no-sharing,
   adaptive no-sharing, adaptive sharing, and static sharing. Verify resumability,
   paired task seeds, final solved-set differences, and complete compute accounting.
8. **Freeze and run the 1,000+ comparison.** Change no protocol selected from the
   pilot. Use independent paired seeds and make capped L1000 the primary outcome,
   with solve rate, L10, acquisition curves, and wall-clock secondary.
9. **Launch the separate hard-knot campaign.** Specialize the best admitted
   solver/portfolio toward L1000, compare against cited upper bounds, and publish
   every claimed improvement with a replayable witness, provenance, and
   independent verification.

Steps 1–3 are complete. Foundation pretraining was deliberately stopped after 11
of 12 planned scientist/seed jobs: completing `cyclic-memory` seed 72 could not
change the deterministic K=3 selection rule. The selected checkpoint paths,
seeds, roles, and SHA-256 hashes are frozen in
`pgx-mcts-bench/research/semantic-v1-k3-selection.json`.

The former step 4 was executed on 24 source-disjoint table representations and
failed as a zero-shot transfer test. At 64, 128, and 256 simulations, coverage was respectively
1/24, 1/24, 1/24 for `strand-graph`; 7/24, 6/24, 6/24 for `raster-axial`; and
1/24, 2/24, 3/24 for `cyclic-memory`. No scientist roster came close to the 70%
floor. A transactional ten-round static smoke confirmed that direct ordinary-
policy sharing and interruption/resume work mechanically, but its 7.5--10%
held-out coverage and worse capped objectives correctly failed the sharing
gate. Exact results and artifact paths are in
[`../paper_partials/2026-08-10-semantic-v1-k3-big-experiment-preflight.md`](../paper_partials/2026-08-10-semantic-v1-k3-big-experiment-preflight.md).

That result no longer blocks a sequential bridge curriculum; it shows why the
bridge is needed. The first active arm is
`SV2-3S-R24-SIM64-F10-AR-EV4-NO-SHARING`. Sharing arms remain closed until the
separate donation phase and paired donation-only rollback gate are complete.

## What survived the archive

Three general findings remain useful:

- search allocation can dominate moderate parameter-count differences;
- failed attempts are useful `p(solve)` data but are not expert policy routes; and
- sharing must be compared on final solved sets and verified objective, not on
  whether canonical-route loss decreased.

The three selected semantic-v1 foundation checkpoints exist, but none becomes a
paper-arm starting checkpoint until the source-disjoint critic, budget,
assessor, retention, and sharing preflights pass.
