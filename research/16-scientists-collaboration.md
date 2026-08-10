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

## Experiments

After all selected scientists pass admission, run a source-disjoint 100+ representation
gate before 1,000+ representations. The long comparison contains five arms:

1. three scientists, adaptive order, sharing;
2. three scientists, static order, sharing;
3. three scientists, adaptive order, no sharing;
4. three scientists, static order, no sharing; and
5. the strongest single scientist, with matched total search/training compute.

Pair initial weights, representation order where applicable, evaluation seeds,
budgets, and total compute. Report solved-set intersections and treatment-only /
control-only identifiers. Primary quality is the fixed-set capped L1000 sum;
secondary outcomes include solve rate, L10, acquisition curves, and wall-clock.

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
7. **Run the corrected 100–200 representation five-arm pilot.** Compare adaptive
   sharing, static sharing, adaptive no-sharing, static no-sharing, and a
   compute-matched strongest solo scientist. Verify resumability, final solved-set
   differences, and complete compute accounting.
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

Step 4 was then executed on 24 source-disjoint table representations and failed
decisively. At 64, 128, and 256 simulations, coverage was respectively
1/24, 1/24, 1/24 for `strand-graph`; 7/24, 6/24, 6/24 for `raster-axial`; and
1/24, 2/24, 3/24 for `cyclic-memory`. No scientist roster came close to the 70%
floor. A transactional ten-round static smoke confirmed that direct ordinary-
policy sharing and interruption/resume work mechanically, but its 7.5--10%
held-out coverage and worse capped objectives correctly failed the sharing
gate. Exact results and artifact paths are in
[`../paper_partials/2026-08-10-semantic-v1-k3-big-experiment-preflight.md`](../paper_partials/2026-08-10-semantic-v1-k3-big-experiment-preflight.md).

Therefore steps 5–9 remain closed. The immediate replacement for step 3 is a
neutral source-disjoint bridge curriculum with static outcome-blind ordering,
no sharing, no adaptive scheduling, balanced replay, adaptive acquisition, and
adaptive rehearsal. The bridge checkpoints must re-pass step 4 before assessor
certification or any paid arm is run.

## What survived the archive

Three general findings remain useful:

- search allocation can dominate moderate parameter-count differences;
- failed attempts are useful `p(solve)` data but are not expert policy routes; and
- sharing must be compared on final solved sets and verified objective, not on
  whether canonical-route loss decreased.

The three selected semantic-v1 foundation checkpoints exist, but none becomes a
paper-arm starting checkpoint until the source-disjoint critic, budget,
assessor, retention, and sharing preflights pass.
