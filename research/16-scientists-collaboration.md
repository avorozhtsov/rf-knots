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

## Stable four-scientist roster

| Name | Information path | Main hypothesis |
|---|---|---|
| `window-local` | local word window plus a scanning controller | Fast learner and continuity baseline |
| `raster-axial` | local braid raster with shared horizontal/vertical axial blocks | Better strand geometry without fixed-strand alphabets |
| `cyclic-memory` | cyclic scan, persistent tape, and global objective conditioning | Can accumulate information outside a local window |
| `strand-graph` | compact shared strand/message blocks with a routed action head | Test strand transfer cheaply, then grow only after a capacity gate |

Names describe architecture only. Objective ratio, simulations per move, seed,
training dose, and checkpoint are explicit run fields.

Every scientist receives the requested `log(A/B)`, remaining semantic objective
budget, remaining internal-step budget, and its architecture-specific observation.
The shared body feeds policy and three auxiliary outputs: crossing changes,
semantic moves, and `p(solve)`. The objective value is constructed from the first
two using `A` and `B`. Budget features reach the shared body and all heads through
skip/conditioning paths so `p(solve)` can genuinely depend on budget.

## From-scratch pretraining

All four scientists begin from independent seeded initializations. The curriculum
starts with unknots, small torus knots, and simple controlled scrambles, then adds
source-disjoint mixed-sign braids and more strands. It samples exactly L10 and
L1000 at a 1:3 ratio because L1000 is the final target.

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

After all scientists pass admission, run a source-disjoint 100+ representation
gate before 1,000+ representations. The long comparison contains five arms:

1. four scientists, adaptive order, sharing;
2. four scientists, static order, sharing;
3. four scientists, adaptive order, no sharing;
4. four scientists, static order, no sharing; and
5. the strongest single scientist, with matched total search/training compute.

Pair initial weights, representation order where applicable, evaluation seeds,
budgets, and total compute. Report solved-set intersections and treatment-only /
control-only identifiers. Primary quality is the fixed-set capped L1000 sum;
secondary outcomes include solve rate, L10, acquisition curves, and wall-clock.

Only after the 100+ gate shows a stable advantage do we open a 1,000+
representation run and a separate hard-knot upper-bound campaign.

## What survived the archive

Three general findings remain useful:

- search allocation can dominate moderate parameter-count differences;
- failed attempts are useful `p(solve)` data but are not expert policy routes; and
- sharing must be compared on final solved sets and verified objective, not on
  whether canonical-route loss decreased.

There are currently no admitted semantic-v1 Pareto plots or pretrained vNext
checkpoints. They will be generated by the from-scratch admission runs rather than
renaming historical artifacts.
