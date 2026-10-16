# F-dominant long curriculum: two-architecture, 24-hour decision plan

2026-10-04. Prospective plan in response to the user's request for sustained
training on 1,000+ different representations. This document is a plan, **not**
an assertion that the required teacher data or training worker exists yet.
The previous 144-step F checkpoint and root-b16 value-head failures remain
unchanged. The independent F/A DEV confirmation (protocol272) runs separately
on CPU and must not be stopped to free a core.

## What counts as 1,000

The target is at least **1,000 distinct TRAIN-owned root query groups** that
each yield a nonzero F preference from two or more distinct fully replayed
successful paths of strictly different objective cost. Count a root only once,
not once per lambda, search seed, cyclic view, or equivalent path. Count
successful F-positive/negative paths and exact state/resource divergence
contexts separately. The old S10 collection has 128 roots and 106 usable
root/objective F groups; it is a bootstrap, not the new 1,000-root dataset.
An unsolved capped search is censored, never a negative action label.

Build a separate constructive TRAIN roster with `conjugate_witness`, explicit
full replay and invariant ownership of the start and every crossing-change
state. Exclude every saved TRAIN/DEV/TEST query group that would violate the
existing partition; never use TEST150 for model choice. Select distinct roots
in fixed SHA-ranked order within four stages, retaining short rehearsal in
later stages:

1. start-word length at most 64, CC=1;
2. length 65..96, CC=1 or 2;
3. length 97..128, CC=1 or 2;
4. length 129..192, CC=1 or 2.

The initial prospective candidate quota is 250 roots per stage; top up by a
new pinned roster if the **audited F-usable** count, not raw candidates, is
below 1,000. Use ten independent searches per root and each lambda as the
first pass; begin at 16 simulations, and use a separately frozen 64/128
simulation escalation for roots whose successful-path diversity is lacking.
Escalation is TRAIN-only and must never turn failure into an action negative.
Archive full outcomes, seeds, CPU/neural evaluations, SHA and replay proofs.
Before a heavy launch, benchmark yield and seconds on a small balanced
pilot; re-estimate total CPU from that measured yield, rather than promising
completion in a fixed number of days.

## Main and auxiliary gradients

F is the dominant gradient. For a given root, lambda and fixed search budget,
use 1..3 near-best complete paths within objective margin 2, and at most K=4
strictly worse complete paths. Compare next actions only at identical
state **and remaining-resource** contexts; shared prefixes cancel. Normalize
the K worse paths to total negative mass -1. The positive paths share total
mass +1. Keep L10 and L1000 groups separate. Preserve the prior 144-step F
as one initialization option, but freeze the final starting checkpoint before
any new DEV scoring and use the same start for both architectures.

Add a small two-output auxiliary prediction of the paired remaining CC and
moves of the **best observed replayed completed path for that lambda**.
These are finite-search conditional targets, not exact unknotting distances;
missing solutions do not supply a cost target. Condition on lambda and
remaining CC/move resources; attach the head to the shared representation.
Use normalized smooth-L1 errors. On TRAIN-only calibration batches choose
loss coefficients so the shared-encoder auxiliary gradient norm is at most
10% of F's and combined distance/known-solution rehearsal gradient norm
at most 20% of F's. Measure and log norms/cosines, not just raw loss values;
if F gradient is zero, skip that update rather than silently training mainly
on auxiliary tasks. No p_solve training or use of the failed root-b16 head.

## Two controlled architectures

Use the same pretrained raster tile trunk, graph width 64, policy head,
embedding dimensions, weights and initial checkpoint in both arms. The sole
architecture difference is depth of the **shared-weight** strand-graph
message block:

- R2: two propagation rounds, the currently tested architecture;
- R4: four propagation rounds, extending graph information flow without
  changing parameter count or inventing incompatible initial weights.

This is a controlled depth comparison within one architecture family, not a
claim to have tested every raster-versus-graph design. Both receive identical
F groups, root-balanced ordering, auxiliary labels, optimizer settings and
DEV queries. Alternate training blocks on the one MPS device; record actual
active training wall time, optimizer steps, CPU, MPS availability, neural
forward/backward counts, data coverage, RNG and AdamW in SHA-pinned
checkpoints. No simultaneous competing MPS trainers or reset on resume.

## 24-hour choice and continuation

Start the 24-hour clock only when both trainers have saved their first
optimizer checkpoints and the F-usable TRAIN bank has reached a separately
audited minimum of 200 distinct roots. The first 24 hours are a matched
architecture pilot, approximately 12 active MPS hours per arm, interleaved
by fixed blocks. Freeze a new disjoint DEV panel and evaluation script
**before** reading either arm's result. Primary simple criterion: number of
distinct DEV roots solved at L1000 by at least one fixed-seed, 128-simulation
search, at the same starts and caps. Guardrails: L10 roots solved may be at
most two fewer than the other arm; no higher CC on common successes; CPU and
neural evaluations are reported and a >10% increase requires an equivalent
compute-matched rerun before selection. If the paired difference is fewer
than three independent DEV roots, call it a tie and continue both only for a
predeclared bounded replication, not an arbitrary winner. Do not select by
TRAIN loss, the viewed TEST150, or repeated-seed counts as if independent.

The selected arm may receive the remaining multi-day budget and full
1,000+-root curriculum only after the data bank actually reaches that scale.
Do not claim a 24-hour decision if the bank or matched training time was not
available; report the measured deficit and shift the decision clock.
Reserve a new independent TEST after method selection. A superiority claim
still requires a predeclared fully non-neural canonical pgx-mcts-bench
comparison with identical starts and resources, full replay, solved fraction,
CC, moves and wall/CPU time separately for L10/L1000.

No cloud purchase, database publication, commit, merge or push. Preserve all
existing dirty files, failed experiments and checkpoints.
