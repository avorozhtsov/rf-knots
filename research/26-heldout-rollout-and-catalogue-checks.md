# Held-out checks for the rollout-value iteration

2026-09-20. Written while the two 2048-step runs are still training, before
scoring their checkpoints on these panels. Their architecture, training data,
hyperparameters and final-step checkpoint selection are unchanged.

## Rollout-value evaluation outside training knot groups

Frozen behaviour pi0 exactly matches the actor used to collect training labels
(all state-dict tensors compared). Reuse all 64 development short-u queries,
not model-selected roots. Nine held-out fingerprint groups. For each root run
eight new seeds under lambda=10 and lambda=1000: 1024 complete episodes per view.
Value targets remain **single-rollout** success, not MCTS success probabilities.
No episode from either evaluation panel is eligible for training.

Resources follow the known-witness-feasible search contract: at least 64 semantic
moves, +2 letters/+1 strand, expanded if necessary to fit the provided witness.
The actor uses the same deterministic O(L)+16 proposal rule as in training.

Original view (`embedding-rollout-value-validation-v1`): 64 979 transitions,
224 successes /1024. Lifted view (`embedding-rollout-value-size-validation-v1`):
67 242 transitions, 30 successes /1024. These counts describe pi0, not the new
trained value or pi1. Lifted states require additional simplifying moves within
their resource contract; the drop is **not** a clean architecture-only size
generalisation comparison at matched remaining work. It does demonstrate that
invariant global embeddings do not by themselves imply equal rollout success.

`evaluate_rollout_value.py` rejects incomplete/training-eligible data, verifies
shard/root/actor hashes, replays all episode actions and requires the exact seed
set for every query. It compares Brier and log-loss against constant 0.5 and the
training-root success rate (not fitted on evaluation outcomes), with results by
lambda and held-out fingerprint group. Initial value Brier on original view is
0.24909, log-loss 0.69133; 0.5 constant Brier is 0.25.

These are development knot-heldout panels, not a fresh final test after repeated
development. Repeated seeds, mirrors, lambdas and same-group representations
must not be counted as independent knot types. Value for pi0 is not automatically
calibrated for the newly updated pi1; that requires fresh pi1 trajectories.

## Model-blind independent short-u certificate panel

The previous random test stopped partial at 55/128, and the historical independent
cohort had ten exact-u2 queries and no exact-u1. Constant 2 therefore scored
perfectly there. This motivates an additional model-blind certificate cohort,
not tuning predictions on the old test.

Selection is **all** cached catalogue knots with at most eight crossings in the
frozen owner=test partition, computed invariants and |signature|<=4. The cached
table supplies braid representatives and identifier provenance, not trusted
unknotting-number labels. This resolves to 5_1, 6_1, 8_20 and 8_3.

Fixed certificate procedure, no model scores: try each original-word position
for one CC if the signature permits u1; otherwise try every two-position
combination for |signature|=4. Each changed braid gets 2048 zero-CC length-guided
expansions, 64 total moves, +4 letters and +1 strand. Stop at the first complete
replayed witness per root. Every failed attempt is retained and is **not** a
lower-bound certificate. A root with no witness inside this protocol is excluded
with that limitation disclosed, never labelled u>1 or unsolvable.

Already certified without looking at trained scores: 5_1 has a two-CC witness
matching its signature lower bound; 6_1 and 8_20 have one-CC witnesses and
determinant 9 !=1, proving u=1. Their original stored signature bounds remain
[0,1]; evaluation must explicitly request and report determinant-strengthened
bounds. Zero-CC lifted copies are also saved. 6_1 and 8_20 share a fingerprint
group: they are not two additional independent group-level trials.

Artifacts: `embedding-catalogue-independent-test-v1`, including source table,
code snapshots, all attempted positions, resulting witnesses and the panel.
Search-success selection, tiny size and fingerprint collisions remain limitations.
This is not evidence about the entire knot population.

Predeclared model comparisons: E/root parent, completed G/value and G/control,
plus the original untrained architecture as a distance control. Report exact
u1/u2 errors and constant 0/1/2 baselines on both views, not just mean interval
violation. No test-dependent training, checkpoint selection or hyperparameter
changes. Search comparisons retain failures and replay every returned path;
one better path is not statistical proof of broad superiority.

## Completed certificate panel and baseline observations

All 33 certificate-search attempts finished. 8_3 did not produce a one-CC
witness within the fixed budget; this does not certify u>1. The panel contains
three source knots, two fingerprint groups: 5_1 (CC2,5 moves), 6_1 (CC1,25 moves),
8_20 (CC1,9 moves), plus lifted views. Panel SHA:
`0abd1310f726a2457dc16d43b36391cc63ff218ae5d856bdc262e522f5943ab6`.

Only predeclared parent/random baselines have been scored so far. With strengthened
bounds, E/root's mean absolute u error is 0.5572 versus untrained 0.3758;
constant 1 scores 0.3333. E/root underestimates 5_1 at about 1.06 versus exact 2.
This is an actual small independent failure of that checkpoint, not proof that
embeddings are useless after a larger, more diverse training programme. G/value
and G/control must finish before their own scores are read; they are not being
retuned on this observation.

## Fixed-budget stochastic-rollout baseline

For the same nine development roots used by MCTS, take pi0 episodes in seed order
0..7 and keep the cheapest complete witness within at most 512 neural-state
evaluations. An episode whose suffix exceeds the remaining budget cannot supply
a hindsight solution. Compare with **MCTS seed 0 only**, not four pooled seeds
with four times the budget. Resource caps and roots match; one MCTS simulation
uses at most one neural-state evaluation, but action scores and CPU time differ.

`compare_rollout_mcts.py` replays witnesses and checks hashes/caps/costs. The pi0
portfolio solves 6/9 versus that parent MCTS's 0/9. The other MCTS seeds also
failed, but are not counted as extra equal-budget portfolio trials. Saved at
`embedding-rollout-value-validation-v1/portfolio-vs-parent-mcts-v2.json`.
Returned rollout paths are typically 60–64 moves, so this is not a claim of
good L10/L1000. It is a necessary control for the upcoming learned-value MCTS,
and evidence that the constant-value search leaves useful policy behaviour
unexploited. Rollout best-of-budget and MCTS first-found are not optimal-cost
solvers. No architecture or training changes were selected from these results.
