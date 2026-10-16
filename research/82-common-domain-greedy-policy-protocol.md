# Common-domain policy diagnostic, frozen before policy scoring

2026-09-21. Use exactly the16 protocol80 roots whose witness-access gate81
passed. This TRAIN diagnostic is not a promotion test, independent generalization
test, MCTS evaluation, or ablation isolating embeddings.

Three frozen policies: Q254 raster, Q304 strand graph (checkpoint identities
in gate79), and RF joint57 (`e58270a34569bfad11127ef4f69a8e67bcdcf4825213f88bce03a73a2233ef84`).
Do not use a partial checkpoint or select models by these diagnostic results.
Run each root at L10 and L1000, one deterministic attempt, seed20261041.

Shared envelope: max word length48, max strands12, at most two crossing changes,
128 policy decisions. One observed state per neural forward, no inference cache,
MCTS, value-based selection, noise, stochastic rollout, or cycle pruning. Pick
the highest-logit legal action with smallest action ID breaking ties. Suppress
PASS in both interfaces: the question is whether policy can find a solution
within the envelope, not whether differently trained stop actions agree.
CC masking applies before selection after two changes, without altering any
model weights. Failures remain censored, never negatives for training.

Q retains its native controller, legal mask, head state and objective-budget
observations (FixedWordGame, native simplify horizon128, no explicit objective
cap). Head shifts consume a decision but not semantic cost. Its original native
termination rules, including controller-only limits, remain active and are
reported. RF receives its existing five budget features, capCC2/capMoves128 and
slack to48/12. Score the full shared semantic catalogue, not four sampled
insertions; no PASS is in this catalogue. This is feasible on these short roots
and removes insertion-shortlist coverage as a confound in this diagnostic.

Equal decision counts are NOT equal FLOPs, semantic work, or historical training
conditions. Native Q sees a local window; RF scores all semantic positions.
Q has no explicit remaining-CC input matching RF's; hardCC2 is an evaluation
constraint. The result compares these declared policies/interfaces, not solely
architectures or solely pretraining. The128-decision envelope is not the earlier
RF8-simulation test and does not estimate full MCTS performance.

Save every selected native/semantic action and before/after raw braid state,
including failures and head-only transitions. Count actual forward samples and
batches, legal action scores, native decisions, semantic moves, and CC changes.
Replay every semantic action through the RF reference, verify native raw-state
agreement at every step, and independently reconstruct every successful witness.
Only successful solutions have L10/L1000 costs; do not assign artificial costs
to budget exhaustion. Record repeated braid states without treating head travel
as an invalid loop. Hash all input sources/checkpoints and completed episodes.
Resume only the identical contract; never overwrite an unrelated run.

Interpretation: check whether strong local Q policies provide a useful teacher
on an accessible common domain. Better/worse performance here cannot establish
benefit/harm of embeddings, long-input transfer, or correctness of a value head.
Keep protocols69/70/74 unchanged. Choose subsequent MCTS/resource or pretrained
local-integration work only after reporting this diagnostic's limitations.
