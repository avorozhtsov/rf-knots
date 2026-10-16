# Unified raster network: policy training and cost-search interface

2026-09-21. Prospective engineering gate. Existing69/70/74 code and artifacts
remain unchanged. This is not a scientific solver comparison or promotion.

Add a separate interface for the dict-return protocol88 network. Policy scores
must validate the raw action pool, legality, remaining resources, finite logits
and exact mask coverage. No silent uniform fallback. Preserve request ordering
through length-sorted bounded microbatches; restore training mode even on error.
Optional head coordinates are explicit integers; the global RF controller has
no moving head and therefore supplies no artificial head index.

Results expose priors only. Reading `.value` must raise, preventing accidental
use as a calibrated success probability in legacy MCTS. Only terminal-cost MCTS
and policy continuations that never read this field are in scope. Learned
distance is a soft action prior, never a bound or pruning certificate.

The new encoder does not implement the old normalized-knot view. Embed actual
raw words and an explicit empty one-strand anchor instead. No persistent neural
cache: every call reflects current parameters and raw words; duplicate identical
keys may share computation within that call. Count all actual encodings,
including the anchor, and policy requests/actions/forward batches.

Training uses the existing audited multi-positive action sets and complete
same-root relative-success groups, without inventing negatives from censored
failures. Positive loss is the mean negative log probability mass of all good
alternatives; relative loss is the original signed sum without trajectory-length
normalization. Microbatching must preserve these losses and parameter gradients.
Reference probabilities are fixed-data reference scores, not MCTS behaviour
probabilities or a claim of PPO. No value objective is added.

Gate execution: load exact protocol88 initialization and verify all transitive
pins, plus the passed diamond-supervision certificate. Choose four multi-positive
rows by shortest raw length then corpus order, plus the first positive of
condition0 to include a long word. Choose the complete nonempty relative group
with smallest total active word length, ties by source order. This is score-blind
TRAIN engineering selection. Run actual positive/relative backward with
microbatch2, verify finite/nonzero gradients in all major architecture groups,
and compare against microbatch1. Do not update or save trained model weights.

Then run the actual terminal-cost search with the same untrained network on two
explicit engineering roots: `(1,)` and `(1,1,1)` on two strands, CC caps0 and1,
move caps4 and8, growth2/1, lambda10 and1000, four simulations, horizon8,
insertion_candidates4, widening_batch4, seed20261046. Replay every successful
AND censored continuation independently; replay all best witnesses and costs.
No oracle incumbent, reference action injection or forced policy. The trivial
one-crossing root must be solvable by the tree's terminal transition; trefoil
success is not required and is not a quality claim. Save transcripts, counters,
source pins and an unchanged-checkpoint attestation. The gate unblocks a matched
training experiment, not a declaration of useful embeddings or RL progress.
