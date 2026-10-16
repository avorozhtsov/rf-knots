# Separate full-network cache correctness gate

2026-09-21. Predeclared follow-up to111. No integration into protocol97, no
training, no TEST evaluation, no new MCTS. Hourly monitoring stays separate.

Use both completed protocol97 checkpoints, with their fixed identities:

- Q254-local: `c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f`.
- Random-local: `17433748fac7308cf4c36ef0c40650c7ec83d58dee8625834ce531c2f5bb40c6`.

Reconstruct the exact architecture, load strictly, and compare an unchanged
reference network with a private copy whose tile-forward method uses the
standalone111 cache. Keep all global layers and policy scoring unchanged.
CPU float32, one thread, eval/no-grad, cache capacity512. Fix tolerances to
`rtol=1e-5, atol=1e-5`; do not loosen them if a comparison fails. Record failures
as failures, not evidence of safe integration.

Synthetic fixtures only: empty/one-letter words; a short word exposing all eight
non-PASS action kinds and its independent one-move successors; a separate
resource-constrained PASS-only state and its unchanged-word successor;
negative-sign words; strand-crop boundaries; structured words of length64/N33
and256/N65 as in110. Verify every selected action with the reference transition
implementation. Use both cost weights10 and1000, candidate subsets rather than
claiming exhaustive action-space coverage, and heads absent/zero/last position.
Use pairs of requests (last pair may have one), preserving identical padding
and candidate order in cached/uncached comparisons.

Compare local/global encoded tokens, all three global embedding heads, effective
action masks, legal logits and normalized probabilities. Check raw learned CC
distance to the empty anchor using the checkpoint's unchanged objective. Count
argmax differences separately: close floating-point tensors do not guarantee
identical discrete choices. Hash all model/objective tensors before and after;
verify checkpoint and source identities after the gate. Save a new immutable
report, including per-case discrepancies and coverage of actions/moves/heads.

This gate does not prove trajectory equivalence, quality improvements, cache hit
rates on ordinary knots, speedup, arbitrary-size quality, or training-gradient
correctness. If it passes, a separate bounded paired-search engineering gate
remains required before any new cached scientific experiment. The ongoing
uncached comparison must retain its own results and frozen protocol.
