# Hourly check: complete distance matrix, mixed result; search active

2026-09-26, 00:43–00:46 UTC. Evaluation uses the corrected155 controller and
`artifacts/raster-long-general-evaluation-v2`, not the absent v1 output.

## Completion and verification

Controller PID574/session17201 has committed8/18 jobs: full paired-training
audit, three-model cache parity, and all six distance reports. It automatically
launched `search-control-initial-distance` at00:12:06 UTC, worker PID7097.
Live command uses canonical pgx-mcts-bench; worker100%CPU,31m30.21s CPU in
31m37s elapsed. There are no committed search batches yet; no search outcomes
can be inferred from that empty prefix.

Read-only verification session95389 exited0 at00:44:57 UTC. It checked all
controller input pins and independently called the registered consumer for
each of the eight committed jobs, matching the controller's stored report
SHA256 values and recomputing metric identity/arithmetic against frozen panels.
Both live search roots hashes and all seven old139 committed shard hashes match.
No traceback/runtime/nonfinite/NaN error matched in either controller's logs.

Training audit: status passed,129 checkpoints per arm, all committed optimizer
states and draw streams checked. Initial tensors identical, data RNG aligned;
both final backend RNG states also aligned in this execution. This is training
integrity, not model quality. Cache parity passed for all three model states;
its small engineering-fixture successes are not scientific search successes.

Audit SHA256: `1fb359eb7e1b287fc8f5fa321fe146c3acb8a3b50aecd4e82a8c12911e8e1402`.
Cache SHA256: `6ffdcaec08355720bacc8aed13d3712bae61a074aeadc3f42ada691e06269962`.

## Frozen TEST distance results

Metric is mean distance outside the certified CC interval (absolute error for
exact labels); lower is better. General has442 pairs/437 conservative invariant
groups. Short-u has64 original roots and their64 longer lifted views. The
artifact key `validation` names the original view of the frozen150 TEST panel,
not a training/DEV selection set. These are dependent examples, not a count of
distinct verified knot types or independent statistical trials.

| Model | General147 TEST | Short-u original | Short-u lifted |
| --- | ---: | ---: | ---: |
| shared initial |0.736273|0.201889|0.230080|
| control final |0.700544|0.273038|0.395326|
| long final |0.350116|0.378000|0.403329|
| constant1 |0.515837|0.171875|0.171875|

Long-data training substantially improves general interval calibration relative
to both initial and control, and beats constant1 over all general pairs.
However171 exact-zero examples contribute near-zero error for all models.
On269 certified-positive general pairs, long error0.567568 improves on control
1.151080 but is still worse than constant1's0.211896. Positive-only ranking
accuracy declines: initial0.735673, control0.706173, long0.638337 across7695
dependent comparisons. All-pair ranking would obscure this distinction.

General exact-class MAE(control→long): exact1,0.966760→0.627114;
exact2,1.916241→0.868562. Equal-group general error likewise improves
0.703306→0.353081; it is not merely repeated-record weighting.

For short-u, both training arms worsen original/lifted calibration versus
initialization and fail to beat constant1. Original ranking also declines:
initial0.676136, control0.542614, long0.477273. Lifted ranking:
initial0.377841, control0.497159, long0.252841 (352 comparisons each).
Thus this is evidence of a calibration gain on general short distances, NOT
evidence of a generally useful metric or improved unknotting. No model promotion.

The full six immutable reports and their hashes are in the v2 controller's
completed prefix; no TEST-based tuning, additional training or condition changes
were performed after reading them.

## Older search139

Controller20100/worker21103 remain live; worker100%CPU,359m47.83s CPU versus
303m35s last observation. Still56/128 committed conditions,336 episodes,
188504 policy evaluations,0 solutions; last block23:27:11 UTC. No new committed
result this hour. Prior block took88 minutes and earlier blocks longer, so this
does not establish a stall. Preserve both healthy searches and frozen settings.

## Next step

Continue the registered five-configuration search and full-path replay matrix.
Distance calibration is not a substitute for CC/success/cost comparisons. A
prospectively fixed fully non-neural heuristic comparison is still required
before any claim of neural superiority. Uniform tree prior with neural
continuations is not that control. Keep hourly checks; no cloud, Git or database
publication, no source mutation and no duplicate worker were performed.
