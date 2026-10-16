# 127 — MPS joint backward and optimizer-resume engineering gate

This is a backend-readiness experiment, not scientific training or promotion.
Protocol97's CPU MCTS, data, sources and checkpoints remain unchanged. No cloud
resources, cache, or automatic unsupported-operator fallback may be used.

## Fixed inputs and comparisons

Use private copies of both final97 arms, the complete TRAIN draw16 from118
(32 metric rows, including empty unknot anchors, eight multi-positive states),
and all four complete relative groups (334, 52, 74, 54 states). Reuse each arm's
own initial CPU reference log probabilities, saved in118. No TEST inputs.
Reuse118 final CPU component gradients instead of recomputing them.

On MPS float32, with one host thread and microbatch1, independently backpropagate
metric, positive and each full relative group. Preserve97 weights (1, .1, .16).
Then accumulate metric + positive + relative0 without clearing between tasks.
Compare scalars at rtol1e-3/atol1e-4, whole gradients and every parameter block:
relative L2 <= .01 and cosine >= .999 when CPU norm > 1e-6; absolute L2 <= 1e-5
otherwise. Require identical layouts and unchanged state bytes. Retain the
original two-pass relative backward's stricter repeated-forward check.

## Disposable optimizer/resume test

Only after an arm passes gradient comparisons, restore its saved AdamW state
(step2048) onto a private MPS model. Perform two engineering updates using all
three tasks, the fixed metric/positive batch and whole relative groups1 then3.
This deliberately is NOT97's every16 relative schedule or a continuation run.
Preserve optimizer settings and clip1. Save model, objective, optimizer and
Python/CPU Torch/MPS RNG states after the first update. Finish the second update,
then reload the intermediate snapshot into a fresh model and repeat update2.

The immediate restore must be bitwise identical, including optimizer moments,
counters and RNG. Replayed final floating state must agree at rtol1e-5/atol1e-6;
integer/RNG/step tensors and non-tensor fields must match exactly. Reject
nonfinite state. Also report bitwise equality separately. Require step2050 and
changed private weights. Three physical optimizer calls per arm; none published.

## Evidence and failure boundary

Freeze all inherited118 inputs and127 implementation/test/protocol before launch.
Save CPU/MPS gradient comparisons, MPS vectors, optimizer snapshots and hashes,
with an exclusive new output directory. Recheck sources after execution.
Unsupported operators, nonfinite values, source drift and numerical disagreement
fail closed; retain partial evidence, do not silently relax criteria or use CPU.
Success establishes only this bounded joint-training/resume backend capability,
not CPU/MPS optimizer trajectory equivalence, long-run stability, search speed,
embedding utility, or readiness to alter the active evaluation.
