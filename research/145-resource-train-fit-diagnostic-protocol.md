# Exploratory TRAIN-fit diagnostic after the135 TEST distance result

2026-09-24. Motivation is the observed144 regression, explicitly post-TEST and
exploratory. This is not another independent test, a learning run, a tuning
procedure or an alteration to the active139 search. No weights are updated.

Question: do the final models fit short distances on their actual TRAIN draws,
or is there substantial residual error there too? A fit gap can describe where
failure occurs but cannot alone identify its causal mechanism.

Reconstruct all2048 protocol135 draws from the frozen samplers, policy-target
pools and seed20261071. Consume the same positive and relative choices so the
metric stream stays identical. Require the final full draw digest and relative
counts to match BOTH independently audited arms before any diagnostic scoring.
Use zero-based updates0,128,...,1920, all32 metric rows from each:512 draw records,
not512 unique pairs. This fixed score-blind systematic sample retains each batch's
7 original,8 suffix,16 ordinary graph and1 sentinel rows. It is not a uniform
sample of the corpus population or a statistical generalization estimate.

Evaluate shared initialization (units-initial), units-final and raw-final on CPU,
float32, one thread, microbatch1, using the original training metric loss and
saved objective. The paired audit establishes identical initial encoder tensors;
139 additionally observed identical initial distances, so no duplicate initial
scoring is needed here. No cache, policy forwards, optimizer, MCTS or graph write.

Save every selected row, its draw location, source, full materialized pair,
normalized batch witness-priority weight, prediction and native loss. Preserve
raw native predictions. Also report predictions clamped to[0,2*saved_scale] to
match the physical cosine range used in TEST reporting; report roundoff-clamp
counts/magnitudes rather than silently hiding them. Native loss uses the raw
training prediction and is separately checked against squared interval violation.

Report record-weighted interval errors and constant0/1/2 baselines, exact/interval
classes, source subsets, endpoint-size ranges, unique raw pairs and the mean of
the16 weighted training-batch metric losses. Keep source weighted contributions
separate from unweighted subset means. No significance claim from repeated draws.
Publish even if one model performs poorly; do not resample to balance outcomes.
Bind reports to the complete paired audit, all training inputs, source hashes
and unchanged model/objective states before/after inference.

This diagnostic may run alongside the existing one-thread MCTS worker. Record
that concurrency; wall times are not a controlled speed comparison. Preserve
all current experiments. Next learning decisions must use a new frozen contract
and fresh evaluation, not tuning on the now-observed139 TEST results.
