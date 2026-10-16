# Pretrained raster / graph training cost: bounded CPU probe

2026-09-21. Prospective engineering protocol, not scientific promotion.

Use the exact untrained protocol88 checkpoint and the audited ordinary graph
TRAIN corpus. Preserve all historical source/checkpoint hashes. Do not change
the running protocol69 pair or either independent evaluation queue.

Sort eligible nonzero-exact graph pair rows by total active crossings and source
index; select actual adjacent rows at quantiles 0.5, 0.9 and 1.0. No score-based
selection or held-out data. At median cost use dense pair batches 1, 2 and 4;
at q90 use dense batch2; at maximum use batch1 and effective batch4 accumulated
one pair at a time. Report exact lengths, strands, bounds and source indices.

Each case starts from identical protocol88 weights in a fresh process. CPU,
one thread, float32, current chord interval objective, witness-priority weighting,
zero ranking loss, AdamW lr2.5e-5 / wd0.01, norm clip1. One warmup update and
two measured updates on the selected rows. Discard modified weights: no trained
checkpoint or validation metric is produced. All encoders are trainable. Check
finite gradients/parameters and nonzero gradient into the actual Q254 trunk.

Accumulation preserves the global weighted mean using each microbatch's share
of witness-priority weight; this does NOT implement cross-batch ranking loss.
Unit-test its loss and gradients against a full batch before running the probe.

Measure forward-plus-backward and full optimizer step wall time. Peak process
RSS includes imports/data/model/optimizer/activations; it is not isolated tensor
memory. Child processes isolate peaks between cases. Measurements are on a
shared 16GiB Mac while the existing control experiment continues. No large dense
long-word batch, no GPU rental/start, and no claim of throughput on Nebius.

This is metric-only cost. Positive policy, relative trajectories, MCTS, complete
action enumeration, sampling and scientific evaluation add work. Two timed
steps are preliminary throughput measurements, not confidence intervals or a
promise of training10M pairs within a particular time. Results must distinguish
pair draws, representations, optimizer steps, and unique knots.
