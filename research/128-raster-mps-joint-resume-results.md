# Full-task MPS gradient and optimizer-resume gate passed

2026-09-21, completed21:51 UTC. Protocol127 passed on private copies of both
final97 arms. This closes the selected joint-backward and checkpoint-resume
engineering checks left open by126. It is NOT new scientific training, a model
promotion, proof of embedding utility, or permission to change the running MCTS.

## Reproducible artifacts

Directory: `artifacts/raster-mps-joint-resume-gate-v1/`.
Report SHA256:
`462ab2f637be99c3b60e6335da960716253c4e8fa31becc988cedae15682dc3f`.
Inputs SHA256:
`02285cb4bd4f5564dd0fc0b347460e6bfa84bb1dd0658a803f65d019c7387d92`.

The runner exited0. All1231 source/input pins were rechecked, including original
training checkpoints; saved gradient and six engineering-checkpoint hashes were
verified against the report. Nine targeted tests passed and changed-code Ruff
passed. No original training/evaluation code or checkpoint was modified. New
engineering weights are retained separately and must not be imported as97 models.

## CPU/MPS gradient agreement

Used the fixed TRAIN draw16 from118:32 metric pairs (including canonical empty
anchors), eight multi-positive policy states, and all four complete relative
groups (334,52,74,54 contexts). Both arms retained their own saved initial CPU
reference log probabilities. No group truncation, per-trajectory normalization,
TEST scoring, neural cache, automatic CPU fallback or cloud resources.

Device MPS float32, one host thread, microbatch1, original tilebatch64 and
architecture. Compared metric, positive, each relative group separately, and
actual accumulated metric+positive+relative0 gradients. The original two-pass
relative backward also checked agreement between its repeated forwards.
Model/objective state bytes remained unchanged through these gradient checks.

Independent NumPy recomputation checked14 whole vectors plus their112 block
comparisons (126 comparisons total) against the fixed criteria from127. Scalar
results were separately checked against118 CPU losses/scores.

| Arm | Maximum relative gradient L2 error | Minimum defined cosine | Maximum scalar absolute difference |
| --- | ---: | ---: | ---: |
| Q254-final | 4.43420e-6 | .9999999999949 | .000610352 |
| Random-final | 7.30799e-6 | .9999999999735 | .000244141 |

The gradient maxima cover whole vectors and non-tiny parameter blocks; tiny
blocks passed the absolute-error criterion. Scalar differences obey the declared
combined absolute/relative tolerance, not an absolute-only threshold.

## Saved optimizer and replayed update

For each arm, loaded its actual67 AdamW state entries at step2048 onto a private
MPS model. Applied metric+positive+full relative1, saved at2049, then applied
metric+positive+full relative3. Restored the intermediate snapshot into a fresh
model/optimizer and repeated that second update. This is an engineering sequence,
NOT97's every16 relative schedule, an RNG-aligned training continuation, or a
CPU/MPS optimizer-trajectory comparison. Six physical optimizer calls in total.

Immediate restore matched bitwise, including model, objective, optimizer moments,
counters and Python/CPU Torch/MPS RNG states. Both branches advanced to2050 with
finite changed private weights. Final control and replayed states met the fixed
rtol1e-5/atol1e-6 criterion, but were **not bitwise identical**. An additional
strict-equality audit confirmed this limitation; tolerances were not relaxed.

Independent NumPy checks covered269 model/objective/optimizer tensors per arm.
Q254 had115 non-bitwise tensors, random120; maximum absolute difference was
7.45058e-9 in each arm. Step counters and all saved RNG states matched exactly.
The tiny numerical difference is observed evidence, not a diagnosis of which
kernel caused it or a guarantee it cannot accumulate during long training.

Each arm retains `engineering-resume.pt`, `engineering-control-final.pt`,
`engineering-resumed-final.pt`, and `gradients.pt`; exact hashes are in the root
report. These files are engineering outputs only.

## Timing and next scientific boundary

This gate did not collect a controlled throughput benchmark. For example, full
relative0 took76.46s for Q254 and31.80s for random, while mixed accumulation took
43.90s and33.30s. Warmup/host-load differences are not isolated. Do not infer a
whole-training speedup, MCTS speedup or a million-example ETA from these timings.

Local MPS is now a technically supported candidate for a separately frozen next
joint-training experiment on this architecture and these task shapes. Long-run
stability, new-view sampling/coverage, scientific loss balance and held-out
distance/search improvement still require actual experiments. Do not silently
continue97 with a different backend, objective, normalization or data schedule.

The active CPU evaluation remains the required scientific comparison. At the
21:46–21:47 hourly check it had16/128 Q254 conditions committed, one success,
and a live advancing worker. All95 committed episodes subsequently passed the
partial replay audit recorded in120. Random-arm search and the complete paired
audits/summary remain outstanding; this backend gate does not complete that goal.
