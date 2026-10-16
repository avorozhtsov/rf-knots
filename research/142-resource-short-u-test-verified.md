# Fresh short-u TEST verified; first resource-training arm complete

2026-09-24,00:51–00:54 UTC. Real process/file observations and a completed
independent data consumer, not a model-quality or public-bound result.

## Training: normal serial transition, no restart

At00:51 UTC PID53207 was still the original live135 runner: elapsed2h07m41s,
CPU57m11s,stateR,42.9% CPU. Session23131 remained active. Root manifest recorded
completed_arms=[units], current_arm=raw. Units completed2048 updates and its final
DEV report at00:32:07 UTC. Raw's fresh checkpoint at00:50:22 contained304 updates.
Both exact immutable files were separately loaded on CPU, with their SHA checked:

| Arm/stage | Step | SHA256 | Optimizer states |
| --- | ---: | --- | ---: |
| units final |2048|`cd1fe65ec47a48658a9c892852c63d464bfe4c9d52bc26a833e428e1fd1e365a`|67|
| raw progress |304|`c20b63d3c1a1a2d1d7e3669f1d11df137cadd1980d487a6c970a8dd5486e53ec`|67|

All optimizer step counters equal the respective checkpoint step; all269 checked
model/objective/optimizer tensors are finite in each checkpoint. Units' relative
counts=[34,32,29,33], total128; raw's=[10,3,3,3], total19. This is integrity and
actual optimizer progress, not the final paired audit or generalization evidence.
All1237 training input pins and71 source pins were unchanged. No learning rate,
data, architecture, stopping rule, process or checkpoint was altered.

## Short-u generation completed and separately consumed

Original generator session46692 exited0, with no remaining generator/wrapper
process at the hourly scan. The fixed10000 attempts completed00:23:13 UTC;
wrapper finalization/replay completed00:25:25 UTC. No early stopping at the first
64 accepted records, changed seed, or denominator reduction was used.

- Accepted64 original queries,32 with one-CC witnesses and32 with two-CC witnesses.
-62 conservative query invariant groups, not62 established knot types.
-64 paired size-shift views; these are correlated views, not64 extra independent
  tasks. Original endpoints have lengths0..413 and strands1..33; lifted endpoints
  have lengths0..420 and strands1..33. Empty/one-strand endpoints are the unknot.
- Panel SHA256:
  `edb4fc32ce4eb13ab2cd5cad4a126a5dcee0436372afad33faac0150ab6a16eb`.
- Final generator progress SHA256:
  `ac31bde2302725e722506944619d009868badb5bc19a8e01e2f3959dc14d4f80`.

Ran the existing independent consumer once:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=scripts \
  .venv/bin/python -u scripts/audit_resource_test_panel.py \
  --panel short-u \
  --out artifacts/resource-units-fresh-short-u-test-v1/consumer-audit.json
```

Session64110/PID86582 was observed using real CPU and then exited0.
Result=statuspassed, evaluation_ready=true. It rechecked input hashes, the frozen
10000-attempt/exclusion contract, query partition, canonical uniqueness, selected
original and lifted witness replay, integer determinants and view bindings.
It made zero model forwards and zero optimizer updates. Report SHA256:
`9354ca116cb1a4b19852f072823a47c5a65dd4e589d385c7c9c8a1430add15bf`.

The panel preserves its original conservative interval labels. The fixed scoring
contract strengthens lower bounds using certified fingerprints/determinants.
Recomputing those scoring labels (without a model) gives, in EACH view:

| Certified interval used for scoring | Queries |
| --- | ---: |
|[1,1]|32|
|[2,2]|8|
|[1,2]|24|

Thus32 two-CC witnesses are not32 proofs of u=2. Twenty-four remain intervals.
The original query size strata relative to the declared length254/strands21
training support are40 within,14 length-only,1 strands-only,9 both. All64 lifted
views exceed both thresholds; this creates an extrapolation test, not evidence
that a network extrapolates successfully.

## Both evaluation input panels are now ready

The controller's actual `verify_job` accepted BOTH saved panel consumers and
rechecked their pins. The general panel remains125 pairs; its consumer SHA is
`1077093010ea916dcd84071cb07332a496e9b5a2f52baabdd070004ba1844c0b`.
Neither panel has been scored by135 models. RF-lineage query separation is
audited; complete historical Q254 pretraining independence is not established,
and reference endpoints may be familiar.

The remaining prerequisite is raw training completion and the full paired
training audit. Do not launch the evaluation controller prematurely or evaluate
only the completed arm as though the comparison were complete. The prepared
24-job139 sequence is documented in141. Next ordinary training inspection is
no earlier than01:51 UTC (02:51 London), unless a completion/error event or an
explicit user request requires earlier action. Preserve session23131/PID53207;
inspect actual process state before any future resume decision.

The broader goal remains active. No checkpoint is promoted, no independent
model-quality gain is claimed, and no public database was modified or released.

## Common-step sampling alignment checked before the control completes

A read-only comparison of the two immutable step304 checkpoints verified both
file hashes and exact agreement of draw digest, relative-group counts, Python
sampler RNG, Torch CPU RNG and MPS RNG. Shared digest:
`967b41850e5d2ffb1d2f3c92aa3dbda5befb83bf554472361f17e32f36561ae1`.
At this common prefix each arm has9728 metric draws,2432 positive-state draws,
and19 relative-group updates with counts[10,3,3,3]. These are repeated training
draws, not independent games. Source inspection confirms that both RNGs are
reset after feature-head construction and initial reference evaluation, so that
the units head's temporary initialization does not shift the training RNG stream.
The final2048-step paired audit is still required; no training files were edited.

## September24 01:51 UTC hourly observation

The original runner PID53207/session23131 remains alive, elapsed3h07m43s,
CPU83m39s,45.2% CPU. The instantaneous sleeping state is not evidence of a stall:
raw's manifest/checkpoint was published01:51:22 UTC, and its session delivered
all subsequent training checkpoints. Raw advanced304→1520/2048 updates; units
remains completed2048. There is no evaluation process or duplicate runner.

Raw step1520 SHA256:
`05e8483984bda9eb96456b36621206de0556fc0f4405a0aa5bd8b51cd25f0d5e`.
Both immutable step1520 arm checkpoints were independently loaded:67 optimizer
states each, every step counter1520, all269 checked tensors finite. Their Python,
Torch and MPS RNG states, draw digests and relative counts match exactly.
Counts[28,22,21,24] sum95; shared draw digest:
`5b54b2dbbadac9f63dfff448cfa1afcd059ae4c9da73bee6aeb55fe88a5990a6`.
All training input/source pins were unchanged. This is a verified wait and a
matched-sampling integrity check, not a new held-out score or scientific result.

Raw still needs528 updates plus final DEV/commit. Keep the existing runner;
do not start a replacement or prematurely bypass the complete-pair evaluation
gate. Next routine inspection is no earlier than02:51 UTC (03:51 London), unless
a terminal/error event or explicit user request requires action sooner.
