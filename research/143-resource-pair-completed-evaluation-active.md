# Resource-units pair completed; independent evaluation active

2026-09-24,02:51–02:54 UTC. Training135 completed without a restart. Evaluation139
is now genuinely running, not merely prepared. Scientific outcomes are pending.

## Completed training and independent pair audit

The hourly02:51 inspection found no remaining training process. Its original
session23131 exited0. Root manifest completed02:17:24 UTC, after raw's final
development report; both arms completed exactly2048 updates. The full independent
`audit_resource_training_pair.py` then ran as controller child15449 and passed.

| Arm | Initial checkpoint SHA256 | Final checkpoint SHA256 |
| --- | --- | --- |
| units |`40a814831c29dfe1907a8f890978234cded6768f3f981b7e850f3a179d2d8507`|`cd1fe65ec47a48658a9c892852c63d464bfe4c9d52bc26a833e428e1fd1e365a`|
| raw |`a7b28ed2264dab4437fd793c89acd857a6bbd0d5c93135b18be8dee4d05b3bc3`|`68582526729be833c48ca7d3a751e4504226516147e1e6f4bdc1aabcea98b6cb`|

The audit reconstructs the expected initialization, checks all learned tensors,
optimizer state/step coverage, configuration, architecture, DEV report bindings,
finite states and final sampling/RNG equality. Initial learned tensors match
exactly despite different checkpoint file hashes/feature-mode metadata.
Both arms changed66 model tensors, including local raster weights, and have67
optimizer states at step2048. Each received65536 metric draws,16384 positive-state
draws and128 relative-group updates; these are not unique games or graph vertices.

Final shared draw digest:
`babfad51362e7d8064b3bac8bd3814418e3f834e2b3d84879c456e9a48fab84e`.
Relative counts=[34,32,29,33]. Python/Torch/MPS RNG states match. Their initial
policy references differ intentionally because resource feature maps differ.

Training contract:
`1f18ce4e7da31d921d360672f2cd3b07c554992076038c1ca573907ed79af312`.
Independent paired audit:
`artifacts/raster-resource-units-evaluation-v1/training-pair-audit.json`, SHA256
`434e1d452c36aeab421e8f159f426bea39c02520a8cabc639ded1671a3bf6057`.

## Single evaluation controller launched and verified live

Before launch, both the process scan and filesystem showed no existing139
controller/evaluator or evaluation output directory. Launched once:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/run_resource_evaluation.py \
  --run artifacts/raster-resource-units-training-pair-v1 \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-resource-units-evaluation-v1
```

Unified session78472, controller PID15420. Evaluation contract:
`0d9fae7421cd265fb28a1d7cafd5d6684cbd7edc4dc258f91b7fba1d85613b14`.
The controller completed four of24 children: paired training audit, both existing
panel consumers (verified/reused), and the new real-checkpoint cache gate.
Its first scientific distance evaluator, PID15877, was observedR at100% CPU,
36s CPU/37s elapsed, on units-initial/short-u. This includes panel replay before
model scoring; an active evaluator is not a completed metric result.

## Cache gate passed on all four actual models

Child15620 completed the non-TEST gate for initial/final units/raw checkpoints.
Each model passed102 forward fixture cases, exact masks/greedy decisions, and
cached/uncached discrete search equivalence for both prior modes at three seeds.
Floating-point comparisons use declared rtol=atol=1e-5, not a universal absolute
error ceiling; largest recorded absolute forward difference was3.33786e-5 for
raw-initial. All four weights-unchanged checks passed.

Gate SHA256:
`ceaaa80feb5b89686fe6699847d4603fa65a65826de2c7f441ec6404947bfc8e`.
A separate consumer verified all four completed controller entries against
current files, hashes, model/panel identities and input pins. No model was
promoted on the basis of engineering parity. No TEST search was performed by
the cache gate and no scientific speedup claim follows from it.

## Next steps and interpretation

Allow this same controller to finish all eight initial/final distance reports,
six L10/L1000 search cases and six full replay audits. It writes the final
comparison only after the complete matrix. Both TEST panels are verified and
fixed (142); historical DEV is not substituted for them. Report absolute errors
against constant controls and initial models; compare search successes and
common-success costs with actual neural work and wall time.

Do not start another controller or edit its hash-pinned scripts while it runs.
Inspect the real process and committed files before considering explicit resume.
The next ordinary observation is no earlier than03:54 UTC (04:54 London), unless
a completion/error event or explicit user request requires earlier action.

The broader learning goal remains active. Neither useful short-distance
calibration, L10/L1000 benefit, unrestricted-size generalization nor public
upper-bound improvement has yet been established for this pair. No public
database, cloud resource, publication, Git commit or push was changed here.
