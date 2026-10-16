# Matched raster training completed; evaluation started

2026-09-21. Protocol97 completed both2048-update arms without changing its frozen
contract. This is training-integrity evidence, not evidence of useful distances
or improved L10/L1000 search.

## Artifacts

Training root: `artifacts/pretrained-raster-log-training-pair-v1/`.

| Arm | Final checkpoint SHA256 |
| --- | --- |
| Q254-local | `c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f` |
| Random-local | `17433748fac7308cf4c36ef0c40650c7ec83d58dee8625834ce531c2f5bb40c6` |

Both final files match their manifests and progress checkpoints. The parent
manifest is completed with both arms; the original training session exited0.
All1143 frozen input pins and67 source pins were independently rechecked.

Pair audit: `artifacts/pretrained-raster-log-evaluation-v1/training-pair-audit.json`.
SHA256: `759e9d4cb114afabbb2f7bb802bec66d19642e8ea933e405b0f3b7d86fa6e57c`.
Status: passed. The controller accepted the audit and advanced to the first
initial-Q254 short-u distance evaluation; the new evaluation worker was observed
live under the existing controller. No duplicate evaluation was launched.

## What the audit establishes

- Exactly2048 updates per arm;67 optimizer state entries each reached2048.
- Exact intended matched initializations:21 local tensor keys differed, while
  upper layers matched. Each arm changed66 model tensor keys during training,
  including its local raster encoder, global heads and policy head.
- Model, objective and optimizer tensors are finite; shapes, fixed optimizer
  hyperparameters, contracts and final hashes pass the declared checks.
- Python/Torch RNG states, draw digests and full-relative-group counts match
  between arms. Shared draw digest:
  `ef8ee40a53aa71d9c3361330fd483077d3922c6ebfec6eb6b1e2a2a2c2c69e41`.
- Each arm consumed65536 metric pair draws,16384 positive-policy state draws
  and128 full relative groups. Group counts were[33,33,34,28], representing
  16766 relative-state visits. These are repeated draws/visits, not unique
  knots, new online MCTS games or a million-example RL run.

The source is the frozen mid-Q254 raster snapshot described in102, not a
certified best or final Q-Run solver. Only the local spatial trunk was transferred;
the original Q policy/value heads were not. Both arms then trained all parameters.

## Remaining scientific evaluation

The predeclared13-job controller continues with eight initial/final distance
reports across the two arms and two panels, then final-model MCTS and full
trajectory audits for both arms. Its final summary requires the complete matrix.
Do not promote on training loss or integrity checks, or tune using partial TEST
scores. TEST panels70/74 were reused and are not fresh independent replications;
historical Q exposure is incompletely audited. Both arms use embeddings, so this
comparison alone does not isolate their causal contribution versus no embedding.

The older graph-mixture result is separately closed in107. It does not replace
the new raster evaluation. Arbitrary-size input support is an architectural
property, not established arbitrary-size solution quality; million-scale online
RL and reliable improvements in L10/L1000 remain open.
