# Resource-units evaluation controller: prepared, not launched

2026-09-24 local time (September23 23:26 UTC). Continues135/139 without changing
their frozen sources, input panels, training configuration or checkpoints.

## Executable evaluation sequence

`scripts/run_resource_evaluation.py` now implements the complete serial sequence:

1. Independent final training-pair audit: both2048-step arms, identical initial
   tensors, matched draws/RNG and verified checkpoint/optimizer state.
2. Separate short-u and general panel consumers. An existing completed consumer
   is verified and reused, not overwritten or redundantly regenerated.
3. Cached/uncached parity gate on all four real initial/final models, on non-TEST
   fixtures. This gate has not yet been executed on the new checkpoints.
4. Eight distance reports: two arms by two stages by two panels. Original/lifted
   short-u views are separate; the general panel is evaluated in full.
5. Six128-condition searches: initial/final distance priors for each arm, plus
   a uniform tree-prior control for each final model. Each search is immediately
   followed by an independent full-path replay audit.

This totals24 child jobs. The controller requires completed adequate training
and panels before creating an evaluation directory. It does not start, stop,
restart, or wait for training. It locks its own writer, pins inputs/sources,
checks each completed prefix and adopts an exactly matching live evaluator.
Only the resumable search can automatically continue partial committed output;
partial nonresumable outputs are preserved and require inspection. It refuses
duplicate matching workers and changed completed artifacts.

`scripts/summarize_resource_experiment.py` requires all24 child jobs, rechecks
their bindings, recomputes metric summaries/constant controls/rankings from raw
predictions, and checks every redundant stored metric field. Prediction identity
is validated before a copied row is passed through the annotating grouping
helper, so validation cannot silently repair malformed evidence.

The summary compares each arm before/after training, raw versus units, and
final uniform versus distance priors. Search costs are compared on common
successes only; lost/new successes remain explicit. Successful solution costs
and reference floors are re-certified and compared with the separate replay
reports. Full censored paths are bound to those reports, not replayed twice.
There is no automatic scientific promotion.

## Validation actually executed

- Ruff passed for the controller, summary and controller tests.
- Combined targeted suite:78 tests passed in6.46 seconds, one known optional
  Plink/tkinter GUI warning. Includes training, resource features, fresh data,
  search/cache/resume/replay, distance scoring and the final controller.
- The controller test consumed the actual completed general-panel audit and
  rechecked its input hashes; this was not a model forward or fresh TEST score.
- Tests reject incomplete/reordered matrices, inconsistent stored distance
  statistics/controls/groupings/rankings/endpoint counts, and wrong checkpoints.
- An earlier suite invocation used a nonexistent test filename and collected
  no tests. The corrected complete invocation is the78-test result above.

## Execution boundary and next action

No evaluation controller was launched in this preparation. No135 optimizer or
136 generator was restarted, stopped or routinely polled ahead of the requested
hourly check. Their last live verification remains137's22:51 UTC observation;
this document does not claim their current progress. Next ordinary inspection
is no earlier than23:51 UTC, and must inspect actual processes plus file progress.

After both training arms and short-u generation complete, verify their real
terminal state and adequacy, then launch once:

```sh
.venv/bin/python -u scripts/run_resource_evaluation.py \
  --run artifacts/raster-resource-units-training-pair-v1 \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-resource-units-evaluation-v1
```

Inspect an existing output or runner first; use explicit `--resume` only for
the same contract after checking real process state. Do not interpret a stale
manifest, temporary observation timeout or occupied lock as a dead worker.

The requested scientific goal remains open: independent distance quality and
search utility have not been established for135. The six-search matrix measures
the distance prior's contribution at fixed policy, not total causal value of
global embeddings. It is not million-scale online RL, a public-catalogue search
campaign, a database import/release, or evidence of improved public u bounds.

## Controller recovery tests

The real controller driver was subsequently exercised with isolated fake child
computation in temporary test directories. Six additional cases cover a complete
run followed by a no-op completed resume, reuse of a completed but uncommitted
child, adoption of an existing live worker without launching another, rejection
of duplicate workers without terminating either, explicit search resumption
after failure, and preservation of a partial nonresumable metric output.
These test the orchestration lifecycle, not neural quality, real process
discovery, or scientific replay. The actual general consumer binding test remains
separate. No production training process or artifact is accessed by the fake
driver fixtures.
The combined targeted suite after these additions passed84 tests in5.97 seconds;
the sole warning remained the optional Plink/tkinter GUI import.
