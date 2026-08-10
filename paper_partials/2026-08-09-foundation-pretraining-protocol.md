# Foundation pretraining protocol and run record

Status: **superseded engineering baseline; stopped resumably on 2026-08-09**.

Run identifier: `semantic-moves-v1/foundation-pretrain-20260809`.

## Purpose

Foundation pretraining gives every scientist basic competence on easy knots
without importing any historical rung-18 or distillation checkpoint. Each
scientist starts from an independent seeded random initialization. This phase is
part of the experimental method, not evidence for any collaboration arm.

## Scientists and seeds

The engineering roster was `window-local`, `raster-axial`, `cyclic-memory`, and
`strand-graph`, with seeds 71, 72, and 73. Architectures are paired by seed, not
by shared weights.

This roster was revised before publication: the proposed fourth scientist is now
the full `raster-routed` architecture, conditional on a matched admission gate.
The objective mixture was also corrected from 1:3 to a neutral 1:1 foundation
mixture. Consequently no checkpoint from this partial is an admissible foundation
initialization for the paper.

The source-disjoint curriculum is:

1. `unknot+2`;
2. `T(2,3)`;
3. `P(3,4)#0`;
4. `P(4,5)#0`;
5. `T(2,5)`;
6. `P(4,7)#0`.

Four strands therefore appear before a long two-strand prefix can dominate
training.

## Adaptive schedule

- `F_native`: 5, 8, 12, 16 iterations;
- MCTS simulations per move: 64, 128, 256, 512;
- initial `F_old=1`, adapted by retention probes up to 8;
- held-out evaluation target: 70%;
- promotion and retention targets: 80%;
- two self-play roots per iteration;
- ten held-out evaluation attempts per objective;
- 24 retrospective attempts;
- objectives L10 and L1000 sampled with weights 1:3 in this superseded run;
- internal controller horizon 5;
- balanced replay and success-only policy/cost training.

A non-promoted rung raises `F_native`. If its held-out solve rate is below 70%,
the same retry also raises simulations. The current representation is retried
before training advances. Search dose is runtime state and does not change the
scientist's architecture identity.

## Reproducer

```bash
uv run pgx-mcts-bench braid-foundation-pretrain \
  artifacts/current/semantic-moves-v1/foundation-pretrain-20260809 \
  --only window-local,raster-axial,cyclic-memory,strand-graph \
  --seeds 71,72,73 --workers 8 \
  --native-levels 5,8,12,16 --simulation-levels 64,128,256,512 \
  --initial-old-cycles 1 --max-old-cycles 8 \
  --evaluation-target 0.70 --retention-target 0.80 \
  --selfplay-games 2 --eval-games 10 --eval-every 2 \
  --retro-games 24 --promote-at 0.80 --stage-limit 6 --device cpu
```

The run started with four workers, checkpointed, and was resumed with eight
workers to saturate the eight-core host. This changes throughput only; each
scientist/seed job has its own deterministic RNG and checkpoint.

## Code provenance

Repository: `pgx-mcts-bench`.

- base commit: `5e046d300dc8c5b16baa745884f38c0ecc72f11f`;
- tracked source/test/docs diff SHA-256 at run start:
  `d7b85ad5bc4acc2b3cfbca87e9531e527f8db3c3fcc8c626e644611121c4bb57`;
- porcelain-status SHA-256:
  `42d5148736c6da71ead05586488cfaa8dcafcb5cfb31032074ef182e1ee5b017`.

The untracked adaptive runner itself has SHA-256
`6f3dd8f9a22380a8df8c2b9e9f3f5853201716024326ebe825dbac90b0191e7d`;
its test has SHA-256
`6e1c6c41d1c79abea2d43e29c66c8eb28e1e06751a29d3db285bf5dc201c7aff`.

The worktree was dirty because the semantic verifier cleanup, assessor gate,
final-block evaluation repair, worker-runtime repair, and this adaptive runner
had not yet been committed. The exact eventual commit must replace these patch
hashes before the paper cites results.

## Metrics to report after completion

For each scientist and seed, report every rung attempt with F, simulations,
held-out solve rate, promotion status, conditional crossing changes, and charged
semantic moves. Aggregate paired results must include the exact solved-set
intersection and seed-specific failures. A final checkpoint is the latest
promoted state, not automatically the weights after a subsequently capped rung.
The only completed run before termination was one `window-local` seed; all
partial states remain engineering evidence only.
