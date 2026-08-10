# Semantic-v1 foundation pretraining

Status: **stopped after checkpoint selection**.

Run identifier: `foundation-pretrain-semantic-v1-20260809`.

## Purpose and roster

This is the neutral reusable initialization stage for the paper experiments. It
contains no sharing and no adaptive task ordering. Every scientist starts from an
independent seeded random initialization.

The candidate roster was:

1. `window-local`;
2. `raster-axial`;
3. `cyclic-memory`;
4. `strand-graph`.

The full `raster-routed` development candidate is excluded after failing its
objective-quality and efficiency admission gate.

Eleven of twelve scientist/seed jobs completed. The remaining
`cyclic-memory` seed-72 job stopped before stage 3 and was not resumed because it
could not change the deterministic selection rule. The paper-arm starting roster
uses `strand-graph` seed 71, `raster-axial` seed 71, and `cyclic-memory` seed 73;
the exact checkpoint hashes and rationale are frozen in
`pgx-mcts-bench/research/semantic-v1-k3-selection.json`. `window-local` remains an
engineering baseline rather than a selected scientist.

## Frozen settings

- paired seeds 71, 72, and 73;
- exact objective mixture L10:L1000 = 1:1;
- internal horizon 5;
- `F_native` levels 5, 8, 12, 16;
- simulations per move 64, 128, 256, 512;
- `F_old` starts at 1 and may increase to 8 from retention evidence;
- ten held-out evaluation attempts per objective;
- acquisition floor 70%, promotion and retention targets 80%;
- balanced replay and success-only policy/cost training;
- six source-disjoint early mixed-strand curriculum stages.

## Reproducer

```bash
uv run pgx-mcts-bench braid-foundation-pretrain \
  artifacts/current/semantic-moves-v1/foundation-pretrain-semantic-v1-20260809 \
  --only window-local,raster-axial,cyclic-memory,strand-graph \
  --seeds 71,72,73 --workers 8 \
  --native-levels 5,8,12,16 --simulation-levels 64,128,256,512 \
  --initial-old-cycles 1 --max-old-cycles 8 \
  --evaluation-target 0.70 --retention-target 0.80 \
  --selfplay-games 2 --eval-games 10 --eval-every 2 \
  --retro-games 24 --promote-at 0.80 --stage-limit 6 --device cpu
```

The v2 manifest embeds every complete candidate specification and code
provenance. At launch:

- base commit: `5e046d300dc8c5b16baa745884f38c0ecc72f11f`;
- executable source SHA-256:
  `b3bde294c7f16e4d491af86ec4f725b84d9d59be4c01be1ec08023b5e382039c`;
- Git status SHA-256:
  `77412bb486ae65f2d3bdf41d5f3ef798056e7c34d831d4831c22484acfd26f6e`;
- eight worker processes, each with one PyTorch thread.

The worktree is deliberately marked dirty in the manifest. Publication must cite
the embedded executable-source hash or the later commit proven to reproduce it;
the base commit alone is insufficient.
