# Actual pretrained Q254 raster transfer: completed engineering gate

2026-09-21. Protocol86 passed, exit0. This is a concrete implementation of the
shared-local-encoder part of plan21, not a newly trained whole solver.

Artifacts: `artifacts/pretrained-raster-tile-transfer-v1/`.
Report SHA256: `9f9a8916c530d210923603ccc55189eaaccd6a69a6ba2d91859f03f7cccf5e45`.
Spatial-trunk checkpoint SHA256:
`397ca781bb38cee215b53cc068219aed80325c1f84a177d829b879bc80a20771`.
All136 input pins and the resulting checkpoint hash were checked after exit.

## What was actually transferred

Exactly131840 learned parameters from the verified Q254 raster checkpoint:
the4->64 input convolution and four masked axial spatial blocks. They are
copied once and shared across every tile. No random replacement, weight
migration, optimizer update, or mutation of the original teacher occurred.
The original full Q254 policy has248418 parameters; its metadata branch,
row pooling, serial action head and value head were NOT transferred.

Implementation: `scripts/pretrained_raster_tiles.py`.
Reproducible gate: `scripts/check_pretrained_raster_tile_transfer.py`.
Each crossing obtains two64-dimensional spatial features, one from each of
its central strand rows:128 local features per crossing, NOT a128-dimensional
global knot embedding. Output count grows with word length; parameter shapes do not.

## Verified evidence

- Ten original-domain native views, including one-letter/short cyclic windows,
  matched the original Q254 spatial output **bit for bit**, maximum error0.
-109 independently compiled dense-raster crops matched the bounded compiler
  exactly, including crops beyond12 strands.
- The copied trunk has finite gradients and a nonzero input-convolution gradient;
  no optimizer step was made. Teacher and copied parameter tensors stayed unchanged.
- Cyclic reindexing of the whole word permutes the output tokens consistently.
- A separate16-dimensional position context contains four head-relative Fourier
  pairs plus global row/boundary/size/crop features. Moving the head changes the
  phase features but not the geometry features or the head-independent raster
  tokens. The caller can reuse encoded tokens; there is no automatic cross-state
  cache or integrated MCTS adapter in this prototype.
- At most64 tiles of12x7 are encoded per inference microbatch. The full N x L
  raster is never built by the tokenizer. Input validation runs once per word,
  not once per microbatch, avoiding an accidental quadratic preprocessing pass.

| Word length | Strands | Output | Parameters | Observed CPU forward |
| --- | --- | --- | --- | --- |
| 64 | 33 | 64 x128 | 131840 | 0.078 s |
| 256 | 65 | 256 x128 | 131840 | 0.299 s |
| 4096 | 1025 | 4096 x128 | 131840 | 4.961 s |

These timings are single CPU smoke measurements during another active training
job, not a performance benchmark. The smoke knots are deliberately constructed
unknots: an N-cycle backbone plus inverse pairs. They test geometry, shapes and
finite computation, NOT hard-knot solving or long-input generalization.

Five focused tile tests passed; together with the preceding common-domain
tests the focused suite has15 passing tests. New source/test Ruff checks passed.
An additional crop-only unit test uses generator indices approaching one million
without allocating one million rows; this is not a million-strand knot-solving test.

## Boundaries and next integration

Only the word direction is cyclic. Interior row crops are marked explicitly;
a crop is not treated as an independent valid braid. GroupNorm statistics are
tile-local, so this is not numerically identical to a full enlarged raster.
The original-domain equivalence does not prove useful behavior outside that domain.

The missing pieces are material: graph/multiscale integration of crossing tokens,
distinct global layout and knot embeddings, a size-independent policy head, and
actual end-to-end training with the right distance/search contracts. No global
invariant, metric quality, complete policy preservation, or RL progress is claimed
by this gate. The new layers need explicit training and matched controls; merely
copying a graph checkpoint without checking its feature/edge semantics would
not establish a valid transfer.

The important change from the old random-start RF lineage is now real and
auditable: a reusable local component actually contains learned Q-Run weights
and accepts inputs beyond48/12. It has NOT replaced the running RF model or
been deployed to UnknotDB/PGX.

At this report the original paired training remained active: control2560/4096,
graph4096/4096 completed. All60 frozen training source hashes still matched.
The independent distance/search evaluators remain necessary for the current
embedding objective and were not bypassed by this architecture smoke gate.
