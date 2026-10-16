# Unified raster/graph/embedding/policy initialization: gate passed

2026-09-21. Protocol88 completed, exit0. This is an initialized, differentiable
whole architecture, NOT a newly trained or promoted knot solver.

Artifacts: `artifacts/pretrained-raster-graph-initialization-v1/`.
Report SHA256: `5c97f8f47d044ba0b0c6deccdbb9a2c4b724e429bb513c9199c7718dd2541309`.
Initialization checkpoint SHA256:
`ca7a2d701bed24d3e2e2bf10ad55355cc7b3ade23b79f5741028ba1f75929c25`.
All142 source/input pins and the checkpoint hash were rechecked after exit.

## Implemented structure and weight provenance

`scripts/pretrained_raster_graph.py` implements:

1. Shared Q254 raster tiles:131840 EXACT pretrained parameters from gate86.
2. Crossing-feature/geometric projection to64 channels.
3. Two applications of a shared strand-message block, then automatic circular
   multiscale context with size-independent parameter shapes.
4. Masked mean/max/min pooling and three separate normalized64-dimensional
   heads: knot, layout, isotopy.
5. A shared action-site policy head, conditioned on these global vectors,
   five resource features and structural insertion-generator features.
6. Optional head-relative Fourier context combined with local tokens through
   an ordinary learned projection. This is not an additive value adapter.

Total267657 parameters:131840 pretrained,135817 newly initialized. The graph
architecture uses existing RF components but its weights are NEW; no Q304 graph
checkpoint was copied. The native Q254 policy/value/metadata branches are not
being presented as transferred. No value head is supplied at this stage.

All parameters retain identical shapes as L/N vary. There is no learned
generator-index or absolute-position table, dense LxL attention, or full N x L
raster. Graph preprocessing uses exact integer braid generators. Global heads
are distinct projections, not yet proven invariants or disentangled factors.

## What passed

- Finite masked policy logits, with invalid/padded sites at negative infinity.
- Finite nonzero gradient groups for pretrained tiles, projection, strand graph,
  multiscale context, global heads, positional projection and policy head.
  This used a synthetic differentiation probe, not an optimization experiment.
- Zero optimizer steps; ALL initialization tensors stayed unchanged throughout
  the gate. Strict checkpoint reload reproduced them exactly.
- Empty-word and mixed-length handling. Large garbage values in inactive padding
  did not affect valid local tokens or global vectors.
- Cyclic word/action/head reindexing covariance; measured policy error0 on the
  declared64/33 check.
- Two different head-position rescoring calls caused zero raster backbone calls.
  Global vectors remained head-independent. Owned copies of input geometry keep
  later caller mutation from silently changing the meaning of encoded tokens.

| Word length / strands | Sparse legal actions scored | Observed CPU forward |
| --- | --- | --- |
| 64 /33 | 5 | 0.076 s |
| 256 /65 | 5 | 0.302 s |
| 4096 /1025 | 5 | 5.094 s |

The long-word gate scores five explicitly constructed legal actions, not every
possible insertion on the long braid. Complexity of exhaustive action enumeration
has not disappeared; a sparse proposal/search contract is still needed. The
smoke words are constructive unknots, not a difficult generalization benchmark.
Timings are single CPU observations while another experiment was training.

Eighteen focused tests passed across the new integration, tile transfer and
common-domain diagnostics; the new integration files pass Ruff. Initial unit
tests caught and fixed a padding-argument error in the new script before this
gate ran. No existing training/evaluation code was changed by that correction.

## Remaining work before a meaningful trained comparison

This checkpoint is explicitly `trained=false`. It does not establish useful
Gordian distances, u prediction, policy quality, or benefit from Q initialization.
It also does not establish isotopy invariance: the current knot projection is
over the raw diagram, and that property must be learned/tested, not inferred
from cyclic covariance or the head name.

Next measure real forward/backward training throughput and memory on audited
training sizes. The overlapping local windows have a significant constant
compute cost; do not extrapolate a five-action inference into a two-hour10M-example
training promise. Then freeze a matched pretrained-versus-random initialization
training protocol, keeping the same global layers, supervision and independent
evaluation. Reuse set-valued policy targets; do not force a single proof-graph
action sequence or label budget failures as intrinsically bad states.

Encoded tokens may be reused only while the word, architecture parameters and
other relevant inputs are unchanged. Cross-semantic-edit or cross-optimizer-step
cache invalidation is not implemented here. An adapter to the MCTS runner must
make that boundary explicit and must not silently invent a calibrated value.

The original scientific experiment is still required: at this verification,
control2816/4096 was advancing, graph4096/4096 complete; all60 training source
hashes matched. Independent distance and long-input search evaluations remain
queued behind the completed pair. This engineering milestone does not replace
those outcomes or complete the overall embedding objective.
