# Pretrained raster tiles: engineering gate before global-network training

2026-09-21. Follow the original architecture direction in21, not a residual
value adapter. Keep active training69 and evaluations70/74 unchanged.

Q254's verified checkpoint contains a four-plane raster input convolution and
four masked axial blocks, width64,131840 learned parameters. Their shapes do
not depend on word length or generator vocabulary. Native raster observations
have12 rows and7 cyclic window columns. Copy exactly these input/block tensors
into an isolated tile encoder; do NOT claim to transfer the full248418-parameter
native policy, its generator-specific action head, metadata branch or value.

For every crossing, gather seven cyclic word positions and a contiguous band of
twelve strand rows containing both endpoints of the central crossing. With at
most12 strands this is exactly the native raster view. For larger braids shift
the band around that crossing, clipped to physical strand boundaries. Compile
the four planes directly for each bounded tile, never materializing N x L.
Extract the two central crossing-row feature vectors (128 floats) as a token.
Share the SAME copied module across all tiles; parameter count stays constant.

For ordinary Artin braids only word position is cyclic; strand rows do not wrap.
A tile cropped through a neighboring crossing can show just one endpoint: it is
a view of the full diagram, not a standalone valid braid or a legal-action oracle.
Provide true global row position, physical boundary distances, and explicit
top/bottom crop flags separately. Do not relabel that crop as a complete knot.

Masked GroupNorm uses the tile's spatial statistics. Therefore local fixed-size
tiling is not mathematically identical to running the pretrained trunk on an
arbitrarily large whole raster, even with a halo. Require numerical equivalence
only on the original native12x7 views. Outside that domain, weights are reused,
but useful extrapolation remains a training/evaluation question.

Keep local tokens independent of the moving controller head. Provide a separate
head-relative Fourier context at frequencies1,2,4,8 for every token, along with
row/boundary/size/crop metadata. A head shift changes this cheap context, not
the already encoded word. Verify cyclic reindexing covariance and head changes.
No learned position table or generator embedding vocabulary may impose a cap.

Gate: exact copied tensor equality; four-plane crop differential checks against
the native dense raster compiler on small inputs; native spatial feature
equivalence; short-word wrap and invalid-generator tests; finite gradients;
same parameter shapes and bounded tile batches at64/33,256/65,4096/1025;
cyclic-shift covariance; checkpoint/report/source hashes. Large smoke words
are braid closures constructed as an N-cycle backbone plus inverse pairs.
Passing the gate proves transfer/shape/geometry engineering, NOT search quality,
knot invariance, E_knot quality, a complete raster+graph policy, or RL success.

Next integration must join these crossing tokens by strand-graph edges and
multiscale global context, provide distinct layout/knot heads and a size-independent
action head, and train/evaluate that unified network. The new integration layers
cannot be called pretrained or function-preserving without their own evidence.
Do not replace current frozen experiments or deploy this prototype to UnknotDB.
