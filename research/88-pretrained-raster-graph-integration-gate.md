# Unified pretrained raster / graph / metric / policy engineering gate

2026-09-21. Extend the passed local transfer86/87 in isolated new scripts,
without editing any live training/evaluation source or checkpoint.

Architecture: exact copied Q254 spatial tiles ->128 crossing features plus eight
head-independent geometric features -> width64 projection -> two applications
of a shared strand-message block -> automatic circular multiscale context ->
masked mean/max/min pooling -> three separate normalized64-dimensional heads
(knot, layout, isotopy). The graph and multiscale blocks are existing RF
architectural components with NEW weights, not a copied Q304 network. No claim
of trained knot invariance or disentanglement follows from naming the heads.

Policy: a shared per-site head consumes the resulting local tokens, all three
global vectors, five resource features, and structural insertion-generator
features. Its parameter shapes do not depend on L or N. An ordinary learned
projection of token + eight head-relative Fourier features supplies local
position context when a controller head is present. No head means zero phase
features, not an arbitrary absolute seam. This is one unified network, not an
additive value/residual correction. No value predictor is fabricated or trained
at this engineering stage.

Separate encode and score APIs permit multiple head positions/action subsets
to reuse one encoded word. Global vectors must not change when only the head
moves. Encoded data owns a copy of word/length/strand inputs, so later mutation
of a caller's tensor cannot silently reinterpret cached tokens. Caching across
semantic edits or parameter updates is NOT implemented/authorized by this API.

Gate before any scientific training: exact Q spatial tensors preserved;
finite policy/metric outputs and gradients through both copied and fresh layers;
padding and mixed-length-batch independence; empty-word handling; cyclic
reindexing covariance for tokens, globals and corresponding action scores;
head changes without another spatial forward; correct action masks; one fixed
state_dict/parameter count at L64/N33, L256/N65 and L4096/N1025. No dense LxL
attention, N x L raster, or learned generator/position capacity table.

Save a reproducible INITIALIZATION checkpoint and source-bound gate report.
Do not call it a trained policy or deploy it. No optimizer step in this gate.
Next scientific training needs a prospective paired initialization comparison,
verified multi-positive policy supervision, genuine CC intervals and independent
distance/search validation. The running69/70/74 experiment must still finish;
this architecture gate is not a substitute for its requested scientific outcome.
