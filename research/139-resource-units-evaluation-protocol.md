# Protocol135 evaluation: distance quality and search utility

2026-09-23. Prospective evaluation code/protocol, before scoring136/138 with any
model. Training135 continues unchanged. Historical DEV is not the final test.

## Required evidence and models

Require the complete135 pair and a passed final training integrity audit. Both
arms have identical initial learned tensors, different policy resource feature
maps, and fixed2048-step endpoints. Load the correct feature map from the arm
contract before loading weights: identical tensor shapes do not identify a
model function. Never interpret a units checkpoint with the raw policy head.
No checkpoint selection based on new TEST or DEV scores.

Require completed, adequate136 short-u and138 general-distance panels, matching
source/input hashes, complete fixed-attempt coverage, and replayed CC intervals.
Before scoring, independently consume their saved artifacts, check selection
coverage/exclusions and recheck selected signed paths or full unknot witnesses.
Do not fall back to old70/74 or shrink a failed denominator.

## Distance matrix

Both arms × initial/final × short-u/general. Short-u includes all64 original
roots and64 associated zero-CC lifted views, reported separately; general uses
every selected pair, with adequacy minima from138. CPU float32, one CPU thread,
microbatch1, no persistent embedding cache or optimizer steps.

Use the unchanged learned CC coordinate and certified lower bound strengthening
from determinant inequality. Save each prediction/interval, exact-class and
group-weighted errors, record-weighted interval errors, constant0/1/2 controls,
and descriptive interval-ranking accuracy. Initial knot embeddings must be the
same in the two arms; policy resource units cannot change an encoder-only metric
forward at fixed tensors. Any discrepancy is an implementation failure, not an
initial quality difference. Size strata and within-arm initial/final differences
must be reported without treating correlated views as independent knot types.

## Search matrix and interpretation

Main comparison: both arms × initial/final, learned-distance tree prior and
learned policy continuations, on all64 original136 roots at both L10/L1000.
Additional controlled comparison: BOTH final arms with uniform tree priors and
the SAME learned policy continuations. This isolates using the learned distance
as a tree prior; it does not remove the global embedding inputs from the policy
network and cannot establish the total causal value of those policy inputs.

Keep eight simulations, c_puct1.5, temperature1, widening4, insertion candidates4,
eight conditions per batch, the native70 per-root resource construction, and
search seed20261074. No oracle incumbents, scalar value, graph tails or training
on these trajectories. Search results include every failure in the denominator.
Equal simulations/resources are NOT equal neural work or wall time: record both.
Compare route costs on common successes, and report exclusive successes rather
than comparing means over different solved sets. Replay all paths and every
solution, and certify cost floors using the existing native move-cost identities.

The serial search implementation and its replay audit remain to be completed
before dispatch. A local tile cache may be used ONLY after checkpoint-specific
cached/uncached equality checks on non-TEST fixtures and an explicit pinned
implementation/configuration. It must preserve logits, chosen discrete paths,
legal action/resource semantics and stored weights. Failed cache parity is an
engineering failure, not permission to silently change the scientific protocol.
No timing claim is made from training running concurrently with CPU acquisition.

The full search matrix is six128-condition searches. It may take materially
longer than metrics; report elapsed work honestly and do not call evaluation
calls new learning iterations. No public database improvement or promotion
follows automatically from synthetic successes, metric improvements or tests.
