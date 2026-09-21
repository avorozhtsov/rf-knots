# Prospective common-block Gordian supplement

2026-09-20. Declared after the positive-seed general panel in note42 exhausted
its stream and failed adequacy, but before generating or scoring this supplement.
The candidate training worker is active; no final candidate score has been read.
This is an explicit evaluation-design amendment, not a new seed/top-up for the
failed panel. That failure remains in the result. Protocol42 training is unchanged.

## Why this family

The clean training expansion concerns small u, whereas general small Gordian
distance need not mean either endpoint is near the unknot. To retain that
distinction, prepend a fixed positive two-strand block H=(sigma1)^(2g+1), and
shift the generators of a constructive core K upward by one. The combined word
uses one shared strand. We need not infer pair labels from the construction's
name: replay every combined witness and recompute BOTH signed signatures.
Require |signature|/2>=3 at both pair endpoints, not merely a nonempty word.

Set1024 attempts, seed20261021, core max_strands12, depth8..32,
base g=5+floor(i/3)%8. No retries or top-ups. Attempt kind=i%3:

- 0: one legal COMMUTE of the combined root, with its inverse prefix; if none
  exists, one declared INSERT/REDUCE pair. The reference is the original root.
  This is an isotopy check, not evidence of genuinely harder size transfer.
- 1: a two-CC core, reference immediately after its first CC; prefix upper1.
- 2: two separately generated one-CC cores sharing H. Traverse left-core-to-H,
  reverse the right-core-to-H path, then finish through right-core-to-H and
  H-to-unknot. Pair prefix upper2. Mirror the right core if needed to oppose
  the nonzero signatures of the cores; if a signature is zero, retain the
  resulting interval rather than manufacturing an exact-two label.

Core indices: one-CC index=20*i+i%10, two-CC index=20*i+10+i%10;
the second one-CC core uses20*(i+1024)+i%10. Full path actions are re-encoded
under a common resource specification. Any non-invertible core transition or
failed exact replay is a construction error, not an example to silently accept.
All lower bounds are actual signed-signature differences/2; classify by actual
interval, not intended attempt kind. Fingerprint ownership/exclusions are then
applied to the combined query, not its constituent core.

## Selection and interpretation

Query owner must be test; reference must not be quarantine. Exclude query groups
appearing at either endpoint in the five base EXCLUSIONS and all additional
panels listed by protocol42, plus BOTH freshly generated protocol42 panels
(including the inadequate general panel). First four per(query group,class),
first32 per exact0/1/2/interval, distinct raw cyclic query representations.
Exhaust all1024 attempts. Gate:>=32 pairs,>=8 query groups,>=4 exact examples
of each distance0/1/2. Inadequacy is retained without changing these rules.

Output `artifacts/embedding-common-block-gordian-supplement-v1`. Keep original
views plus the existing zero-CC lift to N33/L512 as a stability view only.
Independently audit full TRAIN paths, every pair witness and every distinct
query determinant with the exact cofactor checker before any model scoring.

Compare the same final parent/candidate/control models on this panel, retaining
constant0/1/2 controls, per-group errors and exact-class errors. Report it as a
distinct synthetic common-block family, never as a completed positive-seed test
or evidence over arbitrary knots. The common structure may be exploitable;
repeated classes/views and related constructions are correlated. This supplement
adds missing general-distance evidence but cannot erase the original test-design
failure or independently establish arbitrary-size solving quality.
