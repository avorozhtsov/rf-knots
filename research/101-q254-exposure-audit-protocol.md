# Bounded registered-root exposure audit for the frozen Q254 snapshot

2026-09-21. Read-only provenance analysis while97 training and its queued
evaluation continue unchanged. Do not replace the Q checkpoint, alter TEST
membership, or retune optimization from this diagnostic.

The frozen model manifest records23 completed roots and an ordered-prefix hash.
The associated stage registers50 new roots after204 prior registered roots;
its current final report records50. Verify the23-root prefix against the frozen
hash, both banks against the native sorted-JSON hashes, stage-local bank copies,
the inherited state hash, the frozen checkpoint, and current state identity.
The current state may legitimately differ from the historical snapshot; retain
that difference rather than using it as the frozen model's original bytes.

Compute exact determinant/signed-signature fingerprints for all254 registered
roots. Compare both227 prior-plus-completed-prefix roots and all254 stage roots
against frozen70/74 validation endpoints and70 lifted endpoints, keeping left
queries and right/reference endpoints separate. Count exact cyclic-word matches
separately from determinant/absolute-signature collisions.

This cannot establish full historical independence: inherited pretraining and
all MCTS/crossing-change trajectory states are not enumerated. A stage snapshot
with23 completed roots may also contain partially processed next-root exposure.
A matching fingerprint does not prove knot identity; no matching registered
root does not prove a knot was unseen. Report `independence_proven=false`.

The Q254 name is a generation/stage label, not evidence of a terminal254-root
checkpoint or of globally best policy quality. Preserve this qualification in
the final interpretation of the current frozen transfer experiment.
