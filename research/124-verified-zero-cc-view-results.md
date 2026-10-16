# Larger replay-certified TRAIN views: engineering gate passed

2026-09-21. Protocol123 produced96 pair variants with192 distinct raw endpoint
representations from the32 fixed TRAIN parent pairs. This is an isolated data
building block; no model was trained/evaluated, no policy targets were generated,
and the current MCTS, source corpus and frozen training code were unchanged.

## Saved evidence

Directory: `artifacts/raster-zero-cc-view-gate-v1/`.
Report SHA256:
`eaf4e2743545826ec0cf3e334b7cd38045c2b7a8cc9b75abb0bfe2545951a34c`.
Input SHA256:
`27dfc3cc935248350d5ce60a1045fb501545f7118135f9117475d7b9afed25da`.
Variants SHA256:
`ac3b5ecac267b0a6fbab1170f874e3d51ee826d7fa27a9b7f23e5d613f9ec8d9`.
The variants file is4366047 bytes and retains every operation descriptor,
endpoint, seed, parent-row identity and inherited interval/priority metadata.

All1225 input/source pins were rechecked. Every SAVED endpoint certificate passed
forward reconstruction with valid generators and a one-component closure after
every intermediate operation, followed by exact reverse replay to its source.
A second process regenerated all192 endpoints from saved seeds and required exact
word, strand-count and operation-list equality. Both processes exited0.
The original augmentation tests plus12 new strict-view tests passed18/18; Ruff
passed. Zero neural forwards and zero optimizer steps were performed.

| Augmentation rounds | Endpoints | Word lengths | Strands | Operation descriptors |
| --- | ---: | ---: | ---: | ---: |
| 1 | 64 | 20–247 | 3–19 | 1109 |
| 4 | 64 | 106–329 | 7–23 | 4602 |
| 16 | 64 | 426–789 | 22–45 | 18335 |

Totals:17272 inverse-pair insertions,3396 conjugations,1344 cyclic origin shifts
and2034 stabilizations;24046 descriptors. All192 endpoints differ from their
parent and from one another in raw representation. This does not create192 new
knot types: each endpoint deliberately retains its respective parent's type.

## Empty-anchor boundary condition

27 endpoint certificates start from the empty one-strand unknot. The legacy
insertion/conjugation helper can request generator1 even when the input has only
one strand. A one-strand braid has no generator1, so reverse-word equality alone
would not establish validity of that intermediate braid.

The new isolated wrapper first performs a valid stabilization of the empty
one-braid to `(word=(1,), strands=2)`, before calling the existing helper. The new
forward verifier rejects the unguarded one-strand insertion and malformed
stabilizations, invalid generator ranges, altered endpoints and mirror operations.
The pinned legacy helper was NOT edited. This boundary finding does not by itself
invalidate any earlier experiment: its actual inputs/call paths would have to be
audited. Protocol97 itself does not apply this online augmentation in training.

## What these certificates prove—and do not

Insertion of an inverse pair, braid conjugation, cyclic origin changes and valid
Markov stabilization preserve the closure's knot type without crossing changes.
Both transformed endpoints therefore inherit the same certified CC interval as
their parent pair. They do not receive a newly calculated exact distance or a
new lower bound. No unilateral mirroring is permitted.

These are zero-CC MACRO certificates, not native MCTS action traces. Their
descriptor counts are not L10/L1000 move costs; especially origin changes and
conjugations must not be assigned an invented native step cost. Parent witness
semantic length is retained separately and unchanged, so simply adding view
operations does not silently increase sample priority.

This gate demonstrates deterministic size growth and exact structural replay,
not learned invariance, useful novel mathematical content, additional self-play
games, improved Gordian/u calibration or better search. Many operations insert
canceling structure, so raw uniqueness and larger dimensions alone are not a
claim of harder or more diverse knot types. The size bound of this gate is not a
bound on the architecture's accepted inputs, nor a proof of arbitrary-size learning.

## Next integration boundary

This building block can be considered in a NEW TRAIN contract to produce fresh
views of sampled metric endpoints and explicit same-knot pairs, with distinct
seed streams and unique-representation accounting. No augmented policy target
is implied: its legal actions, resource context and move costs would need their
own verified construction. Do not silently replace the active corpus, change its
weights, or insert these macros into UnknotDB as native witnesses.

Combine the data-coverage evidence in122, conditioning evidence in119 and the
still-pending complete paired MCTS/replay results when deciding the next larger
training experiment. No checkpoint promotion or release follows from this gate.
