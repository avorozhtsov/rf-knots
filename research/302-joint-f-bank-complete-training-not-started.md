# Joint F bank completed; R2/R4 optimizer still not started

2026-10-07. In response to the user's question, the bank-preparation step
was actually begun and completed, not merely planned. The initial source
`scripts/build_joint_f_bank_v1.py` failed at root 0 before any root shard:
`SemanticAction` could not be JSON-encoded. Its failed source and manifest
remain in `artifacts/joint-f-bank-v1` (research/301). The separately
corrected v2 source SHA
`95b08c52abe0d3c040cbc40e77f05cead0d5174034644d2cb353f02b0724274b`
completed `artifacts/joint-f-bank-v2` at 1,000/1,000 roots. Manifest SHA
`6a61a0ec0bf47c59cdb601b633c962dc4b2d4df3b46ba9a2ca429fab49f19a12`,
contract SHA
`0cb4647e48463aca272dcc932223873db24772e2a00bbf2465dc83b8a15f8dc8`.

Independent `scripts/audit_joint_f_bank_v1.py` report SHA
`2a4a7a19f6d55da6d64533fba41a9594ce85ad89c808205b0984382ba297dc73`
checked every bank-root SHA, pinned input and source-shard SHA, witness
replay/start/caps, reference inclusion and exact F coefficients. It
confirms **392 distinct F-usable TRAIN roots, 690 root/objective/resource
groups and 9,798 stored completed paths**. By stage/objective: tight-cap
stage0 L10/L1000 103/95, stage1 174/167, stage2 74/63; +16-move
stage3 6/8. This is a verified data artifact, not evidence of model
improvement. The root count remains far below the 1,000+ full-curriculum
target but exceeds the research/273 minimum of 200 for a short matched
R2/R4 start.

No new R2/R4 optimizer step or checkpoint exists yet. The immediate
remaining work is to freeze and implement the actual paired optimizer
contract: identical parent checkpoint, bank, sampled roots, step/CPU
accounting, two versus four shared-weight graph rounds, F-dominant
gradient with measured auxiliary/rehearsal norm ceilings, native-MPS
checkpoint/AdamW/RNG resume, and a fresh independently fixed DEV panel.
Only real optimizer checkpoints for both arms start the 24-active-hour
comparison clock. Do not call this bank construction neural training.
