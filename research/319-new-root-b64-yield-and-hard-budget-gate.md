# New-root b64 F-yield result and bounded hard-budget gate

2026-10-10. Incremental roster-v2 independently audited 256 distinct new
TRAIN candidate roots. The protocol318 R2-v4 b64 search finished all128
conditions; manifest SHA
`5955d30ecc30d092087b63ef864c0fe6fab4a9296e0417bf82d2d73df682c2d2`.
Independent audit report SHA
`a42ae6b1b78123aa71e2f4ba9cb2e52fd8e881a83ad37176949e85af83ed60bb`
verifies every shard/input SHA, solution replay, ownership, start, CC/move
caps and costs. It found only **5/32** distinct new F-usable roots:
stage0 4/8, stage1 1/8, stage2 0/8, stage3 0/8; ten exact F contexts.
Search consumed 5,249.319 CPU seconds, or 1,049.864 CPU seconds per
new F-usable root in this small sample. No admitted path beat the constructive
reference on its paired objective. No-solve at b64 is censored, not a
negative action or evidence of intrinsic impossibility.

Do not bulk-extrapolate the b64 pipeline to 10k/30k F roots: hard-stage
yield is zero in eight roots per stage. The selected R2 continuation has
also completed normally at global step133506 with 7,236.248 *new* active
seconds, and all297 checkpoint SHA plus final AdamW/RNG passed. Its endpoint
SHA is `4b11cb1078f14f64cfabd1d65c97826b8eb7fdefb16edec8d07d982c33d4cad2`.
Do not restart either completed experiment or silently modify their inputs.

Next bounded TRAIN-only diagnostic: select 16 previously score-blind roster-v2
roots, the first four CC1 and four CC2 in each of stages2 and3 in SHA order.
Use the frozen R2 step133506 checkpoint, canonical pgx-mcts-bench, original
paired caps, L10/L1000, two fresh seeds20271321/22, 256 real simulations,
fixed insertion4/widening4/c_puct1.5/temperature1/distance prior/tile cache.
That is 64 independent conditions. Freeze source, inputs and contract SHA
before launch; audit full replay, ownership, F cost diversity, exact contexts
and CPU per *new* F root. If hard F yield remains absent or prohibitively
costly, prioritize a different TRAIN source family rather than multiplying
search budget or counting capped failures as negative actions. The old392
F roots and these five new groups must be retained in any later additive
bank. A new R2 optimizer may use them only via a separate frozen contract
with old-root rehearsal. No DEV/TEST150 selection, cloud purchase, database
publication, commit, merge or push.
