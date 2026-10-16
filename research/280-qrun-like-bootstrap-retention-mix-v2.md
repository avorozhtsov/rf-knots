# Corrected first rung with length-stratified retention mix

2026-10-04. The frozen v1 first block stopped at 16/256 optimizer steps.
On a fixed TRAIN diagnostic of previously unseen reference actions, mean
negative log probability changed 0.6292 -> 0.4703 on 20 short roots but
1.6729 -> 1.9274 on eight long roots. Top-1 counts were 17 -> 18 and
6 -> 6. The checkpoint SHA256 is
`96183f5630ad2ff6470a4310ee10151c90093b40ad1342b075d4bfe0b1295e17`.
This is a retention warning, not a completed search evaluation. Preserve
the v1 source/artifact and its initial missing-pgx import failure; do not
continue v1 to 256 steps merely because TRAIN loss fell.

For a separate v2 controlled repair, start again from the same F144 parent
SHA256 `0deb9418ea71a432644b7370c84879cf887536dcd3757280466e73ed749ff227`.
Use 140 distinct replay-verified TRAIN roots per cycle: first 105 stage-0
roots, plus first 12 stage-1, first 12 stage-2 and first 11 stage-3 roots,
interleaved three short to one longer in each batch of four. Within a
root, sample lambda and a witness position using checkpointed RNG. Keep
the v1 policy loss, 256-step endpoint, AdamW settings and 16-step checkpoint
interval. This is an explicit 25% length-stratified stability mix in the
first rung, not F-dominant learning or a claim that the long roots are
already mastered. No failed search supplies a negative label.

After the first 16 optimizer steps, measure previously unseen reference
action likelihood on short roots and a disjoint long subset before deciding
whether to resume to the fixed endpoint. Any long regression triggers a
separate versioned repair, not a silent edit to v2. Later rungs should
increase the fraction of long roots while retaining short examples, add
fresh verified search solutions and eventually make F the main gradient
once distinct audited F roots are abundant. Search benefit and retention
must be checked at matched resource budgets; imitation loss alone is not
the user-facing outcome.
