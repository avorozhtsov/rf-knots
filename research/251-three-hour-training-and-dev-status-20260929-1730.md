# Three-hour check: matched step4096 DEV, live step4416

2026-09-29,17:30–17:33 UTC. Continuing protocol248 in its original, pinned
artifact namespace; no source/condition/optimizer edits, restarts, duplicate
training/search workers, commits, pushes, merges, cloud resources or publication.
Same ACTIVE three-hour heartbeat/thread verified; AC-only sleep assertion remains
bound to training. No user action required while these processes remain healthy.

## Training integrity and cost

Host process51397, exact `scripts/continue_long_general_pair.py` command,
CPU51.5%, CPU time159m41.01s, wall6h00m41s. Control now cumulative4416 updates
(2368 additional), checkpoint SHA
3daa225aa4123bda0b63aba5e19b273e8c7c86117144e891ffc2962a03cc5fa5,
committed9357.769060s. Long now cumulative4096 (+2048), checkpoint SHA
8c1aebbf8e0ac3db687bb34c9f32736e74e70a093019e189e071e07879e85570,
committed11505.117018s. The temporary320-step lead is expected because arms run
SEQUENTIALLY; parent matched boundary4096 is complete and target5120 per arm.
At4096 both saved Python RNG streams, shared draw digest, relative counts and
Q254 reference identity independently matched. `long` status
`round_completed/waiting_for_matched_arm` is NOT a stopped worker.

Independent readback: contract digest, all1971 pinned inputs, latest checkpoint
hash/config/arm, every64-step metrics hash/history, DEV report hashes, finite
model/objective tensors, exact AdamW step counters,32 metric pairs per cumulative
step and one relative group each16 updates all PASS. Saved work at matched4096
is19802.158172s=5.5006h; including the control lead, current committed work is
20862.886077s=5.7952h,8.05%of72h.69GiB disk available. These are committed
training-phase wall times (including checkpoint overhead), not MCTS results or
the separate initial data-loading time. Batch losses finite but not interpreted
as paired quality metrics. No exception in current training log.

## DEV diagnostics (2048 parent versus4096 continuation)

Both historical(256 records/25 groups) and147 fresh_general DEV(1082 records/
994 groups) reports are present and hash-verified. No TEST150 use or checkpoint
selection. Mean interval violation, lower is better:

| Arm | Historical2048→4096 | Fresh general2048→4096 | Fresh positive-only2048→4096 | Fresh positive constant1 |
| --- | ---: | ---: | ---: | ---: |
| Control |0.6135→0.5327|0.7188→0.7115|1.1017→1.0904|0.1827|
| Long |0.5772→0.5139|0.3641→0.3167|0.5558→0.4835|0.1827|

Positive-only subset has706 records (`lower>=1`). The long arm's mean on all
fresh records beats the constant1 all-record score0.4492, but this is heavily
influenced by357 exact-zero records where both models are nearly perfect.
On positive fresh distances, BOTH models remain much worse than constant1;
long exact1 violation0.6036, exact2 violation0.6491 at4096. Thus DEV shows
partial fitting/calibration improvement, NOT yet useful positive-distance
embeddings, end-to-end search utility or a reason to stop the72-hour run.
Historical positive-only long0.6448→0.5577 versus constant1 0.4183 also remains
above baseline. Full independent new TEST/search is still pending.

## Older searches, kept separate from248

155v2 old149 control-final-uniform worker23761/controller574 live, CPU97.6%,
unchanged committed48/128 conditions=24starts,3 solved distinct roots:
L10=2/24,L1000=1/24.286 episodes,132981 neural policy evaluations,
133225 full moves. Uniform tree priority still has neural policy continuations.
Last shard SHA cdba3e5edab1f4c2a3b3edd09e43e85f2f4016d626bff0ee8a47bdd16cd930ff.
No new batch during this interval; high CPU alone is not an extra result.

139 old raw-final-distance worker31474/controller20100 live, CPU98.1%, now
48/128=24starts (was32/128), one solved: L10=0/24,L1000=1/24;
287 episodes,164245 neural policy evaluations,164487 full moves. Latest shard
SHA ae0602cb9e38b5da7b9cb110cc95b70b288bb02bb8bdc7645a880427dc3e7792.
These roots/panels are NOT pooled with155v2. Lack of solutions in a censored
condition is not a mathematical negative.

All current committed shard hashes and every found solution's complete witness
and CC/L10/L1000 cost passed independent readback. The previous diagnostic
read-only prefix-audit session24395 exited0: its loop replays all48 conditions
for155v2 and then the previous32-condition139 prefix, including censored episodes
and accounting; no experiment files were written. The NEW139 extra16-condition
prefix has not received a full episode-level replay yet; do not call that full
48-condition prefix audited. Final post-exit audits remain required.

Next scheduled check in three hours: verify live commands, checkpoint progress,
hashes/counters and any newly committed search shards. Continue to later
predeclared DEV diagnostics; independent fresh TEST and genuinely non-neural
matched heuristic comparison are still deferred as described in248.
