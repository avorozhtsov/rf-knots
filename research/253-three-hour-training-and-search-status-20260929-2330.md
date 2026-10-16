# Three-hour check: paired6144, searches72/64 conditions

2026-09-29,23:30–23:33 UTC. Continue user's72h protocol248 paired training,
not an early stop or model promotion. No source, loss, optimizer, data condition,
checkpoint, search or frozen artifact modified. No duplicate worker, new test,
heuristic campaign, commit, merge, push, cloud purchase or publication.

## Training248 verified active

Process51397 still runs the exact continuation command under native-MPS/no-CPU-
fallback contract; CPU42.5%, accumulated310m29.91s, wall12h00m44s. AC-only
sleep assertion54133 attached. Saved common boundary6144, with matching
shared draw digest, both Python RNG streams and relative-group counts.

- Control7168 cumulative steps, checkpoint SHA
  c6e65fcc133908041b90953a342e8b63abefdb1458e039158227f93760a35942,
  committed17857.746418s. It has completed its next sequential block and
  awaits the long arm; this is not an independent stopped worker.
- Long7040 cumulative steps, checkpoint SHA
  8c434a9dae7f77ca0a7bfe6ce885ba9a7a7fc4403c4020f48e5a163c2e915d3c,
  committed24368.326874s, actively updating toward target7168.

All1971 bound input hashes and contract digest rechecked. Latest full
checkpoints, immutable64-step history/shard hashes and DEV-report hashes pass.
All learned model/objective tensors finite, every AdamW step equals cumulative
steps, metric-pair counts equal32*steps, relative counts equalsteps/16. No
exception/NaN in log. Current total committed training-phase time including
temporary unmatched lead42226.073292s=11.7295h=16.29%of72h. At last common
boundary parent stores35409.999582s=9.8361h; do not add it again.73GiB disk
available. Current CPU/process view works; GPU utilization not separately
measured. The last diagnostic DEV remains4096 (see251); next at8192 for each
arm. No use of already viewed TEST150 for checkpoint selection or early stop.

## Independent old searches, no new248 checkpoint

155v2 control-final-uniform worker23761/controller574 both live; worker98.5%
CPU,72/128 conditions=36starts,430episodes,221229 neural-policy evaluations,
221593 full moves.3 distinct solved roots, L10=2/36,L1000=1/36. No new found
solution in the latest16 conditions. Last shard SHA
5567705201fa3329f98d7460ba9f1681ceefbb19a6a2c2beb93c68b24a3bf458.
Uniform tree PRIOR does not make this a purely heuristic search.

139 raw-final-distance worker31474/controller20100 live; worker98.4%CPU,
64/128=32starts,382episodes,225731 neural-policy evaluations,226053 full moves.
Two distinct solved roots, L10=1/32,L1000=1/32, unchanged from previous
prefix. Last shard SHA
4be23c384d85d61ac83f35c42032f32d3af735791174f280a0b734efc4fb85b0.
Do not pool its roots/panel with155v2. All current committed shard hashes and
every successful full witness replay and CC/L10/L1000 cost check passed in this
turn. Full independent replay of *all* censored episodes is still pending for
these newer prefixes; old completed prefix audits were scoped to shorter runs.
Search under finite budget gives censored, not mathematically negative, rows.

Next scheduled three-hour check: verify paired7168 then8192 DEV if available,
new checkpoint hashes, processes, search shards/witnesses and disk. Continue
until the prescribed72h active work and matched pair boundary, then require
fresh independent TEST and eventual genuinely non-neural matched comparator
before any claim of superiority.
