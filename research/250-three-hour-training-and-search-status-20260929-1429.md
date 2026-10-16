# Three-hour check: first matched training round complete

2026-09-29,14:29–14:36 UTC. Continue248's72-hour active-training budget;
do not stop at first pilot/checkpoint. No experiment source, condition, optimizer,
data mixture or prior result changed; no restart, duplicate training/search,
commit, merge, push, cloud spending or publication.

## Native training VERIFIED ACTIVE

Current process51397 matches the complete248 launch command. At14:29 CPU51%,
CPU time78m00.18s, wall2h59m47s; AC-only sleep assertion54133 remains attached.
Readback snapshot:

- control3136 cumulative steps,1088 additional;3940.629629s additional committed
  training work; checkpoint SHA
  5d70b9d981f31b7b5552999daecaf8cedcb9df345a4ee61e463fca0452ad6042;
  17 new immutable metrics/checkpoints, latest manifest mtime14:26:24UTC.
- long3072 cumulative steps,1024 additional;6259.267527s additional work;
  checkpoint SHA
  4a59122799022c207135bfdc31543f78e6ab4928e3e8946488cb3559afd057d9;
  16 metrics/checkpoints, latest manifest mtime14:19:41UTC. Waiting for matched
  arm means sequential scheduling, NOT a stopped worker needing restart.

First matched boundary3072 verified independently: both Python sampling RNGs,
shared draw digest, relative counts and originalQ254 reference digest match.
Current64-step lead of control is expected during the next sequential round,
whose common target is4096. Every latest AdamW step counter equals its model's
cumulative step. Metric counts equal32*steps; relative counts equalsteps/16;
all model/objective tensors finite, checkpoint hash/config/arm/contract agree,
both complete committed histories hash-check and contain every required64-step
entry.1971 bound inputs and contract digest checked unchanged. Gate152 independent
consumer again passed1426 pins and recorded parity/resume requirements.

New committed training work totals10199.897156s=2.833305h,3.94%of72h. At the
completed matched boundary the parent records9819.407963s; the higher sum includes
control's next64-step block. Report these separately, do not present parent
boundary time as all current work.73GiB disk free. No exception/NaN in training
log; only the known unusedPlinkGUI/tkinter warning.

Latest batch losses (NOT matched DEV or search quality): control metric0.094297493,
positive1.866924853; long metric0.123369184,positive0.358310438. Different batches
and cumulative steps prohibit an improvement claim. New DEV reports not yet due:
historical and147DEV diagnostics are first scheduled at4096 for each arm. TEST150
is not used for checkpoint selection. New independent finalTEST/audit pending.

## Old searches advance; these are NOT new248 model results

Both exact commands and parents verified:23761(control-final-uniform,155v2)
CPU88.3%,523m02.37s;31474(raw-final-distance,139)CPU87.7%,489m38.35s.
Controllers574/20100 remain active,14/18 and18/24 completed jobs respectively.

155v2 control-final-uniform prefix48/128 conditions=24starts,286episodes,
132981policy evaluations,133225full moves.3 solvedconditions on3distinct roots
(3/24=12.5% of viewed starts):

- condition22/root11/L10:CC1,moves152,costs162/1152;
- condition27/root13/L1000:CC1,moves271,costs281/1271;
- NEW condition44/root22/L10:CC2,moves114,costs134/2114.

Separate objective rates: L10=2/24,L1000=1/24, NOT3/24 in either objective.
Six shard hashes passed; newest conditions-00040 SHA
cdba3e5edab1f4c2a3b3edd09e43e85f2f4016d626bff0ee8a47bdd16cd930ff.
Uniform TREE priority retains neural policy continuations; not a heuristic.

139 raw-final-distance prefix32/128=16starts,191episodes,116325policy
evaluations,116487fullmoves; one solution(root4,L1000,CC1,moves161,
costs171/1161); L10=0/16,L1000=1/16. Four shard hashes passed; newest
conditions-00024 SHA
7839b13a1deba8639806cbe483df0e48d071c638c88d475fb05ef4c1c71e115d.
Do not pool139 and155 roots or count a censored episode as a proved negative.

ALL four found solution witnesses fully replayed and both native costs verified
in this check, without neural forwards or training. A separate READ-ONLY full
prefix auditor also validates every censored/successful episode, tree/resources,
RNG identities and accounting. It is still running at14:36 (process12528,
CPU95.8%,5m26.44s, toolsession24395); collect its result at the next check,
do not launch a duplicate or call that full audit passed yet. It snapshots
48conditions then32conditions; no search/experiment files are written.
This diagnostic process is not an additional search/training worker.

Next: leave healthy training/search running; at the next three-hour check inspect
both4096 DEV reports if committed, new checkpoints and pending prefix-audit output.
No large pure-heuristic campaign or superiority claim at this stage.
