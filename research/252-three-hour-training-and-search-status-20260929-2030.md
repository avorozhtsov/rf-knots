# Three-hour check: paired5120, searches56 conditions

2026-09-29,20:30–20:34 UTC. Read-only checks except this note. Do not alter
protocol248/149/139 sources, checkpoints or ongoing workers. User explicitly
requested72h of active paired continuation, monitored every3h; completion of a
checkpoint or older MCTS pilot is not the scientific end goal.

## Live training, no duplicate

Exact `scripts/continue_long_general_pair.py` command/process51397 still running,
CPU39.2%, accumulated CPU237m00.11s, elapsed9h01m15s; AC-only sleep assertion
54133 attached. Parent manifest contract SHA
48ecf1c8a7c8df2b76803fd9961706b69fddf8042ff4965fdadd402d32dcea29;
1971 input pins independently rechecked. Full checkpoint/history readback passed:
hash, config, arm, every64-step entry, finite tensors, AdamW step counters,
32 metric pairs/update, relative-group frequency and all committed DEV hashes.

- First two arms completed a common5120 steps. Their shared sample digest,
  historical/long RNG states and relative-group counts match exactly at this
  boundary. Parent committed paired work27502.609772s=7.6396h.
- Control currently6144 cumulative (+4096 since original149), checkpoint SHA
  f819bf39b33a620f516fd2ebbc03378eb88d4623e535b74a8e9711a54346afb0,
  14785.865138s additional work. Status `round_completed` simply awaits long.
- Long currently5248 cumulative (+3200), checkpoint SHA
  73bc4e52cda33303b951b4f0e9efea71a90cf57096fb44329de9b85c147486d8,
  16679.038266s additional work. The expected sequential round target6144.

Including control's unmatched lead, total committed training-phase time
31464.903404s=8.7403h=12.14%of72h. Do not add parent paired time on top of
this total.72GiB free disk. No new DEV was scheduled yet: last reports are at
4096, with the positive-distance caveat documented in251. Next DEV due8192
for each model, not a TEST selection. No error/NaN seen in the log; losses are
finite batch values, not a performance claim. MPS runtime contract unchanged;
GPU utilization was not separately measured in this check.

## Old searches: separate roots/panels and no new248 weights

Both controller and child commands verified live (no use of stale PID alone):
155v2 worker23761 CPU98.3%, parent574;139 worker31474 CPU98.2%,parent20100.

155v2 control-final-uniform:56/128 conditions=28starts,334episodes,
171245 neural-policy evaluations,171529 full moves.3 solved distinct starts:
L10=2/28,L1000=1/28; no new solution from the previous48-condition prefix.
Last shard SHA76ece8897365843021ffb1c691862330d2bc0f7815e732faabe7b099f1892e5e.
Uniform TREE priority is not a non-neural heuristic control.

139 raw-final-distance:56/128=28starts,334episodes,187275 neural-policy
evaluations,187557 full moves.2 solved distinct starts: L10=1/28,L1000=1/28.
New condition48/root24/L10 found CC1 and358 full moves, native costs368/1358.
Prior root4/L1000 remained CC1/moves161/costs171/1161. Last shard SHA
37695475cedab317c41aec5e61bbdd6c4a08c7a0a9600ba957977486a176e2e7.

Every currently committed shard SHA and every found complete solution witness
was independently replayed; CC and both native costs agree. Full replay of all
censored/successful episodes was previously complete only through155v2's
48-condition prefix and139's32-condition prefix. The newer56-condition prefixes
still await post-exit independent audits. A censored budget is not proof of no
solution; do not pool139 and155 percentages or claim heuristic superiority.

Next: leave all healthy workers alone. At the next3h check inspect paired-step
alignment, fresh checkpoint hashes, process CPU/commands, any new search shards,
disk, and signs of training failure. No new TEST150-based checkpoint selection,
heuristic campaign, cloud spend, publication or source mutation authorized.
