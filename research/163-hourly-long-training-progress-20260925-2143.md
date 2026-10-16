# Hourly check: long-data training progressing, search active

2026-09-25, 21:43–21:48 UTC. No experiment was restarted or duplicated.

## Training149

Runner PID20138/session98193 remains live with the registered command using
canonical pgx-mcts-bench. At the live process check it used39.7%CPU and had
77m53.56s accumulated CPU time, versus54m11.96s at the preceding hourly check.
Control remains completed at2048/2048. Long-data training advanced from32 at
the previous verified snapshot to992/2048 (48.4%) in this finite audit.
While inspecting, committed manifests advanced928→976→992; this is actual
file progress, not inference from a historical PID.

Read-only consumer session24106 exited0 at21:47:17 UTC. It verified all1439
parent input hashes, control's completed-arm contract/checkpoints/history/DEV
hashes, long's62 committed metric files and two initial DEV hashes, and the
captured long checkpoint:

`long/step-000992-8003156a9ef69618.pt`

SHA256: `8003156a9ef69618b85caf585a13f52bec5a6ab0258e82bbd46eac5ee91c453c`.

All model/objective/optimizer tensors checked finite; all67 optimizer states
record exactly992 updates. This is snapshot integrity, not the complete paired
data-stream reconstruction or an independent quality result. Training losses
fluctuate between batches; no test improvement is inferred from them.

## Search139

Controller PID20100/session23235 and its single worker PID21103 remain live.
Worker96.4%CPU, accumulated CPU183m01.99s versus118m49.75s last hour. Same
units-initial-distance configuration, no replacement process.

Committed prefix remains40/128 conditions,240 episodes,138760 policy neural
evaluations,0 solved conditions. Five committed shard hashes and roots hash
all match. Last committed block remains20:08:17 UTC; there is no new search
result to report this hour. Previous batch took about84 minutes, and earlier
batches were longer; unchanged prefix plus continuing CPU work does not justify
declaring a stopped worker. Search logs contain no matched traceback/runtime
error/nonfinite/NaN indication. Full semantic replay is still pending.

## Next action

Continue both healthy jobs under the unchanged frozen contracts. Evaluation155
output does not exist and must wait for both149 arms and parent to complete.
Then launch the registered independent pair audit, distance evaluation, search,
and replay pipeline. Keep hourly monitoring. No established advantage over
fully non-neural heuristics or new public u-bound improvement is claimed.
No source conditions, cloud resources, public database or Git publication changed.
