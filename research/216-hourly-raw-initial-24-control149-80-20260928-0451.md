# Hourly check: raw initial135 24/128; trained control149 active

2026-09-28 04:49–04:51 UTC (05:49–05:51 London). Read215 and current
controller/search manifests before actual process, command, log and hash checks.

139 raw-initial-distance advanced16→24/128 conditions,144 episodes,
87432 policy neural evaluations,87552 full-path moves,0 successful conditions.
Three committed shards; manifest/log mtime04:19:38 UTC. Latest
`conditions-00016.json` SHA256:
e7a54b21789623b678b97ada4e0bff4e60c1c971b0d7560c98a93035e11b8c44.
The previously observed long block completed in16852.366443 recorded seconds
(about4h41m); lack of intermediate saves was not a stop. Controller20100
remains16/24 jobs. Worker96642 actually running99.7% CPU, accumulated
570m26.98s versus509m31.42s last check:60m55.56s additional CPU. This is
a neural raw-input variant, not a fully non-neural comparator.

155-v2 control-final-distance remains80/128 conditions,478 episodes,
245206 policy neural evaluations,245608 full-path moves,2 successful conditions.
Ten shards; manifest/log mtime03:17:21 UTC. Controller574 remains10/18 jobs.
Worker36622 actually running98.4% CPU, accumulated1118m10.96s versus
1057m14.29s last check:60m56.67s additional CPU. The long-data149 search
remains queued, so paired evaluation is incomplete. CPU activity does not
quantify unsaved progress in the pending block.

Integrity check completed04:50:33 UTC: all1459/1273 controller input pins,
10/16 completed-report hashes, five roots/checkpoint hashes,10/16/3/16/16
committed shards and both149 final2048-step checkpoint hashes match. Logs
agree with manifest timestamps; no traceback/runtime/error/nonfinite/NaN
matches. Authorized read-only process inspection confirmed both workers and
controllers with canonical pgx-mcts-bench in actual commands. No new success
requires replay and no confirmed defect requires a fix. No optimizer updates
this hour: computation is search evaluation, not training.

Completed units-final135 remains128/128 evaluated,20 successful conditions
on16 roots, with previously passed full replay and no CC/reference-cost
improvement. Initial units135 and initial control149 remain completed with0
successes. No claim of superiority over heuristics is warranted.

Continue healthy searches unchanged, then complete queued arms and the
preregistered matched fully non-neural comparison. No duplicate, restart,
frozen-source mutation, cloud purchase, database publication, commit, merge or
push. Dirty work and checkpoints preserved. Limited-budget failures are not
proven negative examples.
