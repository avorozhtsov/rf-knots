# Hourly check: second uniform root; extended training proposed, not launched

2026-09-29 11:12–11:14 UTC. Read246 and current controller/search manifests
before real process commands, CPU advancement, logs, hashes and witness replay.
User asked whether training should resume before more evaluation; recommended
a separate checkpoint continuation, provisionally2048→8192 steps, development
checks and fresh independent final test. User has not yet confirmed this
specific new experiment. No extended optimizer run launched; current frozen
sources, weights and healthy searches remain unchanged.

149 control-final-uniform advanced24→32/128 conditions (16 roots),143→191
episodes,69225→88959 policy neural evaluations,69346→89121 full-path moves,
1→2 successful conditions on2 distinct roots;4 shards. Latest log10:43:14 UTC;
conditions-00024.json SHA256
b978202fca22a738d0545d98ff006e50d3bd6bf8bfa7943df3569d32574889e0;
saved block3897.206207 seconds. Independently replayed both full witnesses,
semantic identity at every prefix, exact starts, action-space dimensions,
CC/move caps and costs. New condition27/root13/L1000:CC1,moves271,
L10=281,L1000=1271. Reference71/1061 implies61 moves:210 extra at equalCC,
not a reference-cost improvement. Earlier root11/L10 remainsCC1,moves152.
Uniform L10=1/16,L1000=1/16,either2/16 (12.5%). Same16-root distance
prefix: control solves root11 only; long solves3/11. Uniform uses neural
continuations despite0 tree embedding evaluations, not a heuristic control.

139 raw-final-distance remains16/128 conditions,95 episodes,47237 policy
neural evaluations,47319 full-path moves,1 success onroot4/L1000,2 shards;
last log08:40:36 UTC. Full successful witness replay repeated and passed:
CC1,moves161,costs171/1161,reference89/1079. L10=0/8,L1000=1/8,either1/8.
No new solution or CC/reference-cost improvement. Full audits await completion.

Worker23761 actually running93.3% CPU,345m04.05s accumulated versus
286m04.60s in246:58m59.45s additional CPU. Worker31474 running93.2% CPU,
311m40.45s versus252m40.62s:58m59.83s additional CPU. Controllers574/20100
remain14/18 and18/24 jobs. Actual commands use canonical pgx-mcts-bench;
CPU activity confirms running, not exact unsaved progress. No restart/duplicate.

Integrity verification11:13:24 UTC passed1459/1273 input pins,14/18 completed
report hashes, eight roots/checkpoint hashes,16/4/16/16/2/16/16/16 saved shards
and both149 final2048-step checkpoint hashes. Logs contain no traceback/
runtime/error/nonfinite/NaN matches. No confirmed defect needs fixing or new
optimizer updates; MCTS is evaluation, not training.

Completed149 distance/full audit unchanged long8/64 (12.5%),L10=6/64,
L1000=3/64 versus control6/64 (9.375%),L10=5/64,L1000=1/64. Heuristic
launch remains deferred. Prefixes are diagnostic, not promotion evidence;
overall goal and superiority not established. Next: continue existing searches,
replay new solutions; obtain direction on separate extended training. Observed
TEST must not be reused for checkpoint selection/final independent promotion.
Finite-budget failures are censored, not proven negatives. Preserve dirty
work, artifacts/checkpoints and frozen inputs. No cloud purchase, publication,
commit, merge or push.
