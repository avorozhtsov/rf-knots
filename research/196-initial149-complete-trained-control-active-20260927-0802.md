# Initial149 search/replay complete; trained149 control is active

2026-09-27 07:59–08:02 UTC (08:59–09:02 London). Read195 and current manifests
before checking live process commands, CPU advancement, logs and artifact hashes.

155-v2 shared-initial149 search completed07:32:02 UTC:128/128 conditions,
768 episodes,376496 policy evaluations,377136 full-path moves,0 solutions under
the declared budgets. All16 shards present. Manifest SHA256:
6078efd39ce5ee3240f27b947caad019917b63b8b512df7d8532ec0899124200.
Full replay audit passed: all successful/censored paths and reference witnesses
replayed; hardware/cache/timing counters recorded rather than recomputed.
Replay report SHA256:
f0d640c0a458f6980d4f539219bbeb1e4a35cdf4519a2ac80733de5e852a725c.
Committed batch time112662.082094 seconds (31.30h). This is measured evaluation
work, not training or evidence that these roots cannot be solved.

Controller574 automatically advanced10/18 completed jobs and launched
search-control-final-distance, PID36622, at approximately07:58 UTC. Fresh
manifest07:59:30 UTC,0 committed conditions/shards. It uses the audited149
control checkpoint after2048 training updates (SHA89c109f06d19f21f6b8ccd06c2e0e619155e962f26e333446d6d094d97e23766).
Actual command verified against canonical pgx-mcts-bench. CPU advanced from
1m40.22s to3m01.01s during this check;99.4–100% CPU. VERIFIED ACTIVE, no scored
block yet. Old initial worker7097 is absent after normal completion; no restart.
The long-data trained arm remains later in the unchanged declared sequence.

139 trained135 worker47974 remains live99.3% CPU with965m46.31s accumulated
versus905m54.62s last hour. Controller20100 remains14/24 jobs. Committed prefix
unchanged:56/128 conditions,329 episodes,182039 policy evaluations,182324
full-path moves,9 successful conditions covering7 roots. Last saved block
04:29:56 UTC. No new success; full trained-search accounting audit still pending.
CPU advancement is not a claim of exact within-block progress.

Read-only session68855 exited0 at08:01:32 UTC: all1459/1273 controller input
pins,10/14 completed-report hashes, all four search roots/checkpoint hashes,
0/16/7/16 shards, and both149 final checkpoints match. Independently reconsumed
both NEW completed jobs with run_long_general_evaluation_v2.verify_job:
search coverage/resources/model and replay report source/input/manifest/model
bindings passed. This consumes the completed full replay certificate; it does
not rerun every trajectory a second time. No error matches in logs. Optional
Plink tkinter GUI-unavailable warning is not an experiment failure.

Continue healthy trained-control149 and trained135 searches, then remaining
replays/comparisons. No optimizer updates this hour or improved public CC bounds;
no established superiority over fully non-neural heuristics. No confirmed bug
required a fix. No duplicate, interruption, frozen-source mutation, cloud
purchase, database publication or Git action. Dirty work and artifacts preserved.
