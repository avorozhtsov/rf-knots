# Old139 initial search complete; replay active; new155 batch profiled

2026-09-26,14:51–14:54 UTC. Read178 before checking manifests, actual commands,
CPU counters, logs and hashes. A real stage transition occurred during the check.

##139 first search configuration completed

Initial observation saw120/128; worker21103 then completed128/128 at14:52:09 UTC.
Final counts:768 episodes,395312 policy neural evaluations,395952 full-path
moves,0 solutions. No success or public-bound improvement is implied. These
are finite-budget results with initial weights, not either trained endpoint.

Final search manifest SHA256:
`1eda550557d6ae4240ffff055dfa16a0109bf81ff4d45220fc8b4ae2d69912e2`.
Last shard `conditions-00120.json` SHA256:
`c6fd4f4fba396e15ba55d58bf00441031095b61f60955adfc2d5f5205f40f29d`.
Total committed-batch time109421.890577s (30.40 hours, not uninterrupted elapsed
runtime or an isolated hardware benchmark). Last batch2135.674135s (35.6min).

Controller20100 automatically committed its13th of24 jobs and launched
`search-audit-units-initial-distance`, PID34989, at14:52:22 UTC. Actual command
is `scripts/audit_resource_search.py`, writing the existing search directory's
`replay-audit.json`. Live verification:96.9%CPU,28.65s CPU in31s elapsed.
Audit has NOT yet passed; do not report complete semantic validation before its
terminal report. Controller should proceed to the next frozen configuration
after successful audit. No manual replacement process was launched.

##155-v2 delayed block: finite diagnostic, no restart

Controller574/session17201 remains8/18 jobs; worker7097 remains active in
control-initial-distance at48/128,288 episodes,136080 policy evaluations,
0 solutions. No new saved block since10:03:06 UTC. CPU grew to869m14.89s from
814m27.65s last hour (98.4%CPU); second launch-transition check870m16.70s.

Because this block has run nearly five hours, took one one-second native stack
sample (10ms interval), without changing the search. Session29390 exited0;
native report `/tmp/Python_2026-09-26_155222_s47n.sample.txt`. Main thread had89
samples,76 in the PyTorch conv2d/convolution path, including NNPACK/NEON numerical
kernels. Physical footprint427.9MB, peak1.5GB. This supports active neural
inference, not a sleeping/blocked main thread at that instant. It does not prove
inner-loop completion or provide a reliable ETA. Larger current roots/caps are
documented in178. Preserve the worker; do not restart based on elapsed time.

## Integrity and next step

Read-only verification session85508 exited0 at14:52:28 UTC, after the139 stage
transition:1459/1273 controller input pins,8/13 completed report hashes, both
roots hashes,6/16 committed search shards and both149 final checkpoints match.
No matching traceback/runtime/nonfinite/NaN error in search logs. New audit log
has only the known optional Plink GUI warning. All commands use canonical PGX.

Continue the automatic139 replay/next-stage sequence and unchanged155 search;
observe hourly. No new optimizer updates or distance results. Useful minimum-CC
search improvement, superiority over fully non-neural heuristics and public
upper-bound improvement remain unestablished. No experiment source mutation,
cloud purchase, database publication, Git publication, interruption or duplicate.
