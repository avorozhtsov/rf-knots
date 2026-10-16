# Hourly check: long149 second replayed solution, paired prefix2 versus1

2026-09-28 11:53–11:55 UTC (12:53–12:55 London). Read223 and current
manifests before actual process, command, integrity and witness checks.

Long-final149 advanced16→24/128 conditions,142 episodes,68369 policy neural
evaluations,68490 full-path moves,1→2 successful conditions on2 roots.
Manifest11:41:13 UTC,3 shards; latest `conditions-00016.json` SHA256:
3a1d1e8cab9a19d0811dea39dc770c460f065ac792a8ac739612e2f0dc954154.
Independent full-witness replay passed both solutions with exact roots/action
dimensions/CC and move caps and recomputed native costs:

| Condition | Root | Search | CC | Moves | L10 | L1000 |
| --- | --- | --- | --- | --- | --- | --- |
|6|3|L10|2|252|272|2252|
|22|11|L10|1|178|188|1178|

Identical24-condition/12-root prefix: long L10=2/12,L1000=0/12,control
L10=1/12,L1000=0/12. Long adds root3, loses no control success so far.
Common solved condition22 has equalCC1 but long uses178 versus control126
moves,52 more. Reference root11 usesL10=134,L1000=1124, so neither improves
reference costs; long's new route is54 moves beyond the reference at equalCC.
More solved roots in this tiny prefix does not establish cost, embedding or
heuristic superiority. Eventual full all-episode/tree audit remains required.

Worker79573 actually running100% CPU,245m02.94s versus184m59.10s last
hour:60m03.84s additional CPU. Controller574 remains12/18 jobs, tracking
the adopted existing long worker without duplicate/restart. Completed control149
remains128/128,6 distinct solved roots,L10=5/64,L1000=1/64,full replay passed.
Its report/manifest checksums remain intact; no optimizer updates this hour.

139 raw-initial-distance remains48/128 conditions,288 episodes,
165072 policy neural evaluations,165312 full-path moves,0 successes,6 shards;
last manifest10:40:44 UTC. Worker96642 actually running100% CPU,993m36.00s
versus933m31.61s last hour:60m04.39s additional CPU. Controller20100 remains
16/24 jobs. CPU confirms activity, not exact unsaved progress. Raw neural
variant is not a fully non-neural comparator; timeout is not a restart trigger.

Integrity verification11:54:49 UTC passed all1459/1273 controller input pins,
12/16 completed-report hashes, six roots/checkpoint hashes,16/16/3/6/16/16
committed shards and both149 final2048-step checkpoints. No traceback/runtime/
error/nonfinite/NaN log matches. Authorized read-only process inspection
confirmed both workers/two controllers and canonical PGX commands. No confirmed
defect needs a fix. Preserve dirty work/artifacts/checkpoints and frozen sources.

Continue healthy searches and paired-prefix reporting; complete full matrix/
replay and preregistered matched fully non-neural comparison before promotion.
No cloud purchase, database publication, commit, merge or push. Limited-budget
failures remain censored rather than proven negative labels.
