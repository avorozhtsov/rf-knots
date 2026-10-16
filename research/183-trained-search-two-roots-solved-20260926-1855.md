# Hourly check: trained135 solves two roots;149 initial search halfway

2026-09-26,18:53–18:55 UTC. Read182 before inspecting current manifests,
actual process commands, CPU counters, logs and checksums. No conditions changed.

## Trained135 result advanced

Old139 units-final-distance committed a second block at18:37:13 UTC:8→16/128
conditions,47→91 episodes,21810→43871 policy evaluations,43954 full-path moves,
2→4 solved conditions. There are now TWO solved roots (panel3 and4), each at
both L10/L1000, among the first8 roots. Initial weights solved none of those
matching16 conditions. This is an incomplete descriptive prefix, not a complete
paired advantage or a comparison against fully non-neural heuristics.

Second shard SHA256:
`91ca714adb6d0686ae2414e7cc9bc8a19d7d8f8205fd4eb66a4a5c52397a64f6`.
Block5460.853609s (91.0min). Session92464 replayed both NEW saved witness
certificates, verified exact roots, resource caps and recomputed costs:

| Condition | Root | Search weight | CC | Moves | L10 | L1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|8|4|10|1|79|89|1079|
|9|4|1000|1|81|91|1081|
|existing reference|4|—|1|79|89|1079|

Certified lower_cc for root4 is1, so both attain known minimum CC. The L10 run
matches the existing reference's CC and full-path cost, but does not improve
them. The L1000 run has two extra moves. Root3 results remain those in182.
These are synthetic frozen TEST roots, not improvements to public named-knot
upper bounds. This is135 trained units, not149's long-data model; do not isolate
an embedding-specific causal claim from a jointly trained network.

Only success certificates were replayed in this hourly check. Full episode/tree
accounting replay remains scheduled after128 conditions, with no TEST feedback
into training. Controller20100 remains14/24 jobs; worker47974 live98.8%CPU,
191m22.45s CPU versus132m59.75s last hour.

##149/155-v2 initial search

Worker7097 committed another block at18:33:59 UTC:56→64/128 conditions,
336→384 episodes,174344→205408 policy evaluations,205728 full-path moves,
0 solutions. Block12801.977841s (213.4min). Latest shard SHA256:
`3824775cb6fbed77a87ea9f38acd0a170d6b4bf41a27e46010b6c5a73eca303f`.
Controller574/session17201 remains8/18 jobs. Worker98.9%CPU,1099m05.00s CPU
versus1040m43.34s. This is halfway through its initial-model configuration,
not halfway through the full matrix. Both live commands use canonical PGX.

## Integrity / next step

Session92464 exited0 at18:54:30 UTC:1459/1273 controller input pins,8/14 completed
report hashes, all three roots/search checkpoint hashes and8/2/16 committed
shards match. No traceback/runtime/nonfinite/NaN indication in either log
directory. No confirmed issue required a fix. No new optimizer update occurred.

Continue unchanged searches and full replay matrix, observing hourly. No model
promotion or general superiority claim from this small prefix. No restart,
duplicate, source mutation, cloud purchase, database or Git publication.
