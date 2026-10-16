# Full-search diagnostic: completed, audited, and compared with greedy

2026-09-21. Protocol84 completed96/96 conditions, exit0. Its separate audit
passed all96, replaying Q's executed paths and discovered native proofs plus
every successful/censored RF continuation. All408 audit input/artifact hashes
were rechecked after completion. Ten focused tests and the new files' Ruff
checks passed. No training source or previously frozen evaluation was changed.

Artifacts: `artifacts/common-search-policy-diagnostic-v1/`.
Manifest SHA256: `ffd972cac655060a50a8fc90a6d2a606b3e0fdbd24e6a6ef9835862b583add50`.
Audit SHA256: `d13978c13855e3812ef3c15689c993c9e551e102a5a18c319084031d3f1f3c88`.

## Results on the same16 TRAIN diagnostic representations

| Model | Greedy L10 | Search L10 | Greedy L1000 | Search L1000 |
| --- | --- | --- | --- | --- |
| Q254 raster | 2/16 | 2/16 | 1/16 | 1/16 |
| Q304 strand graph | 4/16 | 1/16 | 2/16 | 2/16 |
| RF joint57 | 4/16 | 8/16 | 4/16 | 9/16 |

Q used eight native MCTS simulations PER EXECUTED DECISION with trained values;
RF used eight TOTAL cost-tree simulations with stochastic full-policy
continuations, repeat filtering and learned embedding priors. Equal simulation
labels do not mean equal work or the same algorithm. RF's untrained scalar
value head was not used. No root noise, no checkpoint updates, fixed seed.

Measured inference totals over16 roots at each objective:

| Model | L10 | L1000 |
| --- | --- | --- |
| Q254 actual network samples | 11125 | 10044 |
| Q304 actual network samples | 11116 | 11052 |
| RF policy samples + tree embedding samples | 9610 +80 | 8463 +81 |

Q counters matched actual cache misses. RF policy counters matched independently
replayed continuation moves. These units are NOT matched FLOPs: RF sees/scans
the full word and action catalogue; Q has a local controller. Measured search
loop wall time was approximately21.7/21.8 seconds for Q254,18.8/18.2 for Q304,
73.8/62.0 for RF, with ongoing training and warmup effects. This is not an
isolated throughput benchmark and cannot establish a hardware efficiency rank.

No additional Q proofs were found exclusively in unselected branches in this
run. The collector nevertheless handles that case, with a test ensuring it
retains such a proof across subsequent controller decisions.

## Coverage improves without guaranteeing lower cost everywhere

RF gains four L10 successes and loses none compared with greedy. At L1000 it
gains six but loses index8, for a net increase from4 to9. On common successes:

- L10 index7 improves33 ->31; index42 remains19. Index2 worsens20 ->41,
  and index27 worsens26 ->29.
- L1000 index0 improves1008 ->1006; index42 remains1009; index27 worsens1016 ->1019.

Thus reporting only8/16 and9/16 would hide real cost regressions and one lost
solution. Successful proofs are upper bounds and must be retained, not replaced
by a worse subsequent search outcome. Budget exhaustion is still censored.
RF continuations stopped by solved/move-cap18/72 at L10 and24/63 at L1000.
More search may help, but these counts alone do not prove it will.

An OFFLINE union of the two already audited runs, keeping the cheaper proof
for each root, gives RF8/16 L10 and10/16 L1000; Q3044/16 and3/16; Q2542/16 and1/16.
This is a post-hoc portfolio of saved proofs, NOT another executed experiment
or free extra compute. It includes the costs of both runs and must not be
reported as the standalone MCTS result. A prospective search can first run a
cheap policy attempt and preserve that MODEL-GENERATED incumbent while searching.
Do not seed with the supplied reference witness and then claim it was discovered.

## Interpretation and next decisions

1. On this short diagnostic the current RF policy/search is functional, and its
   low old long-input success rate is not explained simply by failing to load Q
   weights. Neither this nor greedy82 establishes embedding benefit: no
   embedding-disabled matched comparison was performed here.
2. Q304's small-budget search can be worse than its greedy policy. This does not
   prove its MCTS implementation or trained value is wrong. Native8 simulations,
   one attempt, hardCC2/PASS mask and128 native plies are different from its
   historical40-simulation evaluation. A larger native-budget check is a separate
   diagnostic, not permission to retune this completed run or rewrite its result.
3. Preserve independently found incumbents. If comparing search mechanisms,
   separate stochastic rollout and repeat avoidance from tree/value guidance;
   do not attribute the whole bundle's gain to one component.
4. Keep the global architectural objective: reuse learned local computations
   in a network supporting long words/many strands, with global context and
   positional information. This common-domain experiment has NOT implemented
   pretrained raster tiling and does not replace that work with residual heads.
5. Finish the already-running graph/control training and its independent
   distance/long-input evaluations before drawing a scientific promotion
   conclusion. At this report control2240/4096 was advancing, graph4096/4096
   complete, both evaluators alive and waiting. All60 frozen training source
   hashes were checked unchanged during this diagnostic.
