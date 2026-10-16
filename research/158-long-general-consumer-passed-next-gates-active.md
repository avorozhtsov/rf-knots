# Long corpus independently passed;150 and152 launched

2026-09-24,09:18–09:20 UTC. Explicit user status request permitted a fresh
inspection. Previous goal turn made progress by completing structural checks
and launching the independent147 consumer, not by merely restating status.

The consumer session32772 is terminal with a passed report. Its process73655
is absent. Report mtime07:46:23 UTC; SHA256
`d0fc9852b22870e77c6de8d98452f2a905fd3b7f0efd6ad43f9c3c44dc5bc6f5`.
All4743 selected certificates were checked, all6144 constructions covered,
training_ready=true and support_gaps empty. Corpus SHA remains
`27d15e34873bddeeb7bd89db0f94929328b5b9441f595085421662bed044e6a1`.
The150 wrapper successfully loaded the ready source and rechecked its pins.

## New executions, not production training

Before launch, no matching worker or output directory existed for either job.

-150: session78213, wrapperPID24470, childPID24509. Output
 `artifacts/long-mixture-fresh-short-u-test-v1`. Exact frozen command uses
 `scripts/build_long_mixture_short_u_test.py --long-data
 artifacts/long-general-cc-supply-v1 --out
 artifacts/long-mixture-fresh-short-u-test-v1`.
 The child uses10000 attempts,seed20261078,max_strands33,depth24..96 and
 the frozen7916-group exclusion registry. Initial child observed100%CPU,
 elapsed39s,CPU37.33s. At09:20 its actual progress advanced0→100 attempts:
 accepted7,other_partition81,determinant_one5,previous_group7. This is not
 final64-root readiness. Progress snapshot SHA
 `00c138c35be14e1173ea9a6e76a81875415b46b074be1b558f1c62ea3a3615cb`.
-152: session89169,PID24478. Exact command uses
 `scripts/check_long_general_mps.py --long-data
 artifacts/long-general-cc-supply-v1 --pgx-root
 /Users/artemvorozhtsov/projects/pgx-mcts-bench --out
 artifacts/long-general-mps-execution-gate-v1`.
 Environment additionally sets PYTORCH_ENABLE_MPS_FALLBACK=0; both jobs use
 OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Gate process observed100%CPU,
 elapsed48s,CPU45.69s. At09:20 it was still in pre-output preparation; no
 gate directory/report existed. Do not claim an actual GPU forward, a passed
 numerical gate, or production optimizer progress from this process alone.
 Source/data/runner/protocol pins required by150/152 must now remain frozen.

No149 production training has launched. Finish150 and its separate --audit-only
consumer; finish152 and validate its report; then proceed with149 and155 under
their existing contracts. Do not duplicate current jobs or relax failing gates.

## Untouched139 and completed97

139 controller15420/session78472 and worker21909 are alive at09:18. Worker
99.5%CPU,elapsed6h02m05s,CPU360m28.65s. Current search advanced8→16 of128
committed conditions at06:49:26 UTC:96 episodes,48064 policy evaluations,
0 solutions. Controller still12/24 jobs; this is not a final comparison.
Next routine combined check no earlier than10:18 UTC, unless a terminal/error
event or explicit user request needs earlier action.

Completed97 controller/summary and all13 child artifact hashes were rechecked.
Summary SHA `83037de0cb75a9ade88173cbb1a65a93d9e8efd2a8dfd9835c5b84db9f5dd50e`.
The reported Q254-vs-random successes9/64 vs5/64 atL10 and11/64 vs9/64 atL1000
remain unchanged; no reference CC or route-cost improvements. These synthetic
results do not establish public-bound improvements or isolate embeddings.

No public database campaign/import/release, cloud rental, Git commit/push or
changes to live139 occurred. Preserve all dirty work and generated evidence.
