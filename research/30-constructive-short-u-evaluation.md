# Prospective constructive short-u/search cohort

2026-09-20, before this construction is sampled or scored. Older random-walk
panels remain insufficient and unchanged. The newer historical pool
`pgx-mcts-bench/artifacts/nebius-main32-final-20260817/artifacts/evidence-catalog-v1/best-solutions-pool.json`
(SHA115a9b08366da3f05d9ba8c6afe7bda855dfd5dd35d6561eff2406b89441b66c)
has47 replayable CC<=2 witnesses,12 test candidates, but no query groups novel
relative to the present panels. It is not a fresh test cohort.

New construction starts from the positive chain [1,...,n-1], whose closure
destabilizes to the unknot. Before selected chain letters insert blocks
`X g g X^-1`. Flip the second g in each block; then cancel the central pair
and successive inverse pairs, rightmost block first, and destabilize the chain.
Every step is independently replayed in the existing Artin action environment.
One/two blocks certify u<=1/2, not equality. Word complexity and the number of
strands can grow without a vocabulary change, but the witness structure is
synthetic and contains easy cancellations after the decisive CCs. Do not
claim this is an unbiased real-knot distribution or a hard-topology benchmark.

Fixed2000 attempts, seed20261010. At index i: n=3+i%6,
blocks=1+(i//6)%2 and prefix depth=4+i%13. Independent deterministic Python RNG
per index chooses block positions, signed generator prefixes and central g.
Only test-owner query groups, determinant!=1, absent from the five panels of
note28 and the two completed fresh/general panels of notes28/29 are eligible.
Keep first four distinct circular words per group and first32 per replayed CC
count. No model scores, retries with other seeds, or early stopping. Exhaust
the whole stream. Adequacy: at least eight groups,32 representations and eight
examples in each replayed CC class1/2. Otherwise do not score or relax the gate.

Retain signature lower bounds and replayed upper bounds as primary data.
Report original and determinant-strengthened lower bounds separately; det!=1
certifies u>=1, so one-CC witnesses here give exactu1. Two-CC witnesses with
signature0/2 usually remain [1,2], not exactu2. Generate zero-CC lifts to
L256,N33 for stability only, not independent trials.

Freeze models to note27's final4096-step joint/control and pinned initial.
No tuning on the general-pair results, even if they finish first. Distance
evaluation: original/lifted views, constants0/1/2, group and exact-class errors,
both bound conventions. Search every original query with128 anytime expansions
for both L10 and L1000, witness-feasible caps, layout tie-break, compare joint,
policy, initial and length. Sparse MCTS: one shortest reference witness per
group (CC,moves,length,index tie-break), four seeds,512 simulations, widening4,
constant leaf0.5, root noise0.25, both CC costs, joint/control/initial.
Replay every returned witness, preserve censored failures and report paired
costs, new/lost solves and actual compute. No independent-trial interpretation
for same-group representations or related synthetic constructions. Independent
training-path audit is required before any scoring. This remains a bounded
test, not million-node RL or proof of arbitrary-size solving.
