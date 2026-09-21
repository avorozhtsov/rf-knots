# Clean paired-cost rollout iteration on the multisize architecture

2026-09-20. Frozen before collecting these episodes or scoring the prospective
panels. Overall success still requires useful short CC distances AND better
replayed L10/L1000, not just finite execution at large dimensions.

## Rationale and parent

Notes25–26 contain several completed rollout iterations with little reliable
out-of-group improvement. They must not be presented as new evidence or silently
reused as clean training. Note27 subsequently created a from-scratch, audited
multisize lineage; notes29–32 found a general-pair signal, no robust short-u/search
gain, and feasible execution at L16384/N8193. This experiment asks whether actual
policy/value training improves that clean parent, not whether another tiny output
adapter helps. No new learned layers or output residuals are added.

Parent `embedding-multisize-clean-joint-v1/checkpoint.pt`, SHA
`076a12a86ca83c03d736edcaea09b31234cae329c9d8d5c743e8c8bef2fdc723`.
Metric corpus `embedding-multisize-clean-corpus-v1/corpus.json`, SHA
`04ffeca1bebccc5e8af6a7f7553b2faa2895e098041320e73bf4a2cd8093600e`.
Parent `u_views.json` SHA
`56a5f26b9e816a659b72928d24b1c9a164a0414517bcebafe11301bcf7cf54a0`.
No legacy checkpoint is an ancestor of this parent.

## Rollout construction

Use `collect_rollout_value.py`, seed20261011,256 base roots, eight episodes per
resource/cost condition. Select views with reference remaining moves1..128,
round-robin over `(det,abs(signature), capped 8-move band, 32-letter band,
8-strand band)`, shuffling within bins with that seed. Do not use predictions,
successful rollouts or held-out measurements to select roots. Add no extra
trivial lifts; the input corpus already contains the fixed size views.

The model-blind selection is256 roots in21 fingerprint groups:124 words below
or equal32 letters,68 in33..64,64 in97..128. These are correlated representations,
not256 new knot types. There are86 roots whose reference paths exceed64 moves.
Root-u bound counts:105[2,2],61[0,2],58[1,1],16[0,0],16[1,2].

Each selected root appears at BOTH CC weights10 and1000, with identical root,
move/CC/length/strand caps and reference witness. This is512 separately grouped
conditions and4096 complete episodes, not a million-node RL result. Different
cost contexts never share a winner/loser group. Caps fit the witness with eight
extra semantic moves; do not adapt them after seeing this experiment's outcomes.
The fixed policy samples from all ordinary actions plus16 state-hashed insertions.
This is the existing bounded rollout behaviour, not the whole-word top-k MCTS
provider from note32 and not a complete-action policy softmax.

Any trajectory entering a non-train invariant partition remains stored but is
ineligible for training. Recheck ownership at all CC endpoints; zero-CC moves
preserve the invariant. Record this selection bias. Only full trajectories can
receive a return; interrupted chunks resume from the last immutable shard.
Output: `artifacts/embedding-clean-paired-rollouts-v1/`.

## Targets, training and adequacy gate

Seeds0..5 provide training labels;6..7 are same-root diagnostic seeds, not a
knot-disjoint test. Require at least32 eligible complete six-episode groups with
nonzero relative-policy terms, including at least eight with original length
>=96. Otherwise retain data as insufficient and revisit the TRAIN curriculum;
do not manufacture negatives, change the gate, or select a replacement seed.

Value predicts success of ONE full rollout of the pinned parent behaviour within
its supplied resource horizon. It does not predict mathematical solvability,
arbitrary search budgets, MCTS success, or the updated policy's success.
Same-root/resource/cost winners share total+1; strictly worse complete outcomes
share total-1; all-failure/all-tied groups supply no preference. Cancel shared
state-action prefixes exactly before clipping; never normalize each trajectory
by its own length. Unknown alternate moves are not proof-graph negatives.

After the gate passes, run two fixed2048-step arms from the same parent, seed
20261013, batch64, lr2.5e-5, relative-policy weight0.1, clip0.2, metric weight1:

- `embedding-clean-paired-value-v1`: value BCE weight1.
- `embedding-clean-paired-control-v1`: value BCE weight0.

Both process identical batches/RNG streams, both train the full encoder and
policy, with no independent residual module. Same source, architecture and final
step; no early stopping or checkpoint selection on evaluation. Store exact
initial tensors in each arm, source snapshots, hashes and optimizer/RNG state.
The frozen parent is a further control. These arms test incremental value
supervision, not removal of the embedding channel or proof of its necessity.

## Prospective independent evaluation inputs

Reserve new model-blind panels before scoring the updated checkpoints. Old
notes29/30 panels are consumed development evidence and are explicitly excluded,
along with the five older panels from note28 and the failed fresh-group panel.
Exclusion is by either endpoint's mirror-folded fingerprint group. This remains
an audit relative to the declared lineage/history, not all legacy experiments.

General short-Gordian panel `embedding-clean-paired-gordian-test-v1`: fixed4000
attempts, seed20261014, positive seed strands2..7 and genus1..10, using the existing
general-pair walk and caps. Same first-four-per-(group,class) and32-per-class
selection. Adequacy unchanged:>=8 groups,>=32 rows and>=4 exact0/1/2 each.

Short-u panel `embedding-clean-paired-short-u-test-v1`: fixed4000 attempts,
seed20261015, conjugate-block constructor with strands3..12 and depth8..32,
one/two CC blocks, test-owner/nonunit-determinant roots, first four per group and
32 per CC-count. Adequacy unchanged:>=8 groups,>=32 rows,>=8 witnesses per CC
class1/2. These are synthetic short-u constructions, not arbitrary natural knots;
large complexity may still normalize away. Lifted views are stability checks.

Exhaust each declared attempt stream. An inadequate panel is NOT scored or
silently topped up. No retries with new seeds. Independently replay all metric
train paths, all evaluation witnesses and all eligible rollout CC endpoints;
check no query overlap and independent determinants before scoring. The two
new panels may overlap each other; they are separate tasks, not pooled
independent trials. No panel enters training or root selection.

## Predeclared evaluation after both training arms finish

Report parent/value/control interval errors, exact-class errors and constant
0/1/2 controls on both views. For short-u report original and explicit
determinant-strengthened bounds, not whichever convention looks favourable.
Use all original short-u queries for128-expansion anytime embedding cost search
at both costs, same witness-feasible caps and layout tie-break; compare all three
models and length. Replay new/lost/common solves and costs, report real compute.

Sparse MCTS: one shortest reference witness per query group, two seeds,512
simulations, widening4, root noise0.25, same witness caps, both CC costs. Parent
uses constant0.5. Each trained arm is tested with learned and constant0.5 leaf
values, identical default catalogue proposals (note32 gave no reason to promote
site proposals). First-found costs remain nonoptimal; no-solve costs censored.
Also report the same-budget stochastic parent-rollout portfolio control, charging
all neural work and not pooling seeds into more compute than the MCTS baseline.
This final test is not used to tune this experiment; a failure remains a result.

Million-node RL and genuinely nontrivial long-word transfer remain future
requirements. This iteration changes the clean parent using real complete
behaviour, both cost contexts and long training roots; it is not a claim that
the overarching goal is already achieved.
