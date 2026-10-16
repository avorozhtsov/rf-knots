# Protocol97: complete paired evaluation and verified final summary

2026-09-23. All13 child jobs and the controller's final combined verification
are complete. Summary committed22:26:49 UTC; controller committedcompleted
22:26:50 UTC and its original execution session60520 exited0. The subsequent
process scan found no remaining controller/search/audit process. This is a
completed experiment, not a checkpoint promotion or completion of the broader
embedding research goal.

## Evidence and scope

Both trained arms completed2048 matched updates. Q254 supplies a pretrained
local raster trunk; the other arm starts that trunk randomly. Both have the
same new upper architecture, training data sequence and budget, and both use
global embeddings. The original Q254 policy/value heads are not used. This is
not an embedding-versus-no-embedding test or a before/after-training MCTS test.

Each final model searched all64 frozen synthetic TEST roots at both L10 and
L1000, with the unchanged resources, seeds and eight simulations. Roots include
long representations, but are not a campaign over named public-catalogue knots.
Shared70/74 TEST panels and incompletely audited historical Q pretraining limit
independence claims. No TEST trajectory is authorized as new training data.

Root directory: `artifacts/pretrained-raster-log-evaluation-v1/`.

| Evidence | SHA256 |
| --- | --- |
| Common128-root contract | `c1da996e1e6e95251986dd04bdad1467d45ad3f07055fc2b6ca79424131d7967` |
| Q254 final checkpoint | `c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f` |
| Random final checkpoint | `17433748fac7308cf4c36ef0c40650c7ec83d58dee8625834ce531c2f5bb40c6` |
| Q254 search manifest | `a88c3f09f54bdee789670df0cdd37eed3d2c45b139955b45b87d3803227d88a6` |
| Q254 full replay audit | `9ccc083ea71f2ae7378d03beea5f4471b8545ac16222ebd41b0ad84416e0291f` |
| Random search manifest | `1018ea18b77338fdc076aedc947c77709aa23ad08de6f834f8291e885b4d0ab7` |
| Random full replay audit | `5afbf9c6299bc8b8303f9bbf82f76910638c30cfd5d36d78917a0d14f0a756fe` |
| Final combined summary | `83037de0cb75a9ade88173cbb1a65a93d9e8efd2a8dfd9835c5b84db9f5dd50e` |
| Final controller manifest | `e1f5888bc225f9d69f8ffec434267088c44873350e6641c7c50a3a714b58aae5` |

The random search finished21:37:28 UTC, its audit21:54:44 UTC. Each full audit
replayed all successful and censored trajectories, all64 reference witnesses
and every found solution, checked semantic cost identities/resources and integer
determinants. The random audit covers128 conditions/759 episodes/14 solutions.

A separate read-only arithmetic verification checked all13 committed job hashes
and1345 combined input pins twice, identical root files, all committed shards,
solution coverage, certificate costs/floors, counts and committed-time sums.
It independently recomputed the paired success table below from raw records.
It did not run another semantic replay while the controller's summary was active.

After controller exit, a final independent consumer verified the summary SHA,
controller contract, all13 final children/checkpoint identities and1345 combined
input pins twice. It recomputed all initial/final distance aggregates, constant
controls, prediction identity/ranking checks and paired metric deltas, plus
paired search results and descriptive behavior counts. An additional direct
grouped arithmetic pass rechecked interval violations and constant1 errors.
Both full cost-audit payloads match the final combined certificates exactly.
No new neural inference, optimizer step or duplicate semantic replay was run.

## Final search counts

| Measure | Q254 L10 | Random L10 | Q254 L1000 | Random L1000 |
| --- | ---: | ---: | ---: | ---: |
| Queries |64|64|64|64|
| Solved |9|5|11|9|
| Budget-censored |55|59|53|55|
| Found cost certified optimal |2|0|2|3|
| Better than reference cost |0|0|0|0|
| Equal to reference cost |2|0|2|3|

There are20 Q254 successful conditions on16 distinct raw roots, and14 random
successful conditions on13 roots. These are not counts of distinct knot types.
Every found CC bound equals its reference CC bound; neither arm improves one.

| Paired outcome | L10 | L1000 |
| --- | ---: | ---: |
| Both succeed |1|0|
| Q254 only |8|11|
| Random only |4|9|
| Neither succeeds |51|44|

The sole common success is root54 atL10, with identical cost103. There is no
common-success L1000 cost comparison. Therefore mean route-cost comparisons
over each model's different successes would be misleading. Budget failures are
censored observations, not infinite-cost witnesses or bad-action labels.

The post-hoc union covers33 of128 conditions and28 of64 raw roots, but requires
both already executed searches. It is not a measured same-budget ensemble or
evidence for selecting an oracle best policy at each query. Complementary
successes motivate a future controlled test, not a promotion here.

## Last random solution and optimality

Final shard `conditions-00120.json`,1300028 bytes, committed21:37:25 UTC,
SHA256 `9324b032f57123e7387337cff450bcd57c08cb1ef8491e84cb5d71292d9eb953`.
It adds48 episodes,17821 continuation moves,17861 full-path moves,1502578 action
scores and one solution. Recorded batch time2889.314294 seconds.

Condition123/root61 atL1000 finds1CC/85 moves, cost1085, matching the optimal
reference. Its164-letter/5-strand start hasq0=84, determinant717875 and signature-2,
hence lowerCC1 and semantic cost floor84+1001=1085. The found path has one CC,
80 REDUCE and four DESTABILIZE actions, zero excess. Random's three proven cost
optima are roots51/59/61 atL1000. Optimality is for these exact representations
and the native semantic cost contract, not all presentations of their knots.

## Compute and distance context

| Recorded work | Q254 | Random |
| --- | ---: | ---: |
| Episodes |746|759|
| Policy-state evaluations |351448|369159|
| Policy neural batches |181676|187165|
| Action scores |55953078|63553982|
| Tree-embedding evaluations |3072|3006|
| Committed batch seconds |85743.710965|96441.999324|

These are work counters, not unique representations or new training examples.
The searches performed evaluation, not online learning. Timing is descriptive
under the recorded local execution conditions, not a controlled hardware-speed
comparison. The Q254 time above comes directly from its final manifest; an
earlier narrative rounded/reported the trailing microseconds differently.

Distance results remain those in109: both final models are worse than constant1
on all three aggregate interval-error measures, short-u calibration worsens
versus initialization, and general positive distances remain underestimated.
Q254's extra search successes do not reverse those distance findings or isolate
the contribution of the embedding. All successes remain existing-bound matches,
not a justification for importing a new public improvement or releasing UnknotDB.

## Next decision, not a launched experiment

The existing controller has finished; no restart or second summary process is
needed. Keep both final models and all data; neither is scientifically promoted
here. The finished evaluation's hourly heartbeat was setPAUSED according to its
existing completion instruction; the persisted status, unchanged prompt,
schedule and target were checked. The active broader research goal remains.

For a subsequent larger experiment, the TRAIN-only findings in119 and130 give
specific hypotheses about resource-input conditioning, task interference and
distinct-pair coverage. They do not prove which intervention works. Any such
comparison needs a new frozen contract and development/evaluation separation;
do not tune using the already examined TEST70/74 results. Preserve full-group
prefix cancellation, multiple valid positive actions and budget-censoring
semantics. More evaluation steps alone are not more training, and a longer run
is not a guarantee of learning useful knot invariants.

The broader embedding goal remains open: useful calibrated short distances,
causal or suitably controlled search benefit, reliable length extrapolation
and million-scale online learning are not established by this experiment.
