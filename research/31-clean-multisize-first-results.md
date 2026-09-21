# Clean multisize training: first independent distance results

2026-09-20. Overall goal remains active: useful short Gordian/u distances AND
improved returned L10/L1000 witnesses with a size-independent architecture.
The results below establish a bounded distance signal, not search promotion.

## Training and provenance

Both note27 arms completed4096 steps from the same initial tensors on the same
frozen4096-record corpus (lengths6..128, strands2..17). Each processed262144
sampled metric presentations and32768 sampled policy states; these are not
distinct knots or self-play games. No held-out checkpoint selection was used.
Final model/objective tensors are finite. Metric, anchor, policy and torch RNG
states match between arms at the final step. Initial checkpoint SHA:
`e0e0b4acf6518c64e60c2bbd8a57a9040bda13662ff5bd431f52f5e60d12737b`.

- Joint checkpoint: `076a12a86ca83c03d736edcaea09b31234cae329c9d8d5c743e8c8bef2fdc723`.
- Policy-only checkpoint: `b37e6d006a8fbb0cea49846170c59ccee59e0b963138323005a474d7260d841c`.

All files are under `artifacts/embedding-multisize-clean-{joint,policy}-v1/`.
The policy uses a conditional candidate shortlist, not the full-action softmax;
value outcomes were not trained, so search evaluation keeps leaf value0.5.

## Prospective general-pair test (note29)

84 pairs,18 fingerprint groups:29 exact0,20 exact1,9 exact2,26 intervals.
References may be known, but no query fingerprint appears on any of the4096
training trajectories (123 intermediate-path groups) or the five excluded
previous panels. All original/lifted records replay; the independent determinant
implementation agrees for all18 query groups. Audit completed before scoring.
This is relative to the audited lineage/panels, not an exhaustive legacy
experiment exposure audit. Fingerprint groups are not unique knot identities.

Error is distance outside the certified interval; for exact labels it is
absolute error. Lower is better. The group column gives every fingerprint
group equal weight, rather than every representation.

| Model | Mean interval error | Equal-group mean | Exact0 MAE | Exact1 MAE | Exact2 MAE |
|---|---:|---:|---:|---:|---:|
| Joint | 0.332823 | 0.328625 | 0.215296 | 0.549025 | 0.546557 |
| Policy-only | 0.538357 | 0.712889 | 0.002109 | 0.950034 | 1.965957 |
| Same initial weights | 0.486369 | 0.653420 | 0.029430 | 0.808713 | 1.808258 |
| Constant1 | 0.452381 | 0.508140 | 1 | 0 | 1 |

Other constant row errors: constant0=0.559524, constant2=1.011905.
Joint is better/equal/worse in11/2/5 groups versus policy-only,10/2/6 versus
initial. Thus improvement is not uniform; exact-zero invariance got worse.
The positive aggregate signal does not establish population-level statistical
superiority on a small, structurally related synthetic cohort.

Lifted left views (L160,N25) have identical predictions. These lifts are removed
by deterministic global-view normalization, so this is consistency of that
preprocessing contract, NOT evidence of learned long-range reasoning or hard
large-knot generalization. The architecture accepting large inputs remains
distinct from proving useful decisions at those sizes.

Reports under `artifacts/embedding-multisize-general-evaluation-v1/`:

- Joint JSON SHA `74a4a1f38c7d60e15f92ce1b40a53cb093ccb58f82c477aef7bc3b091a34fd27`.
- Policy JSON SHA `923c8980347eb668e76dd6b22f725860cab29ec4d9efc4369d63f1dbe25c3ba4`.
- Initial JSON SHA `bfaa39f55a4ba37b10025a8152b5f8b2d27b7668eeae4200f64aac77b83ed17a`.

Runner completed normally; report and checkpoint hashes were checked again.

## Short-u and actual search: still pending

The old note28 generation finished insufficient (four rows/three groups), and
was not scored or silently supplemented. Note30 declares a separate, model-blind
constructive cohort before this metric scoring: all2000 attempts completed,
40 accepted roots in33 new groups, eight with a one-CC witness and32 with a
two-CC witness. The adequacy gate passed without changing its thresholds.

Panel SHA `d00d6b6fb72a2918fe3f5be761984be302f49eb4b78aa0909905dd06e2b5abd9`,
at `artifacts/embedding-constructive-short-u-test-v1/panel.json`.
Its independent full-train-path audit is currently running (historical PID54822,
exec82908), not yet claimed passed. Do not score it before the audit completes.
One-CC/nonunit-determinant roots have exactu1 after the explicitly declared
determinant bound; two-CC roots need not have u=2. The synthetic inverse-pair
structure is a substantial limitation, not a representative natural population.

Next authorized work is the frozen note30 comparison: both bound conventions,
all40 roots in fixed128-expansion anytime search at L10/L1000, and the declared
four-seed512-simulation sparse MCTS on one model-blind representative per group.
No model/architecture tuning from the first distance result before this test.
Replay returned witnesses, compare common/new/lost solves and charge real
neural work. Until these results exist, no smaller-L10/L1000 claim is justified
and the full goal is not complete.

## Short-u audit and distance evaluation completed

The subsequent full audit passed:4096 train records,123 train-path groups,
80 test-view records,33 query groups, no query overlap with train paths or
the seven excluded panels. Independent determinants agree. Audit SHA:
`1b9274457c0c093bc1137f44f6d5798971f5a55e8ebf970b9a491d2c44fdf946`.
This supersedes the in-progress snapshot above. Both constructors and all
24 deterministic replay tests passed before sampling this cohort.

Six distance reports completed under
`artifacts/embedding-constructive-short-u-evaluation-v1/`, with names
`{joint,policy,initial}-{original,determinant}.json`. No retraining intervened.

| Model | Original-bound error | Strengthened-bound error | Strengthened group mean | Exactu1 MAE (8) | Exactu2 MAE (6) |
|---|---:|---:|---:|---:|---:|
| Joint | 0.171309 | 0.171309 | 0.176303 | 0.601084 | 0.340618 |
| Policy-only | 0.563993 | 0.613024 | 0.592203 | 1.214006 | 0.986153 |
| Initial | 0.161473 | 0.171882 | 0.200571 | 0.042085 | 0.974421 |
| Constant1 | 0.15 | 0.15 | see per-group report | 0 | 1 |
| Constant2 | 0.20 | 0.20 | see per-group report | 1 | 0 |

Thus the general-pair gain does NOT establish a useful short-u estimator.
Joint is essentially tied with initial on strengthened aggregate error,
slightly worse on original bounds and worse than constant1 by row average.
It improves exactu2 but substantially harms exactu1 versus initial. Broad
intervals and unequal class counts make an aggregate-only interpretation unsafe.
Do not choose a favourable bound convention to declare success.

## Actual search underway (not yet a matched result)

All40 original roots, both CC costs,128 anytime expansions, witness-feasible
caps and the declared layout tie-break. Length control completed80 rows:
5/40 solves at each cost. Learned joint, initial and policy controls are live;
partial solves must not be compared to the full length denominator.
Output root: `artifacts/embedding-constructive-search-v1/`.
Historical launch sessions: joint74861, initial5063, policy25742; length70593
has exited successfully. Check live commands and fresh report rows on resume.

Sparse MCTS joint-L10 has also launched at
`artifacts/embedding-constructive-mcts-v1/joint-l10/report.json`, exec54922,
33 model-blind group representatives,4 seeds,512 simulations, widening4 and
constant leaf0.5. The other five predeclared model/cost MCTS comparisons still
need execution. Preserve source files until active evaluators finish because
their reports pin code hashes. Search promotion and the full goal remain open.

## Anytime cost search completed: no learned gain over initial

All four80-row reports completed. Independent replay-audited paired comparisons
are `embedding-constructive-search-v1/joint-vs-{length,initial,policy}.json`.
Each model solved the same count at both costs: joint18/40, initial18/40,
policy17/40, length5/40. This alone does not show which cases improved.

For EACH of L10/L1000, comparing joint against:

- Initial: six new solves, six lost, twelve common solves with identical costs,
  sixteen both unsolved. No net solve gain or common-solution cost reduction.
- Policy-only: ten new, nine lost, eight common with equal costs, thirteen both
  unsolved. A one-root net difference is not robust promotion evidence.
- Length: fourteen new, one lost, four common (three cheaper, one equal),
  twenty-one both unsolved. Mean cost change on the four common roots is -4.75.

Joint spent11201 neural state evaluations per cost; initial8572/8574 and
policy8653/8699. Wall times were about182/189s joint,152/139s initial,
129/125s policy,26/26s length, with concurrent local processes. These are
recorded observations, not isolated timing benchmarks. Equal expansions did
not mean equal neural work. Initial architecture plus deterministic normalization
already improves strongly over length; the trained metric has not improved
that outcome on this cohort. This does not prove embeddings fundamentally
unhelpful, but it does prevent a learned-search improvement claim here.

## MCTS comparison continuation and scope caveats

Joint-L10 finished512 simulations per tree with0/132 solves,33 groups x4 seeds.
Its report SHA is `eb43fa4e67b869a382dcb0b4f0180fff6d27e6306423d79cf3abb8db8a95c3ec`.
Coverage/censoring and root/resource consistency were independently validated.
The other five declared model/cost jobs have now been launched; compare only
their complete immutable reports. Sessions: policy-L10=98772,
initial-L10=85267, joint-L1000=60802, policy-L1000=82681; initial-L1000 was
launched in the same continuation and its live report records the PID.

`compare_matched_mcts.py` now enforces identical config/selection/source,
complete query-seed coverage, replayed costs and caps, and censored failures.
Eight tests pass, including corruption, missing/duplicate seeds and mismatched
budgets. This comparison tool is not a dependency of the live search processes.

Inspection confirms MCTS `_record_solution` stops at the first witness and
backs up terminal value1. CC cost only conditions policy inputs; it does not
turn this MCTS into an anytime cost optimizer. Witness policy samples use
default CC weight1000 throughout training, so weight10 is also an out-of-
training input condition for trained heads. Keep both declared tests but do
not call them evidence of learning two cost objectives.

A read-only catalogue diagnostic over the33 selected reference witnesses found
all first moves inside the nominal92-action prefix (4*ceil(sqrt(513))), but22
witnesses have a later action beyond that prefix, maximum zero-based rank186.
This is a potential proposal-coverage bottleneck, NOT proof those solutions
are unreachable: force-widening can exceed the nominal schedule, dominance
changes offered sets, and other witness orders/routes exist. Do not label
unoffered actions bad or conclude states are unsolvable. No active search code
or budget was changed in response to the diagnostic.

## All six fixed MCTS evaluations completed

All six jobs finished normally at the declared512 simulations. Every arm had
0/132 solves (33 groups,4 correlated seeds), at both L10 and L1000. All four
joint-versus-initial/policy comparisons passed replay/resource/coverage checks;
their cost differences are null because there are no common solved cases,
not zero. These are censored failures at this budget, not unsolvability proofs.

| Arm | CC cost | Neural state requests | Recorded seconds |
|---|---:|---:|---:|
| Joint | 10 | 67547 | 362.94 |
| Policy | 10 | 67559 | 484.37 |
| Initial | 10 | 67453 | 465.01 |
| Joint | 1000 | 67518 | 508.83 |
| Policy | 1000 | 67512 | 505.79 |
| Initial | 1000 | 67472 | 427.07 |

Timing is heavily confounded by concurrent load: a live OS snapshot showed
about21GB swap in use and load average19. No unrelated process was stopped.
Do not rank model efficiency from these wall times. All six processes have
exited; their immutable reports and source snapshots remain intact.

Additional final-report SHAs (joint-L10 is listed above):

- Initial-L10 `9ef8e0ab53785f63611563e0101157820070822b19d5f7bb31aae80f1e57234a`.
- Initial-L1000 `25e52894f35ec2c6d4f364678aee67ef9c237e16d36ea718e36290f588420dc3`.
- Joint-L1000 `e7c0bfacf98a27c498de08bb01d9808c77583ae52478f0b7fedddb6577b78f90`.
- Policy-L10 `f71f4d6e23326a9c0fe6c5e044bbccfd86afd4cab7a0c0a6c6fd9aea8b6433ce`.
- Policy-L1000 `0c33f294ba042323765d999ca9e6f005eef1e4e00a96a63fd73da2d4b90f0a9d`.

The catalogue diagnostic is now reproducible with
`scripts/audit_witness_proposal_coverage.py`; output is
`embedding-constructive-mcts-v1/reference-route-catalogue-audit.json`.
Six inverse-index tests cover all actions in small catalogues with/without
growth and CC; together with eight paired-comparison tests,14 tests passed.
It remains a reference-route diagnostic, not an observation that actual search
could never propose those actions.

Conclusion: the completed experiment shows a general short-distance signal,
but does not show learned improvement in short-u estimation or returned
L10/L1000 over the matched initial architecture. Do not promote the model,
discard embeddings as a concept, or claim the full objective is complete.

The next bounded engineering hypothesis is whole-word model-scored ordinary
action proposals, with sparse insertion proposals and exact catalogue fallback.
That targets a structural size-dependent search bottleneck without a residual
value add-on or new model parameters. Count proposal inference explicitly;
preserve legal-action checks, witness replay and eventual fallback coverage.
First validate on training/development data and engineering tests, then freeze
an independent follow-up protocol. The consumed panels above are now
development evidence and must not be rebranded as untouched tests after tuning.
