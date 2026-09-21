# Whole-word site proposals: development engineering check

2026-09-20. Declared before scoring this change. This is NOT a new held-out
promotion experiment or a new neural architecture. Note31's independent tests
are finished and now consumed evidence. No model weights are changed here.

## Why this step is necessary for the large-word architecture

The size-independent graph/window encoder exposes a policy at every crossing,
but catalogue-prefix widening can delay actions far along a word. The earlier
route-index audit only suggested this risk: it did not prove alternative
solutions impossible, or prove the cause of the zero-solve MCTS results.

`SiteProposalProvider` uses the existing frozen site head to score every legal
non-insertion action over the whole word, plus at most k arithmetic samples of
insertions. It ranks these and offers up to k candidates. Only selected children
are materialised by the tree; the exact lazy catalogue remains the exhaustive
fallback. There is no extra learned adapter and no new value target.

Candidate storage is O(L+k), instead of enumerating O(L*N) insertions. Sorting
the candidate scores is O((L+k) log(L+k)); the encoder has its own compute cost.
Each materialised successor still copies O(L) symbols. The insertion sample is
not a guarantee of finding the best insertion; full catalogue fallback can be
very expensive. Unopened moves and failed searches are not negative labels.

Proposal requests are batched per search round. Identical requests across seeds
share one inference; the cache is discarded next round. The key preserves the
literal layout and all remaining-resource/cost context, not only knot identity.
Proposal state evaluations and candidate scores are charged separately; the
ordinary policy evaluation currently recomputes the encoder. Equal simulations
therefore do not mean equal compute. Wall times on this loaded host are not an
isolated speed benchmark.

## Frozen development comparison

- Checkpoint: `embedding-multisize-clean-joint-v1`, SHA
  `076a12a86ca83c03d736edcaea09b31234cae329c9d8d5c743e8c8bef2fdc723`.
- Development corpus: `embedding-multisize-clean-corpus-v1/corpus.json`, SHA
  `04ffeca1bebccc5e8af6a7f7553b2faa2895e098041320e73bf4a2cd8093600e`.
- Both existing views, `validation` and `size_shift`. For each view choose the
  first six distinct fingerprint groups under the evaluator's model-blind
  `(witness CC, witness moves, word length, row index)` order. Views may choose
  different representations; only comparisons within each view are paired.
- Same checkpoint, two seeds (0,1), 256 simulations, widening batch4,
  witness-feasible resource caps, root noise0.25, constant leaf value0.5.
- Both CC policy contexts10 and1000; the policy was trained at1000, so10 is
  out-of-training context. Neither setting changes terminal reward or makes
  first-found MCTS an anytime path-cost optimizer.
- Catalogue versus site-ranked proposals: eight runs total. Default catalogue
  control is rerun with the SAME current source, including duplicate-action
  prevention; old-source reports are not used as the matched baseline.
- Outputs: `artifacts/embedding-site-proposal-dev-v1/{view}-{mode}-l{cost}/`.
  Replay paired witnesses and report new/lost/common solves, censored costs,
  proposal overhead and total neural evaluations. No selection of a best seed,
  view or cost after scoring; this is development evidence only.

## Independent engineering gates (not scientific holdout tests)

Unit checks cover exhaustive small catalogues, duplicate prevention, legality,
far-position ranking, resource-conditioned batching, fixed weights, replay and
candidate generation for L4096/N4097 without enumerating 30M+ insertions.
The same pinned checkpoint will additionally run eight actual sparse MCTS
simulations with at most two moves at (L,N)=(64,33),(4096,2049),(16384,8193),
uniform control and trained/site-ranked policy. This deliberately is NOT a
claim to solve those large positive torus knots. It checks real successor
expansion, bounded candidate materialisation and unchanged model shapes.
Output: `artifacts/embedding-site-proposal-dev-v1/large-search.json`.

## Subsequent decision

Do not launch a large training job merely because this interface works. First
finish and audit all development reports. A useful-search hypothesis needs a
separately declared untouched cohort and matched training/objective controls.
Long words created by trivial cancellable lifts are insufficient: future
training and evaluation must include long-range structure that normalization
does not immediately erase. The overall architecture/training goal stays open.

## Completed results

All eight development runs completed normally; all four paired comparisons
passed the same-source/same-checkpoint, coverage, resource and censoring gates.
The final checkpoint hash, corpus hash and every evaluation source snapshot
were checked again after execution. The ordinary catalogue control also contains
the duplicate-action fix: a site proposal cannot become a second copy of the
same tree edge when the exhaustive cursor eventually reaches it.

Each arm solved 0/12 trees (six groups, two correlated seeds). Consequently
there is no paired path-cost difference to estimate; it is null, not zero.
Do not pool repeated costs/views/seeds into independent knot-level trials.

| Development view | CC context | Catalogue neural states | Site-proposal neural states | Catalogue action scores | Site total action scores |
|---|---:|---:|---:|---:|---:|
| validation | 10 | 3066 | 5777 | 25573 | 161695 |
| validation | 1000 | 3072 | 5762 | 25194 | 150971 |
| size_shift | 10 | 3062 | 5752 | 26595 | 270841 |
| size_shift | 1000 | 3062 | 5814 | 28164 | 282447 |

The new path uses roughly1.88–1.90 times the neural state evaluations here and
many more scores; no search improvement was demonstrated. Keep the feature
opt-in. Merely exposing all ordinary positions is not a sufficient remedy for
the observed small-budget search failures. This does not establish that global
embeddings or the architecture are useless after meaningful joint training.

The large-word engineering run also completed, with unchanged pinned weights:

| Word length | Strands | Trained/site time, seconds | Total neural states | Retained successor states | Process peak MiB |
|---|---:|---:|---:|---:|---:|
| 64 | 33 | 0.190 | 14 | 26 | 262.7 |
| 4096 | 2049 | 5.080 | 14 | 25 | 298.2 |
| 16384 | 8193 | 17.297 | 14 | 25 | 409.3 |

These are eight simulations with a two-move horizon, not solved large knots.
At the largest size the implicit root catalogue contains268451844 actions;
only14 root edges were opened, and seven proposal evaluations scored114756
candidate actions in total across visited states. No Cartesian insertion
array was constructed. Uniform controls also completed. RSS is cumulative
process high-water memory, including model loading; timings are not an isolated
hardware comparison. Large-search report SHA:
`f3492ed3944b169b7ba0f7f7e45fbe4a01abde4e8e293316489dbe1d43372b3b`.
Its measured source revision is archived in `large-source-snapshot.tar`.

All evaluator processes exited successfully. No cloud GPU was started by this
step, and no unrelated process or artifact was removed. This statement is not
a fresh inspection of the historical Nebius VM.

## Next training question

The required next evidence is not another tiny output residual. It is whether
joint training of this size-independent encoder and policy/value on sufficiently
varied complete trajectories can improve search under matched compute. Separate
short-distance supervision from path-cost preferences; train both CC-cost
contexts rather than merely testing an unseen10 input. A rollout value must
name the behaviour policy and resource horizon it predicts, not pretend to be
an oracle of solvability. Revisit the actual rollout controls/results in notes25–26
before changing that target or collecting a larger batch.

Use same-root/same-resource competing trajectories, cancel shared-prefix signed
weights before any nonlinear loss, and do not independently normalize route
lengths. Keep alternative successful continuations as a set rather than force
the proof graph's one ordering. Study nontrivial large-word families, not only
cancellable lifts; measure independent distance error and replayed L10/L1000
separately. Neither this zero-solve pilot nor finite large-word execution replaces
that prospective training-and-evaluation experiment.
