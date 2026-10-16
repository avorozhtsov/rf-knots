# A million metric draws is not a million representations or games

2026-09-21. Protocol129 completed a TRAIN-only sampling comparison. No neural
forward, optimizer step, TEST scoring, MCTS game or active-sampler change occurred.
It tests the data-exposure implications of scaling the current static graph pool,
not whether a sampling distribution improves embeddings or search.

## Evidence

Root: `artifacts/graph-million-draw-coverage-v1/`.
Report SHA256:
`83596e30c72d719921faf1c13f9b958c158a2ac03cfc204b2dcd8e176a611aaf`.
Inputs SHA256:
`28494f7e41e619a08adf1a9ff22f9d673f470d5ba271c8edbb196c0e772c9afa`.
Saved trace SHA256:
`cd8a54dd60b95120719676998b0ab620c14f2b6b82bca21199a5daf6767bf5ba`.

All1158 pinned inputs/sources rechecked. The runner exited0; five targeted tests
and Ruff passed. Three million selected record indices and one million shared
category/group/orientation contexts are stored in `traces.npz` (about8.7MiB).
Every method's million-draw trace was regenerated from the frozen seed and
matched the saved arrays. Method1 was additionally checked row-by-row against
one million calls' worth of the ORIGINAL graph sampler, including swapped raw
endpoints, pair specs and witness-priority metadata.

A separate NumPy audit checked all12 milestone record-frequency summaries,
class/group counts and group-distribution distances. It also verified682865
complete shuffled cycles each contains its entire eligible pool exactly once.
The original graph corpus's replay/partition audit was verified by the unchanged
loader. No new distance labels or topological-equivalence claims were invented.

## Controlled comparison

All methods use the same fixed graph TRAIN support:50615 pairs,58237 distinct
raw `(N, word)` endpoints. Class choices and endpoint orientations are identical;
the first two methods also have identical group choices. Seed20261063 defines a
new graph-only stream, NOT the original97 mixed-source RNG stream. Therefore
4387 pairs at32768 draws below need not equal121's actual-training4516.

The three methods are:

1. Current equal-class/equal-group sampling, rows drawn with replacement.
2. Same class/group weighting, shuffled row cycles within each group.
3. Same class weighting, shuffled cycles over ALL records within each class.
   This weights groups by their record populations instead of equally.

| Graph draws | Current: distinct pairs | Group cycles: distinct pairs | Class cycles: distinct pairs |
| ---: | ---: | ---: | ---: |
| 32,768 | 4,387 | 4,834 | 23,063 |
| 65,536 | 5,807 | 6,539 | 44,859 |
| 262,144 | 9,748 | 10,935 | 50,615 |
| 1,000,000 | 15,768 | 17,394 | 50,615 |

At one million draws:

| Measure | Current | Group cycles | Class cycles |
| --- | ---: | ---: | ---: |
| Distinct raw endpoints | 23,682 | 25,689 | 58,237 |
| Unvisited available pairs | 34,847 | 33,221 | 0 |
| Maximum repetitions of one record | 3,055 | 3,055 | 275 |

All methods have identical class totals333530/333184/333286 and visit all1434
class-specific endpoint-group buckets by one million draws. These buckets are
not1434 proven knot identities; the same fingerprint pair may occur in more
than one distance class. Raw representations are likewise not distinct knots.

Method2's modest gain is informative: removing within-group repetition does not
solve underexposure of large groups when every group receives similar draw mass.
Method3 covers all available pairs by the262144 checkpoint, but is not a free
coverage improvement at unchanged sampling weights.

## The weighting tradeoff is large

For each class, total variation distance between empirical group frequencies
and an equal-group distribution at one million draws is:

- Current and group cycles: .0201, .00672, .0152.
- Class cycles: .8490, .7093, .9103.

Thus method3 makes a substantial distribution change, not just an implementation
optimization. At32768 draws it reaches only913 of1434 class/group buckets, versus
all1434 for both balanced methods, despite exposing many more records. Minority
group exposure and unique-record coverage are different objectives.

## Consequence for the next scientific experiment

Increasing static draw count alone does not realize the intended million-
representation RL program. Even perfect traversal of this pool is bounded by
58237 existing raw endpoints and does not generate a single additional self-play
game. A million games/nodes must come from actual search acquisition with
independently replayed trajectories; a million representation views needs a
separate certified generation stream with measured duplicates and lineage.

A useful controlled NEXT comparison is current group balancing versus class
cycles at matched draws, initialization, losses and independently frozen DEV/TEST
contracts. Report equal-group quality as well as record-weighted quality so large
groups cannot conceal regressions. Keep source mixture and witness-priority
weights fixed for that comparison; otherwise changes cannot be attributed to
the sampling intervention. The paired97 MCTS result remains pending and should
inform the decision to spend a larger training budget.

Certified zero-CC views from123 can subsequently expand representation support,
but do not automatically improve knot-identity diversity or policy supervision.
Apply their invariance labels to the knot embedding; do not transplant old local
action positions or old L10/L1000 move costs onto the altered word. Preserve the
canonical empty anchor where the specific task requires it, and use separate
general zero-CC view pairs rather than silently redefining the anchor.

No learned conclusion, checkpoint promotion, data release or architecture change
follows from this sampling experiment. Together with121 and127, it supplies a
concrete implementation option and an explicit tradeoff for scaling, not evidence
that embeddings have already become useful or should be discarded.
