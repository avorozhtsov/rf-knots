# 17 — A scalable braid raster and shared cylinder network

## Decision

The two-dimensional representation is worth testing.  Its strongest feature is
not “images look natural to convolutional networks”; it is that one local Artin
generator becomes one locally checkable pair of cells, while the same shared
block can run at any word length and any strand count.

The first prototype used a **cylinder**:

* closing the braid makes the word/column coordinate cyclic;
* the first and last strand positions are not adjacent generators in the
  ordinary braid group `B_k`.

Naively wrapping both coordinates and imposing only affine Artin relations would
silently change the group.  The corrected main experiment now uses a torus with
a different semantics: the seam is the Birman--Ko--Lee band generator
`a_(1,k)`, which compiles exactly into the ordinary braid group `B_k`.  Thus the
network receives the desired cyclic strand geometry without changing the ambient
unknotting problem.  The cylinder remains the old-action ablation.

## Verified toroidal action alphabet `B*`

`B_k` is the ordinary braid group, not an action set.  `B*` is the experiment's
name for a redundant cyclic generator alphabet inside that same group.  For a
state with `k` active strands, signed letter `+/-k` means the seam band
`a_(1,k)^+/-1`.  It compiles with

```text
w = sigma_(k-1) ... sigma_2
a_(1,k)^+/-1 = w sigma_1^+/-1 w^-1.
```

The compiler feeds the historical faithful Artin-representation verifier.  Free
reduction/insertion, cyclic-neighbour braid relations, remote commutation, and
crossing change are implemented directly on the compact `B*` word and checked
after compilation.  The duplicate seam action is masked at `k=2`, where it is
exactly `sigma_1`.

A seam letter's code depends on the current strand count.  Therefore Markov
stabilization and destabilization are illegal while a seam letter remains; the
agent must first remove or rewrite it.  This conservative rule prevents a stored
`a_(1,k)` from silently becoming an ordinary `sigma_k` after growth.  A future
endpoint-pair representation can remove this restriction.

For comparison, report both intrinsic band cost and compiled Artin cost.  The
current early ladder gates compare solve rate and crossing changes; mature
objective comparisons must add compiled move cost before claiming that `B*`
shortens solutions.

## Lossless local encoding

Let rows be the strand heights immediately before a word letter and columns be
word positions.  A cell contains three route bits plus one mask bit:

| local route | bits |
|---|---:|
| inactive/padding | `000`, mask `0` |
| straight | `010`, mask `1` |
| left and under | `100`, mask `1` |
| left and over | `110`, mask `1` |
| right and under | `001`, mask `1` |
| right and over | `011`, mask `1` |

For positive `sigma_i`, rows `i` and `i+1` contain `011` and `100`; for
negative `sigma_i`, they contain `001` and `110`.  Every other active row is
`010`.  The pair constraint is enforced by the compiler, not learned by the
network.  The explicit mask distinguishes absent capacity rows from meaningful
cells and supports masked pooling when a checkpoint is evaluated at larger `k`.

Two growth operations must not be confused:

* inserting an all-straight **column** inserts the identity and is safe;
* inserting an unused straight **row** adds an unlinked closure component and
  changes a knot into a link.  Strand-count growth must use a Markov
  stabilization: add the row together with one terminal `sigma_k` or
  `sigma_k^-1` crossing.  Destabilization is its checked inverse.

## Architecture family

All serious variants use shared weights and dynamic masked shapes.  No trunk
parameter, absolute position embedding, or dense flattening may depend on `k`
or `n`.

### `conv-window-128`: representation ablation

This is now implemented in `pgx-mcts-bench`.  It changes only the input/trunk of
`s-window-128`:

1. compile the centred seven-letter window to `k x 7 x (3+mask)`;
2. use bounded-row/circular-column residual convolutions with GroupNorm;
3. masked mean and max over rows recover one feature per column;
4. reuse the existing positional policy, global actions, value heads, MCTS, and
   training protocol.

This arm is intentionally not the final scalable policy: its insertion logits
still enumerate generators up to `max_strands`.  It cleanly answers the first
question—whether the raster is a better input for the proven window learner.

Two representation variants are also executable:

* `conv-window-axial-128` gives word-neighbour and strand-neighbour interactions
  separate shared convolutions before mixing them;
* `conv-window-recurrent-128` reuses one axial block four times, increasing
  receptive field by inference-time stacking without adding new block weights.

### First-stage smoke, seed 71 (2026-08-08)

This is a wiring/early-learnability result, not an architecture ranking.  All
arms used 128 MCTS simulations, four self-play games per iteration, 96 optimizer
steps, four evaluation attempts per ratio, and at most three iterations on
`unknot+2`.

| arm | parameters | iterations | representation SR | result |
|---|---:|---:|---:|---|
| `s-window-128` | 140,308 | 2 | 100% | objective |
| `conv-window-128` (joint 3x3) | 137,620 | 3 | 50% | capped |
| `conv-window-axial-128` | 117,012 | 2 | 100% | objective |
| `conv-window-recurrent-128` | 108,756 | 2 | 100% | objective |

The first joint design failed the 70% validity floor.  Its failure is not
evidence against the raster: separating horizontal and vertical interactions
restored exactly the old learner's 100% early solve rate, and recurrent stacking
did so with the smallest model.  The axial and recurrent variants have advanced
to a paired five-rung gate; neither is yet comparable with mature rung-18 or
mixed-strand checkpoints.

### First `B*` smoke, seed 71 (2026-08-08)

All networks were initialized from scratch.  The action head has one additional
signed generator per actionable position, although that generator is masked on
the two-strand elementary rung because it duplicates `sigma_1`.

| arm | iterations | representation SR | result |
|---|---:|---:|---|
| `s-window-128-bstar` | 2 | 100% | objective |
| `conv-window-axial-128-bstar` | 3 | 50% | capped below validity floor |
| `conv-window-recurrent-128-bstar` | 3 | 50% | capped below validity floor |

This proves that the enlarged observation/action schema does not break the old
window learner.  It did not rank the convolutional arms: 50% was below the
pre-registered 70% floor, and earlier architecture work showed delayed takeoff.

The clean six-iteration continuation confirmed that delay.  Axial reached 83.3%
overall SR and promoted after iteration 4 (`1000:1` 50%, `10:1` 100%, `1:10`
100%).  Recurrent reached 100% at all three objectives and promoted after
iteration 6.  Both therefore enter the paired five-rung `B*` comparison, but the
axial objective imbalance is a reported warning rather than hidden by its pooled
rate.  The longer run uses eight self-play games, twelve evaluation attempts,
`F_old=1`-style retrospective rehearsal, seed 71, and fresh initialization for
all three `B*` arms.

### Adaptive old-rung rehearsal

The corrected ladder no longer treats `F_old=1` as a sufficient constant. Each
earlier rung starts at one fresh MCTS rehearsal attempt per frontier cycle. At
every evaluation the controller measures a paired retention panel at the
crossing-dominant objective. A rung is unhealthy when its solve rate is below
80%, or its mean crossing changes on solved attempts exceed the proved `u` by
more than 0.25. Only that rung's dose doubles (`1 -> 2 -> 4 -> 8`); healthy
rungs remain at their existing dose. A frontier success cannot promote while an
old-rung certificate is unhealthy. The dose vector and every probe are
checkpointed, and rehearsal seeds are stable when the dose grows.

The first paired five-rung test used fresh `s-window-128` weights, trained only
for `L1000`, eight frontier games, and 256-simulation final evaluation. A small
eight-attempt retention panel underdetected failures. With 24 retention attempts,
the adaptive arm raised only rung 1 to `F_old=8`; the other old rungs stayed at
one. On the same 120 final attempts it solved 118 and attained exact `u` on all
118, versus 112 solves / 107 exact-`u` for fixed `F_old=1` and 115 / 109 for the
original broad-objective baseline. Unsolved attempts charged at `64,064` gave
capped `L1000` totals of 200,624, 587,985, and 405,805 respectively. On the 114
attempts shared with the baseline, adaptive rehearsal also reduced uncapped cost
from 81,473 to 70,460. This is a positive single-seed result, not yet a
multi-seed architecture conclusion.

Artifacts:

* `pgx-mcts-bench/artifacts/s-window-l1000-fold1-five-rung-seed71-v2/`;
* `pgx-mcts-bench/artifacts/s-window-l1000-adaptive-fold-five-rung-seed71-v2/`;
* `pgx-mcts-bench/artifacts/s-window-l1000-adaptive24-fold-five-rung-seed71/`;
* `pgx-mcts-bench/artifacts/conv-window-family-five-rung-seed71/`.

### Full-canvas scalable controller and critic warmup, 2026-08-08

The first fully strand-capacity-independent policy is now executable. It scans
the complete head-relative word into a `max_strands x max_len` canvas, reuses one
axial residual block at least four times, scores every adjacent row pair with one
shared insertion head, and gathers the seven current semantic action sites from
the compact occupied word. The checkpoint parameter shapes are identical when
`max_strands` changes. Unused columns can be all-straight identity slices without
becoming no-op MCTS actions.

For `B*`, the torus variant wraps at each state's **active** strand count, not at
the configured capacity. Thus a four-strand braid inside a five-row tensor sends
messages directly between rows 4 and 1 and masks row 5. The cylinder variant
keeps that interaction only in the seam action scorer. A separate safe-height
ablation uses a genuine Markov stabilization; an all-straight active row remains
forbidden because it adds a closure component.

The representation screen on seed 71 selected identity columns: it reached 100%
elementary SR, versus 50% for inactive full-canvas padding and 33.3% for adding
one initial Markov stabilization. Two fresh identity-column confirmations reached
91.7% and 100%.

Activating a newly initialized factorized critic in MCTS immediately was a
negative control, not a capacity result. After four cycles, cylinder seeds
71/72/73 reached only 33.3%, 0%, and 50%; torus reached 66.7%, 0%, and 50%.
Three differences from the proven learner were then isolated and repaired:

1. stage 0 trains `p(solve)`, conditional crossings, and conditional moves in
   shadow while MCTS retains the legacy scalar;
2. solve loss is detached from the shared encoder during that warmup, then gains
   encoder gradients together with factorized-search cutover at stage 1; and
3. balanced replay supplies four spread positions from every selected episode,
   rather than silently falling back to one.

The new remaining-budget metadata weight is zero-initialized, so adding the
channel does not randomly perturb a fresh raster controller before learning.
With this protected warmup, the cylinder restored 12/12 solves, zero crossing
changes, and promotion after two cycles on all three seeds. The dynamic torus
matched it on seeds 71 and 72, but seed 73 needed four cycles and still averaged
0.583 crossing changes; despite 12/12 solves it correctly failed the objective
gate. The cylinder therefore advances to the adaptive six-stage mixed-strand
gate, while the torus remains an ablation.

Primary artifacts:

* `pgx-mcts-bench/artifacts/conv-cylinder-idcols-factorized-balanced-vector-stage0-seed{71,72,73}/`;
* `pgx-mcts-bench/artifacts/conv-torus-idcols-factorized-balanced-vector-stage0-seed{71,72,73}/`;
* `pgx-mcts-bench/artifacts/conv-cylinder-idcols-protected-warmup-balanced4-stage0-seed71/`; and
* `pgx-mcts-bench/artifacts/conv-torus-idcols-protected-warmup-balanced4-stage0-seed{71,72,73}/`; and
* `pgx-mcts-bench/artifacts/conv-cylinder-idcols-mixed6-adaptive24-seed71/` (running).

### `conv-cylinder-local`

A fully convolutional policy emits local rewrite/crossing-change scores at every
valid cell pair.  Global action kinds are pooled.  Insert/stabilize actions use a
factorized `(action kind, row, column, sign)` pointer instead of a fixed generator
alphabet.  This removes the remaining `max_strands` dependency.

### `conv-cylinder-pyramid`

Use shared local blocks at the fine scale, then masked `2 x 2` coarsening.  Apply
the same combiner recursively, so the hierarchy can represent `2 x 2`, `4 x 4`,
and larger blocks without introducing a new set of weights for every scale.
Odd sizes are padded under a mask.  A U-Net/FPN top-down path sends coarse context
back to fine cells; a hierarchical pointer first selects a coarse block, then a
child block, then the action site.  This is the precise version of “the 4 x 4
embedding spots the small block.”

Fixed non-overlapping blocks create an arbitrary seam.  Either use overlapping
blocks or alternate the block origin by half a block between layers.  Column
origins wrap; row origins respect the boundary.

### `conv-cylinder-recurrent`

Repeat one residual interaction block `R` times with shared parameters.  Train
with randomized `R` (for example 4--8), then test at 16 and 32.  More inference
compute expands the receptive field without widening the checkpoint.  This is
the most literal stackable architecture and should be the first hierarchical
competitor because it is much simpler than the pyramid.

### `conv-cylinder-dual`

Fuse the raster tower with the exact physical-strand crossing graph already used
by `s-strand-graph-128`.  Local image neighborhoods make braid relations easy;
graph edges make two distant crossings on the same physical strand adjacent.
Cross-attention or gated sums at every scale test whether those views are
complementary.  This is the highest-capacity variant, not the first one to run.

### `conv-torus-ablation`

Wrap the row coordinate too, with a seam channel and illegal affine actions
masked.  It tests whether computational symmetry helps despite the artificial
adjacency.  It must never be reported as the faithful braid model.

## Cheap gates before reinforcement learning

### Gate 0 — representation correctness

For random words over `k=2..12`, word lengths `1..128`, and both signs:

* exact word -> raster -> word round trip;
* pair consistency and illegal-state rejection;
* equivariance under cyclic column rotation;
* transition equivalence for every existing semantic action;
* identity-column insertion leaves the decoded word equivalent;
* stabilization/destabilization preserves closure, while bare-row insertion is
  rejected.

### Gate 1 — scale extrapolation without MCTS

Train on `k<=5, n<=32`; validate separately on familiar sizes and unseen
`k=6..10, n=33..96`.  Use abundant exact labels:

* legal action mask and rewrite site;
* inverse/cancellation and braid-relation sites;
* closure component count;
* next semantic action from stored successful witnesses;
* conditional crossing-change and move costs.

A model is not “strand agnostic” merely because its tensor accepts a larger
height.  It passes only if unseen-size loss remains close to familiar-size loss.

### Gate 2 — corrected early curriculum

Run three paired seeds through the same elementary ladder that made
`s-window-128` learn.  Use identical search simulations, training examples,
optimizer, replay balance, and `F_old=1`.  Evaluate every cycle.  Require at least
70% representation solve rate before comparing objective quality; otherwise the
comparison is search-starved.

### Gate 3 — mixed-strand complementarity

Evaluate paired attempts on a source-disjoint panel containing 2-, 3-, 4-, 6-,
and 8-strand representations.  Report:

* solved-set intersection and each arm's exclusive solves;
* capped `L1000` over the whole panel and uncapped `L1000` on shared successes;
* optimal-`u` attainment where `u` is proved;
* network evaluations, wall time, parameters, and peak memory;
* the extrapolation gap from trained to unseen strand counts.

The comparison roster is `s-window-128`, `s-cyclic-tape8-192`,
`s-strand-graph-128`, `conv-window-128`, and the best fully scalable convolutional
variant.  Use the same attempts and stochastic seeds, and keep at least 100
representations for any conclusion about scheduling or collaboration.

## Selection order and stopping rules

1. Compare old tokens against `conv-window-128`; kill the raster input if it is
   slower and does not improve paired solve coverage or `L1000`.
2. Compare local and recurrent shared blocks at matched network evaluations;
   keep the recurrent model only if extra repeats buy measurable unseen-size
   gains.
3. Add the pyramid only if long-range failure remains after recurrent depth.
4. Add the graph tower only if raster and graph scientists have complementary
   solved sets or their exact supervised tasks reveal complementary errors.
5. Admit one `conv-*` scientist to the 200-representation collaboration study
   only after three-seed retention and mixed-strand gates pass.

The central hypothesis is therefore falsifiable: shared local structure should
preserve early learnability while producing a smaller degradation as `k` and `n`
move outside the training range.  Parameter count alone is not success.

## Relation to published work

The closest precedents found are separate pieces rather than this exact model.
[Learning to Unknot](https://arxiv.org/abs/2010.16263) uses braid words as a
language and reports Transformer and RL simplification;
[rectangular/Dynnikov-diagram work](https://arxiv.org/abs/2011.03498) presents
knots as a grid and learns classification; [recent diagram-based
RL](https://arxiv.org/abs/2603.07955) applies to arbitrary knots and links.  These
support sequence learning, spatial knot encodings, and variable diagram scope
respectively.  I found no primary-source example in the searched literature that
combines an exact periodic braid raster, shared multiscale 2D blocks, and a
spatial RL action head whose parameters are independent of strand count.  That
is a novelty claim to investigate more systematically before a paper, not a
claim that no such work exists.
