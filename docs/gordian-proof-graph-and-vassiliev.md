# Gordian proof view and Vassiliev sidecar

Status: design contract for the next data layer. Catalogue claims are not proof
edges until an exact primitive program has independently replayed.

## Shared proof substrate

Unknot and Gordian routing share one representation registry, one immutable
program dictionary, and one immutable directed edge store. Every edge has a
replayed semantic cost of zero or one crossing change. Logical views are kept
separate:

- `unknot_routes(source, U_upper, next_edge)` is the shortest route to a
  certified unknot terminal;
- `gordian_routes(source, target, d_upper, next_edge)` is a sparse, query- and
  campaign-driven cache of shortest verified routes between arbitrary targets;
- `gordian_lower_claims(source_knot, target_knot, lower, provenance)` stores
  catalogue or theorem lower bounds and never masquerades as a proof edge.

The Gordian view is sparse. It stores useful queried pairs, exact pairs, and
frontiers, not a quadratic all-pairs closure. A path remains reconstructible
from immutable edges; route rows are derived and replaceable.

Cross-relaxation uses only verified upper paths:

```text
U(A) <= d_upper(A, B) + U(B)
d_upper(A, B) <= U(A) + U(B)
```

When a verified upper path meets a trusted lower claim, the pair is exact.
Zero-CC paths may propagate attested knot identity; lower-bound agreement alone
must not.

## Vassiliev invariant sidecar

Values belong to an attested knot identity, with representation postings kept
separately. Each row records:

- knot identity and orientation/mirror convention;
- degree;
- basis/weight-system identifier and ordering;
- coefficient domain and normalization;
- calculator version and input/provenance hash;
- verification status.

The reverse index stores hashes of complete degree prefixes
`(v_1, ..., v_N)`. Equal prefixes generate equivalence candidates; a differing
verified component proves non-equivalence but does not by itself quantify
Gordian distance.

## Curriculum strata

A degree-`N` Vassiliev hard negative agrees in every stored component through
degree `N` and differs in at least one component of degree `N+1`. Difficulty is
tracked by `N`, but evaluation also records word length, crossing number,
mirror relation, and whether simpler invariants already separate the pair.

Splits are by attested knot identity. All representations, mirrors, scrambles,
and intermediate edge states of a held-out identity stay out of training.
Unknown identities are never silently promoted to the held-out set.

The sidecar is useful for candidate generation and training, but finite-prefix
equality is not an isotopy certificate. Candidate equivalence requires an
attested common identity or a replayed zero-CC path.

## Branching training trajectories and leakage quarantine

Training uses greedy checkpoint branching, not an experiment comparing seeds.
At rung zero, five trajectories train for one million examples and the best
validation checkpoint survives. At every later rung, that complete checkpoint
(model, optimizer, scaler and schedule state) is forked into five continuations,
each trains for the next million examples, and again only the best survives.

A master trajectory seed expands into recorded seeds for pair selection,
zero-CC augmentation, shard order, dropout and other stochastic operations. It
may therefore select a genuinely different admissible data sequence, not merely
a different parameter initialization. Branch corpora need not have identical
membership. Each branch materializes its own audited shards, derived from
`parent_checkpoint_sha256, rung, master_seed`, before GPU training.

The selection objective and validation protocol are immutable. Every branch
result and generated-corpus manifest is retained for provenance, but branch
means or seed comparisons are not the product objective. The untouched test
split is evaluated once after the final winning trajectory is selected.

Validation contains 15--20 anchor knot identities. Seventy percent are
stratified hard identities and thirty percent are deterministically sampled
from the train-like input distribution. Every known representation, mirror,
preprocessing result and proof-edge intermediate state of an anchor belongs to
the same split.

Before materializing each branch's shards, build a quarantine in this order:

1. all representations carrying an anchor knot identity;
2. the complete verified CC=0 component of every anchor representation;
3. a buffer of all vertices reachable through at most one verified CC edge,
   closing under CC=0 before and after that edge;
4. structural duplicates identified by exact canonical/orbit keys;
5. pair-level near duplicates detected by one frozen, versioned audit encoder.

The embedding gate is secondary and may only remove examples. It cannot admit
an example rejected by exact identity or graph rules. For unordered pairs use
the better endpoint alignment when comparing a candidate pair `(a,b)` with a
validation pair `(x,y)`:

```text
max((cos(a,x)+cos(b,y))/2, (cos(a,y)+cos(b,x))/2)
```

The audit encoder, threshold and validation prototypes are frozen before the
branch search starts. Rejected near-neighbours form a separate near-validation
challenge set instead of being discarded. Every branch manifest records exact
closure sizes, rejection reasons, similarity quantiles, maximum remaining
similarity and hashes of every membership list.

Repeated selection can overfit a small validation set even though validation
examples never enter training. Scores are therefore aggregated per anchor knot
identity rather than per representation, use a fixed core plus fresh replayable
validation scrambles, and require a predeclared minimum margin before replacing
the parent. The final test identities remain completely untouched.
