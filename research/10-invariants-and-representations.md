# 10 — Invariants as leverage, and the choice of representation

> **Scope constraint added since this was written.** The experiment is unknotting
> with **zero human knowledge**, which reclassifies rather than cancels most of what
> follows. Proposals 4 and 5 below hand the network human-computed answers, so they
> are **oracle arms**: run them to bound the upside before building the fair
> version, quote them as oracles, never as the headline. Proposals 1, 2 and 3 are
> unaffected — a transposition table is search bookkeeping, certified lower bounds
> verify output rather than feed input, and potential-based shaping on word length
> and strand count uses only raw state. Note also that the observation *already*
> carries three human-knowledge channels (the `top_generator` plane and the two
> `top_count` scalars), so the standard is currently applied unevenly; the audit is
> in [12 §4](12-serial-formulation.md).

> **Partly built since.** Proposal 2's certified lower bound `|σ|/2`, and knot
> identification against a table, are implemented in `rf_knots.invariants` and
> `rf_knots.knot_table`, with the ladder's own knots worked out in
> [../docs/rungs.md](../docs/rungs.md). They verify output rather than feed input,
> so the zero-knowledge constraint above is untouched. Rasmussen `|s|/2`, `|τ|`,
> and branch-and-bound over them remain unbuilt.

Written after reading Sossinsky's *Узлы: хронология одной математической теории*,
chapter 7 (finite-order invariants), which turns out to contain the reframing this
project needs.

---

## 1. The reframing: unknotting number is a distance in a stratified space

Sossinsky's picture of Vassiliev's construction is this. Consider the space `F`
of **all** knots, including *singular* ones with transverse double points. The
ordinary knots form `Σ₀`; the singular ones form the **discriminant**
`Σ = Σ₁ ∪ Σ₂ ∪ …`, stratified by number of double points. A path through `F`
that deforms one knot into another passes through `Σ₁` exactly when a strand
crosses another — that is, **exactly at a crossing change**. He defines an
invariant by declaring its value at the unknot and specifying that it changes by
±1 at each transversal crossing of `Σ₁`, with the well-definedness (path
independence) being the hard theorem.

Read that back with the unknotting number in mind:

> `Σ₁` is a **wall**. Knot types are **chambers**. A crossing change is a
> **wall crossing**. And `u(K)` is the **minimum number of wall crossings on any
> path from `K`'s chamber to the unknot's chamber.**

So `u` is a *graph distance*. Let `G` be the graph whose vertices are knot types
and whose edges join knots differing by a single crossing change. Then

```
u(K) = d_G(K, O)
```

This is not a new theorem — it is the definition, seen geometrically. But it is
the right frame, and it has three consequences that are directly actionable.

### Consequence A — the state of the search should be the knot, not the diagram

Search happens in `G`, whose vertices are knot *types*. Every diagram of the same
knot is the same vertex. So an invariant fingerprint (Jones, Alexander,
signature, `n`, writhe) should be the **transposition-table key** in MCTS. Two
search branches reaching the same knot by different move sequences currently
count as different nodes; they should not. This is a pure search-efficiency win
and is cheap to implement.

### Consequence B — lower bounds are exactly the 1-Lipschitz functions

Graph distance has a dual characterisation. For any function `f` on `G` that is
**1-Lipschitz** (`|f(K) − f(K')| ≤ 1` whenever `K, K'` differ by one crossing
change) and vanishes at the unknot,

```
|f(K)| ≤ d_G(K, O) = u(K)
```

and `u` itself is the *largest* such function. **Every unknotting-number lower
bound in the literature is an instance of this.**

| bound | statement | why it is 1-Lipschitz |
|---|---|---|
| signature (Murasugi) | `|σ(K)|/2 ≤ u(K)` | `σ` changes by at most 2 under a crossing change |
| Rasmussen `s` | `|s(K)|/2 ≤ u(K)` | same, from Khovanov homology |
| Ozsváth–Szabó `τ` | `|τ(K)| ≤ u(K)` | `τ` changes by at most 1 |

Two things follow immediately.

**The max of certified pieces is certified.** A maximum of 1-Lipschitz functions
is 1-Lipschitz, so `f = max(|σ|/2, |s|/2, |τ|)` is a valid bound with no new
mathematics. Compute all of them and take the max — cheap, and it is what turns
"we found a short unknotting sequence" into "we determined `u(K)`".

**Finding new bounds = finding new bounded-jump functions.** This is where
machine learning has an honest role, and it is *not* "predict `u(K)`". It is:
search for a computable invariant whose jump across `Σ₁` is bounded, then have a
human prove the jump bound. That is the same division of labour that produced
murmurations — the network finds the pattern, the mathematician proves it. The
self-play loop generates the constraint samples **for free**: every crossing
change made during search is a labelled pair `(K, K')` at graph distance 1.

Honest caveats: `G` has infinitely many vertices and each vertex has infinite
degree (a knot has infinitely many diagrams, hence infinitely many
crossing-change neighbours), so Lipschitzness can only ever be *tested* on
samples, never verified by enumeration. A learned `f` therefore yields
**conjectural** bounds plus a pointer to where to look — not proofs.

### Consequence C — branch-and-bound, not just MCTS

In unknotting-number mode the true value is `V*(s) = −u(K_s)`. A certified lower
bound `L(s)` gives `V*(s) ≤ −L(s)`, which is an **admissible heuristic**. That
buys two things a plain value network cannot:

* **Pruning.** Any branch whose optimistic value cannot beat the best sequence
  found so far is cut. This is classical branch-and-bound inside the tree.
* **Proof.** When the best found sequence has length `L(s)`, the node is *solved
  exactly* — `u(K) = L(s)` is a theorem, not an estimate. This is precisely how
  the 43 previously-unknown unknotting numbers in
  [arXiv:2409.09032](https://arxiv.org/abs/2409.09032) were pinned down.

**This is the single highest-leverage change available**, because it converts the
system's output from a benchmark score into mathematics.

## 2. Where invariants do *not* help — and this matters

In the Scrambler-vs-Simplifier game as built, **the closure is the unknot at
every single state, by construction**. Therefore

```
V(t) ≡ 1,   Δ(t) ≡ 1,   σ ≡ 0,   s ≡ 0,   τ ≡ 0,   every v_n ≡ v_n(O)
```

throughout the entire episode. Knot invariants are *constant* along the whole
trajectory and carry **exactly zero** information about progress toward the
trivial word. Any reward shaping built from them is identically zero.

That is worth stating plainly because it is the obvious first idea and it is
dead. What changes during the simplification game is the **diagram**, not the
knot. So the only usable progress signals there are diagram complexity measures —
word length, strand count, and combinations of them. Those *are* usable, via
**potential-based reward shaping**: with `Φ(s) = −(length + strands)`, the shaped
reward `γΦ(s') − Φ(s)` leaves the optimal policy unchanged (Ng–Harada–Russell)
while densifying a very sparse signal. That is safe, cheap, and worth a variant.

This also explains an observation in the neighbouring literature: Dranowski–
Kabkov–Tubbenhauer name "sparse progress" as one of their two central
difficulties, noting that R3 moves do not change the crossing number at all. They
are describing the same hole.

## 3. Vassiliev invariants specifically

Sossinsky's chapter gives the structure that makes them interesting *for the
action space*, not for the reward.

A Vassiliev invariant is defined by its behaviour across the wall: `v` extends to
singular knots by the skein relation

```
v(K₊) − v(K₋) = v(K_singular)
```

where `K_singular` has one extra double point. So the **jump of an order-`n`
invariant across `Σ₁` is itself an order-`(n−1)` invariant** of the singular knot.
Kontsevich's theorem (which Sossinsky states) identifies `V_n/V_{n−1}` with the
space `A_n` of chord diagrams modulo the one-term and four-term relations; these
are small (`dim A₃ = 1`, `dim A₄ = 3`), so low-order invariants are a *short*
feature vector, not a sprawling one.

The actionable consequence:

> For each crossing `p` in the current diagram, the exact change `Δv_n` that
> flipping `p` would cause is computable **before making the move**, and it is a
> lower-order invariant.

That is a per-action feature vector, aligned one-for-one with the
`CROSSING_CHANGE(p)` block of the action space. Feeding it as extra observation
channels gives the policy head a direct, computed answer to "what does changing
this crossing do to the invariants?" — which is the actual question. The cheapest
useful instance is the Casson invariant `v₂ = a₂` (second Conway coefficient),
obtainable from the Alexander polynomial, obtainable from the Burau
representation of the braid word in `O(n³L)`. Cheap enough to compute in the
environment loop.

One caution, since it is the tempting mistake: Sossinsky's pedagogical `v₀`
(value 0 on the unknot, `+1` on the trefoil, `−1` on the figure-eight,
blind to mirror images) reads as the Casson invariant `a₂`, and his "changes by
±1 at each wall crossing" is a simplification. `a₂` in fact changes by a linking
number, which is **not** bounded by 1. So `u(K) ≥ |a₂(K)|` is *not* a theorem —
do not use it as a bound. `a₂` is a good *feature*; `σ`, `s`, `τ` are the bounds.

## 4. The grid/tile representation

Your proposal — an `N × N` table of cells, each empty / `|` / `−` / one of four
corner arcs / one of two crossings — is **knot mosaics** (Lomonaco–Kauffman
2008). It is a real, studied representation, and mosaic knot theory is known to
be equivalent to tame knot theory, so nothing is lost in principle. It comes with
its own Reidemeister-style local tile moves, and there is a tabulation literature
(e.g. *Tabulating Knot Mosaics: Crossing Number 10 or Less*,
[arXiv:2303.12138](https://arxiv.org/abs/2303.12138)).

**Where your intuition is right.** Moves become local tile rewrites, so the
action space is `position × move-type` — the same clean shape as the braid
environment. Over/under is explicit per cell. And a genuinely 2-D state is the
natural home for a convolutional network, whereas the braid word is a sequence
that only *encodes* planarity.

**Where it costs.** State size is `O(N²)` in tiles times 11 channels, and — the
real problem — simplification often needs **slack in two dimensions** rather than
one, so the padding cost is quadratic where the braid word's is linear.
Validity is also a *global* constraint: tile connection points must match across
every edge, so most tile assignments are not diagrams at all, and every move must
preserve that. Long-range moves (slide a strand across the diagram) decompose
into many local steps, lengthening episodes.

**I would use grid diagrams instead.** An `n × n` grid with exactly one `X` and
one `O` per row and column, joined by horizontal and vertical segments with a
fixed over/under convention. This keeps everything you want and drops the costs:

| | mosaics | grid diagrams |
|---|---|---|
| genuinely 2-D | yes | yes |
| state size | `N² × 11` one-hot | two permutations of `n`, or `n × n × 2` |
| valid states | rare; global constraint | *every* pair of permutations is valid |
| move set | many planar-isotopy + Reidemeister tile moves | commutation, (de)stabilisation, cyclic permutation — finite and clean (Cromwell's theorem gives completeness) |
| connects to invariants | no | **yes** — grid diagrams are the combinatorial home of knot Floer homology, which is where `τ` comes from |

That last row is the argument. `τ` is one of the three certified unknotting-number
lower bounds from §1, and it is *computable from the same combinatorial data the
agent is manipulating*. In the braid representation you compute bounds in a
separate pipeline; in the grid representation the bound and the state are the same
object.

**On "will the network understand 2-D and then 3-D better".** I would resist that
framing. The evidence in this domain points the other way: sequence models on
braid *words* did well in *Learning to Unknot*, and the DeepMind/Oxford work found
the **Jones polynomial** — an algebraic invariant, not a picture — to be the most
useful feature in their models. What actually determines whether a representation
works is narrower and more testable: **are the relevant moves local, and are the
relevant invariants computable from the state?** Mosaics score yes/no. Grid
diagrams score yes/yes. Braid words score yes/partly.

**Recommendation.** Keep braid words for the Scrambler-vs-Simplifier game — they
are built, verified, and fast. Add **grid diagrams** as a second environment when
the project moves to unknotting-number mode, and run the controlled comparison:
same game, same budget, two representations. That comparison is itself a result
worth reporting, and it answers your question with data instead of intuition.
Mosaics are the interesting third option only if the 2-D convolutional bias turns
out to be the deciding factor, which the grid experiment will reveal more cheaply.

## 5. Concrete proposals, ranked

1. **Invariant-keyed transposition table in MCTS.** Cheap, pure win. (§1A)
2. **Certified lower bounds `max(|σ|/2, |s|/2, |τ|)` computed per knot, cached**,
   and branch-and-bound pruning plus exact closure when upper = lower. This is what
   produces theorems. (§1B, §1C)
3. **Potential-based shaping on diagram complexity** for the simplification game,
   since knot invariants are constant there and cannot help. Policy-invariant, so
   it is safe. (§2)
4. **Per-crossing `Δv_n` channels** aligned to the `CROSSING_CHANGE` action block,
   starting with the Casson invariant. (§3)
5. **Grid-diagram environment** for unknotting-number mode, with a controlled
   comparison against braid words. (§4)
6. **Learned 1-Lipschitz potential** on the crossing-change graph, trained on the
   distance-1 pairs that self-play produces for free, treated as a *conjecture
   generator* whose output a human proves. Highest risk, highest ceiling, and the
   natural bridge to the LMFDB endgame in [07](07-domain-choice.md). (§1B)
