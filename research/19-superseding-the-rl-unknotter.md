# 19 — Superseding the RL unknotter, and what a network must be able to represent

> Reading of *RL unknotter, hard unknots and unknotting number*, Dranowski,
> Kabkov, Tubbenhauer, [arXiv:2603.07955](https://arxiv.org/abs/2603.07955), with
> code at [dtubbenhauer/unknotter](https://github.com/dtubbenhauer/unknotter),
> [dtubbenhauer/upperbounds](https://github.com/dtubbenhauer/upperbounds) and
> [annedranowski/untangling-number](https://github.com/annedranowski/untangling-number).
> Companion to [10](10-invariants-and-representations.md) (unknotting number as a
> graph distance) and [18](18-raster-representation.md) (the diagram as a picture).

---

## 1. What they actually do

Worth stating precisely, because the shape of the system is where the openings are.

**The pipeline.** Their upper-bound loop runs on a live spreadsheet of knots whose
unknotting number is only known as an interval `[l, u]` with `l != u`:

1. **inflate** — a bounded random Reidemeister walk from a base diagram of `K`, up
   to a target crossing count (about 20), which is knot-type preserving;
2. **flip** — apply exactly **one** crossing change, at each crossing in turn;
3. **simplify** — run the trained agent for a fixed step budget;
4. **identify** — compute the Jones polynomial of the reduced diagram and match it
   against KnotInfo, trusted only for prime knots of 3–13 crossings;
5. **update** — if some single flip lands on `K'` with `u(K') = m`, record
   `u(K) <= m + 1`;
6. repeat until the interval collapses.

**The agent.** This is the part worth dwelling on. The observation is a
**six-dimensional vector**: crossing number, component count, a step counter, two
binary flags ("does basic simplification reduce crossings", "did the last action
reduce crossings"), and a bias. The action is a pair (macro-action, repeat count)
over four macro-actions, three of which are calls into `spherogram`'s own
`simplify` routines and one of which is stochastic backtracking. The algorithm is
PPO with an MLP policy.

So **the network never sees the diagram.** It is a meta-controller choosing which
canned simplifier to call and when to backtrack; the knot theory is done by
`spherogram`. That is not a criticism of the result — it works — but it bounds
what the learned part can be responsible for.

**The results.** 94.57% per-run success on the 385-instance "very hard unknot"
benchmark, with all 385 unknotted in at least one of ten runs; **72** knots with
improved intervals in their Table 1, of which **30** intervals collapsed to exact
values in that published snapshot; and `4_1 # 9_10` recovered at `u <= 3` via a
single flip to `15n4866`, which has `u = 2`.

The published rows, a current KnotInfo overlay, and executable braid
representations are frozen in
[`../benchmarks/dkt2026-table1-upper-bounds-v1.json`](../benchmarks/dkt2026-table1-upper-bounds-v1.json),
[`../benchmarks/dkt2026-table1-knotinfo-20260814.json`](../benchmarks/dkt2026-table1-knotinfo-20260814.json),
and
[`../benchmarks/dkt2026-table1-authors-pd-braids-v1.json`](../benchmarks/dkt2026-table1-authors-pd-braids-v1.json).
The current KnotInfo snapshot has since made 48 of the 72 values exact; 24 remain
open and are the actual upper-bound contribution targets.

**Their own stated limits.** No certification of minimality; Jones identification
detects only about 73% and fails outside 3–13 crossings; results are empirical
upper bounds within a step budget; composite examples like `4_1 # 9_10` are not
automated.

## 2. The four openings, ranked by leverage

### 2.1 Their search is depth one in the crossing-change graph

Step 5 is `u(K) <= 1 + u(K')`. One flip, then a table lookup. Everything harder
than "one crossing change from a knot whose `u` is already known" is out of reach
by construction.

[10 §1](10-invariants-and-representations.md) gives the right frame: `u` is the
graph distance `d_G(K, O)` in the crossing-change graph. Their loop explores the
ball of radius one around `K` and asks whether it meets a labelled vertex. The
natural generalisation is a **depth-`k` search**, `u(K) <= k + u(K_k)` — and when
`K_k` is the unknot the table is not needed at all, because the witness is
self-certifying.

**This project's environment is already the depth-`k` version.** Unknotting-number
mode composes crossing changes with the full Markov move set and terminates on the
empty 1-braid, so a solution is a machine-checkable sequence rather than a lookup.
That is the single biggest structural difference and it is in our favour.

### 2.2 Identification is the bottleneck, and it is avoidable

A 73% detection rate means better than one in four successful simplifications is
thrown away, and the 3–13 crossing trust window caps how far inflation can go.
Two independent fixes:

* **A richer fingerprint.** `rf_knots.invariants` already computes Alexander,
  Jones, determinant, genus and signature, and `knot_table` matches against a
  bundled KnotInfo snapshot. [10](10-invariants-and-representations.md) also
  records the trap: 384 fingerprints in a 2870-knot table are shared, and `5_1`
  and `10_132` agree on **both** Jones and Alexander while having different
  unknotting numbers. So a multi-invariant fingerprint must report ambiguity
  rather than take the first match.
* **Certified lower bounds, which remove the lookup entirely.** If after `k` flips
  you can *certify* `u(K_k) = m` rather than look it up, the trust window
  disappears. This is [10 §1B/§1C](10-invariants-and-representations.md) —
  `max(|sigma|/2, |s|/2, |tau|)` and branch-and-bound — of which only `|sigma|/2`
  is implemented here. **This is the highest-value unbuilt item in the repository,
  and it is exactly the thing their pipeline cannot do.**

There is a further cheap obstruction they do not use and neither do we:
**Lickorish's criterion**, via the Montesinos trick — if `u(K) = 1` then
`H_1(Sigma_2(K))` is cyclic of order `det(K)` and its linking form takes a
prescribed value. It certifies `u >= 2` outright for many knots and needs only a
Goeritz matrix. The exact statement should be checked against Lickorish before
implementing; Stoimenow's *Polynomial values, the linking form and unknotting
numbers* is the survey of this family.

### 2.3 Their agent cannot choose *where* to flip

Step 2 flips **every** crossing, because a six-number observation cannot rank
crossings — the policy has no way to refer to one. After inflation to ~20
crossings that is twenty simplifier runs per diagram, and the simplifier is the
expensive part.

A network that sees the diagram can rank crossings, turning an exhaustive loop into
a guided search and freeing an order of magnitude of budget to spend on depth
(§2.1). This is precisely what [18](18-raster-representation.md) built the head
for: a cell-indexed policy over the raster gives one logit per crossing, with a
parameter count independent of the strand count.

Note the honest asymmetry: they work in PD codes via `spherogram`, which handles
arbitrary diagrams; we work in braid closures, which is complete by Markov's
theorem but is a restriction on *diagrams* and pays for it in word length.

### 2.4 Composite knots are manual, and we have the machinery

They flag automating composites like `4_1 # 9_10` as future work.
`rf_knots.invariants.connected_summands` already splits a braid word into prime
summands. Subadditivity `u(K_1 # K_2) <= u(K_1) + u(K_2)` is free; the interesting
cases are exactly where it is not tight, which is why `4_1 # 9_10` is a case study
at all.

## 3. What "supersede" honestly requires

Two prerequisites, and one calibration that should temper the ambition.

**A witness, or it is not a result.** An upper bound is a claim about a sequence,
so the sequence has to be stored and re-verifiable. `artifacts/bounds.jsonl`
currently records the knot's *defining word*, so a bound can be attributed but not
re-checked — already listed as open in `pgx-mcts-bench/HANDOFF.md`. Until that is
fixed, no claim against their table is checkable, and an unverifiable improvement
is not an improvement.

**Certified lower bounds, or the interval never closes.** Their intervals collapse
because KnotInfo supplies the lower bound. Ours would have to compute it (§2.2).

**The calibration.** The bounds ratchet in this project stands at **6 on
`R(3,18)#0`, whose `u` is 2**, and at **11 on `P(3,20)#0` against a theorem of 9**.
On the knots where the answer is known, the current search is well off optimal. The
right order is therefore: close that gap on knots with known answers first, then
point the machinery at their open intervals. Targeting their spreadsheet before
the ratchet is tight would produce loose bounds that nobody can use.

## 4. Jones as a network input: what is actually usable

The question was whether the Jones polynomial — over a Galois field, modulo a
polynomial, modulo a prime — is useful as an input. Three separate answers.

### 4.1 Do not feed the polynomial; feed the tractable evaluations

`V_K(t)` has unbounded degree span and unbounded coefficients, so as an input it is
a variable-length integer vector with no natural scale. Worse, evaluating it is
**#P-hard except at `t = 1, i, e^{2 pi i/3}, -1, e^{i pi /3}`** (Jaeger–Vertigan–Welsh),
which is the same list of tractable points [12 §3](12-serial-formulation.md)
already leans on. At those five points:

| `t` | value | usable? |
|---|---|---|
| `1` | `(-2)^{c-1}` | no — component count only |
| `i` | determined by `c` | no |
| `e^{2 pi i / 3}` | `1` for every link (Lickorish–Millett) | no |
| `-1` | `Delta_K(-1)` = `det(K)` | **yes** |
| `e^{i pi / 3}` | `± i^{c-1} (i sqrt 3)^m`, `m = dim H_1(Sigma_2(K); Z/3)` | **yes** |

So the entire poly-time, fixed-size, non-trivial content of Jones is **the
determinant and the `Z/3` rank of the double branched cover**. That is also the
precise answer to "Jones in a Galois field": the tractable Galois-flavoured part of
Jones *is* a mod-3 homology rank. `determinant` is already implemented here; the
`Z/3` rank is not, and is cheap.

This is worth more than it looks, because both quantities are exactly the sort of
thing that feeds unknotting-number obstructions (§2.2) rather than being merely
correlated with them.

### 4.2 Coefficients modulo `p` are a sound encoding, and are not information-free

Reducing coefficients mod `p` fixes the scale problem: bounded integers with a
natural categorical encoding. And the reduction is far from destroying structure —
Jones polynomials occupy a set of density at most `4/p^7` among polynomials mod `p`
([arXiv:2204.12259](https://arxiv.org/abs/2204.12259)), so a mod-`p` Jones vector
is a highly constrained object.

One implementation warning, because it is the classic error: **`Z/p` must be fed
one-hot, not as a float.** As a number, `1` and `p-1` are far apart; as residues
they are adjacent. A network given residues as floats will learn a metric that does
not exist.

And one scope warning: a mod-`p` feature is a *feature*, not a *certificate*. It
can guide a search; it cannot close an interval. §2.2 is what closes intervals.

### 4.3 The zero-knowledge caveat that already applies here

[10](10-invariants-and-representations.md) and [12 §4](12-serial-formulation.md)
settled this: computed invariants handed to the network are an **oracle arm** —
run them to bound the upside, quote them as oracles, never as the headline. Feeding
Jones is legitimate under that rule and only under it. It is also, notably, what
the DeepMind/Oxford agent did: it fed invariants of the diagram and of its
one-crossing-change neighbours rather than the diagram itself, and found Jones the
most useful feature.

## 5. The part I would actually chase: make the network *be* an invariant

The stated goal is for RL to auto-discover important invariants, with the network
capable in principle of representing them. There is a way to get the "in principle"
for free, and it is stronger than predicting a known invariant.

**The observation.** Every classical polynomial invariant of a closed braid is a
*Markov trace*: pick a representation `rho: B_n -> GL_d(R)`, and read off a
normalised trace of `rho(w)`. Alexander comes from Burau, Jones from
Temperley–Lieb. So an architecture that is *itself* a product of learned per-letter
matrices with a trace readout is in the same class of object.

**The three conditions, and what each buys.**

| impose | on what | what it buys |
|---|---|---|
| the braid relations on the learned matrices | `M_i M_j = M_j M_i` for `\|i-j\| >= 2`; `M_i M_{i+1} M_i = M_{i+1} M_i M_{i+1}` | the output depends only on the **braid group element**, so it is invariant under every tier-1 move — by construction, not by training |
| a trace readout | `w -> tr(rho(w))` | invariance under **conjugation**, i.e. `ROTATE`, the first Markov move |
| the Markov condition | `tr(rho(w sigma_n^{±1})) = z_± tr(rho(w))` for two learned scalars | invariance under **(de)stabilisation**, the second Markov move |

Satisfy all three and **whatever the network learns is a link invariant, provably**.
The human's remaining job is only to *identify which one* — the murmurations
division of labour, but starting from an object that is guaranteed well-defined
rather than one that merely fits.

That is strictly stronger than [10 §1B](10-invariants-and-representations.md)'s
learned 1-Lipschitz potential, whose output is conjectural because Lipschitzness can
only ever be sampled. Here invariance is architectural; only the *usefulness* (and,
if you want a bound, the Lipschitz constant) still needs proving.

**Most of the machinery already exists.** `pgx_mcts_bench.networks.SequenceBraidNet`
has a `finite-field` encoder that learns matrices over `F_p`
(`field_matrices`, `serial_encoder_prime`, `_mod_centered`), an `fsa` encoder that
learns a soft automaton, and a `burau` encoder that is the fixed oracle arm. **What
is missing is the constraints** — nothing currently penalises violating the braid
relation, and nothing reads out a trace. Adding a relation-violation loss and a
trace head is a small change to an existing arm, not a new system.

**And the capacity question has a concrete answer.** [12 §3](12-serial-formulation.md)
worked out that `TL_5` decomposes into irreducibles of dimensions 1, 4 and 5, and
that the 4-dimensional block *is* reduced Burau. So:

* `d = 4` reaches **Alexander**, and cannot reach Jones;
* Jones needs the 5-dimensional block as well, so `d >= 5` at minimum, and the full
  42-dimensional `TL_5` for the whole invariant.

That converts "should the network be able to discover Jones in principle?" into a
sizing decision rather than a hope, and it says the existing carriers were too
small.

**Prior art.** AIDN ([arXiv:2012.01141](https://arxiv.org/abs/2012.01141)) does
exactly the "impose a finite presentation's relations as an optimisation problem"
half, and gestures at Reshetikhin–Turaev for constructing new link invariants. So
the mechanism is not novel. What appears unoccupied is using it **for unknotting
bounds**, with the Markov conditions enforced rather than only the braid relations,
and with the self-play ratchet supplying the training signal.

## 5b. What the bounds did to this project's own ladder

Items 1-3 below are built (`rf_knots.seifert`, `rf_knots.unknot_search`,
`rf_knots.verified_bounds`), and running the branch-and-bound over all 22
non-trivial rungs that had a standing record settles most of the ladder:

| | rungs |
|---|---:|
| `u` **determined exactly** — the sequence found meets a certified lower bound | **17 / 22** |
| standing record **improved** | 4 / 22 |
| no sequence found within the beam and the record's budget | 4 / 22 |

Full table in `artifacts/rung-sweep-20260808/report.md`. The improvements:

| knot | ratchet record | found here | |
|---|---:|---:|---|
| `7_5` (`R(3,18)#0`) | 6 | **2** | `u` determined |
| `7_3` | 3 | **2** | `u` determined |
| `3_1` on five strands | 2 | **1** | `u` determined |
| `6_3` | 2 | **1** | optimal, but no certified bound reaches 1 |

Three things worth taking from this.

**The reference the whole ladder is scored against was loose, and nobody could
have seen it.** `R(3,18)#0` stood at 6 against a true `u` of 2 — a factor of
three — and the record carried no witness, so there was nothing to check. Every
"gap to reference" number computed against that rung was measuring the reference's
error, not the agent's.

**On the labelled rungs the agents were already optimal.** Every `T(2,q)` torus
rung comes back at exactly its record, and now with a certificate rather than a
comparison against the Milnor conjecture. The trained agents are good; the ratchet
was not.

**Validation, since a search that reports its own optimality needs checking.**
Against the bundled unknotting numbers, on every rung whose `u` is tabulated: no
certified bound exceeds the truth, no sequence found is shorter than the truth, and
every sequence found *equals* it. Zero violations in 22.

### The 20-letter rung, and why branch-and-bound was the wrong tool for it

The hardest of the four blanks was a 20-letter three-strand **positive braid**:
genus 9, certified bound 9, ratchet record 11. It is now **9, determined** —
`u = 9`, with a 27-move witness that replays.

Branch-and-bound could not do it, and the reason is worth recording. Nine levels
deep it pruned *nothing*: every candidate's bound was inside the budget, so the
search paid for a Seifert matrix and a knot Floer homology at every distinct knot
on every frontier and never finished. The bound is what makes the answer a
theorem; it is not always what should be in the loop.

For a positive braid it need not be. `u = g = (L - n + 1)/2`, and every useful
crossing change turns an adjacent `sigma_i sigma_i` into `sigma_i sigma_i^-1`,
which free-reduces and drops the length by exactly two. **Length is the signal**,
and the certified bound is needed exactly twice: once at the start to know what
optimal is, once at the end to say whether the sequence met it.

Greedy descent on length is not enough — it reached two letters after nine changes
and still needed a tenth, having landed on the wrong two-letter word. Keeping sixty
candidates per level (`research/experiments/beam_rung.py`) finds the one that ends
on a two-letter word that destabilises away: **9 changes in 32 seconds**.

Note which bound did the work. The signature gives only `|sigma|/2 = 7` here;
**`tau = 9` is what makes 9 certified**, so this rung is precisely the case that
would have stayed open without the knot Floer bound added in §5b.

That leaves three rungs with no improvement found, all three-strand words where the
record already equals the certified bound — so there is nothing to improve unless
the bound itself is loose.

## 6. Ordered next steps

1. ~~**Store the unknotting sequence as the witness.**~~ **Done** —
   `rf_knots.verified_bounds` replays every claim on read.
2. ~~**Certified lower bounds beyond `|sigma|/2`.**~~ **Done for `tau` and the
   Montesinos obstruction**, and `|sigma|/2` now actually runs. Rasmussen `s`
   remains: it needs Khovanov homology, and there is no offline backend here.
3. ~~**Close the calibration gap.**~~ **Done** — §5b. `R(3,18)#0` went 6 -> 2,
   and 17 of 22 rungs now have `u` determined exactly.
4. **A per-crossing flip policy** over the raster head of
   [18](18-raster-representation.md), replacing their exhaustive flip loop.
5. **Depth-`k` crossing-change search**, with the ratchet as the value of an
   intermediate knot.
6. **The constrained matrix accumulator** of §5: `d >= 5` over `F_p`, braid
   relations and Markov conditions as losses, trace readout. Highest ceiling,
   lowest certainty, and the only item on this list that could produce new
   mathematics rather than new bounds.
7. **Then**, and only then, target the open intervals in `unknotting.xlsx`.

Items 1–3 are prerequisites, 4–5 are throughput, 6 is the research bet.
