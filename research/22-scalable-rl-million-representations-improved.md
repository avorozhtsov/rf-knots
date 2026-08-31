# 22 — Witness generation at depth five: an instance-driven revision of note 21

> **Relationship to note 21.** Note 21 (`21-scalable-rl-million-representations.md`,
> an untracked draft in the working copy) is an *architecture and data-hygiene*
> plan. Its outcome taxonomy A/B/C/D, its refusal to treat a budget-censored
> failure as a mathematical negative, and its set-valued proof-graph targets are
> the best things written about this project's training contract and are kept
> here in full. What it lacks is the other half: a **target instance**, a
> **search that can reach it**, and an **objective that does not silently prefer
> the wrong answer**. This note adds those and revises the parts of 21 that break
> when they meet the target.
>
> **Stated goal.** A network that, driven by MCTS, emits replay-verified
> unknotting witnesses with high probability on both easy knots and hard ones —
> the named hard case being `7_1 # m(7_1)`, whose published upper bound is `u <= 5`.
> **Next goal.** Gordian-distance witnesses from a two-representation network.
> §7 co-designs the second goal into the first rather than deferring it.

## Notation

Terms shared with the rest of the project are in
[00-glossary.md](00-glossary.md). These are the symbols this note leans on, and
several are used before the section that introduces them.

| symbol | meaning |
|---|---|
| `u(K)` | unknotting number: the fewest crossing changes taking `K` to the unknot |
| `d(A, B)` | Gordian distance: the fewest crossing changes taking `A` to `B`, so `u(K) = d(K, 0_1)` |
| `k` | **number of crossing changes.** Generating: how many the backward walk makes (§4.1), hence the certified `u <= k`. Searching: the depth of the outer crossing-change search (§3.1). Same quantity, run backwards or forwards. |
| `l` | length of the type-preserving (zero-CC) isotopy walk *between* two consecutive crossing changes (§4.1) |
| `R`, `R'` | a seed representation, and the representation generated from it (§4.1b) |
| `L(K)` | a **certified** lower bound on `u(K)` — a theorem, never a table row (§1.2) |
| `cap` | the strict crossing-change budget a search is conditioned on; the target's is 5 (§1.4) |
| `dose` | the compute budget of one search run — simulations, or beam width (§9) |
| `A`, `M` | objective weight and internal move cap in `A*cc + moves`; M0 recommends `A = 100_000`, `M = 10_000` (§2.1) |
| `recall@k` | **unrelated to the `k` above** — the top-`k` of a ranked list, used only for the site head and retrieval (§3.4, M1) |

## The milestones at a glance

`M0`–`M6` label the stages of work. They are defined in full, with gates and
kill criteria, in **[§8](#8-milestones-gates-kill-criteria)**; this is the map,
because the sections above refer to them before §8 arrives.

| id | stage | one line | status |
|---|---|---|---|
| **M0** | Measure the floor | replay the published witnesses, bound the chain, measure the untrained search and the throughput — no training | **done**, §Status |
| M0.1 | — replay | do the published witnesses replay in this move set? | done — **no** |
| M0.2 | — certified bounds | what does the certified machinery say about each chain state? | done |
| M0.3 | — untrained floor | what does today's search do at cap 6 and cap 5? | done — `0.00` at cap 5 |
| M0.4 | — throughput | what does an evaluation cost, and what does that make affordable? | done |
| **M1** | The per-site head | rank crossing-change *and* growth sites; supervised, gateable before any RL | not started — feasible |
| **M2** | Depth-`k` outer search | the two-level search of §3, with identification as terminal test | not started |
| **M3** | Generator at scale | `10^5` then `10^6` instances with certified difficulty (§4.1, §4.1b) | not started — `10^5` feasible, `10^6` needs a record-format change |
| **M4** | Joint training | monotone survival head, GRPO with stabilisers, set-valued targets (§5) | not started — feasible at entry scale only |
| **M5** | The target | a replay-verified 5-change witness for `7_1 # m(7_1)` | **the point of the plan** |
| **M6** | Gordian | the two-representation network and its witnesses (§7) | after M5, or parallel once M3 lands |

Milestones are not strictly sequential: M6's corpus accrues as a by-product of
M3 (§7.4), and M1 gates on its own supervised metric without waiting for M2.

---

---

## Status — M0 results: the kill criterion fired, in one of its two senses

M0 has been run in full. Artifacts in `artifacts/m0/`. Status by bullet:

| M0 stage | verdict | where |
|---|---|---|
| **M0.1** replay both witnesses | **kill criterion fired** — the certificate is not portable | §Status, §2.1 |
| **M0.2** certified bounds on the chain | done; the draft's successor interval was **wrong** | §1.2, §1.3 |
| **M0.3** the untrained floor | `witness_rate@(cap=5) = 0.00` at every dose | §Status |
| **M0.4** throughput and projection | M3 `10^6` and M4 at cube scale **not feasible as stated** | §6.3, §8 |

Downstream feasibility, measured rather than assumed: **M1 feasible**, **M3 at
`10^5` feasible**, **M3 at `10^6` not feasible as stated**, **M4 feasible at the
entry scale and not at cube scale**. Three separate measurements independently
converge on the same instruction — work from the 20-letter word.

The decisive measurement:

| | measured |
|---|---:|
| words that parse as `rf_knots.evidence.BraidState` | **all** — including the 254-letter 21-strand one, 1 component |
| action spec needed | `(max_len 512, max_strands 24)`, 25,604 actions |
| marked crossing changes expressible as `CROSSING_CHANGE` | **7 of 7** |
| moves of the 518-move Wang–Zhang trace that translate | **2** |
| RII bigons realisable in any order the word accepts | 1 of 129 (112 blocked by non-commuting letters) |
| crossing-*creating* trace moves inside rf-knots' move set | **0 of 16** |
| cube reduction after all 5 changes, 20-minute beam | 254 → **250** letters, from 82.3M states |
| B5 reduction after its 2 changes, 5-minute beam | 20 → **14** letters, replay-verified |

**What this does and does not establish.** It establishes decisively that the
published certificate is **not portable** into this project's move set. The
reason is structural, not a tuning failure: after the first RII the object is a
252-crossing 4-valent diagram with **no braid axis**, and no braid word for it is
recorded anywhere in the corpus. Any plan that expected to import the published
bound and replay it is dead, and §8's M0 kill criterion fires in that sense.

It does **not** establish that the target is out of reach. Markov's theorem
guarantees a braid-word path exists; what failed is a *length-ordered beam*,
which by construction will never grow a diagram — and §2.3 already says the
witness needs the diagram to grow before the third change is available. The
82.3M-state result is evidence about that search, not about the move set.

The B5 row is the encouraging one and was nearly buried: 20 → 14 letters is
exactly `14a_18636`'s crossing number, so the move set does reach a
minimal-crossing diagram, and it correctly fails to reach the empty braid
because that state still has `u <= 3`. At small scale the machinery works.

**The floor itself, measured (M0.3).** Two algorithms — the shipped
branch-and-bound and a seeded beam over word length — 8 seeds each, five compute
doses, both caps, 80 beam runs in 33 minutes:

| dose (beam width) | 8 | 32 | 128 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|
| `witness_rate@(cap=5, 8 seeds)` | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| `witness_rate@(cap=6, 8 seeds)` | 0.75 | 0.62 | 0.50 | 0.62 | 1.00 |

`witness_rate@(cap=5) = 0.00` at every dose, every seed, both algorithms. §9 asks
for `>= 0.5` before the word "mastery" is used; the entire interval is open.
28 claimed witnesses, **28 verified** — each replayed twice, once at return and
once round-tripped through JSON. The cap-6 row is deliberately kept separate and
is **not evidence of anything**: subadditivity gives 6 for free.

**Branch-and-bound found nothing at either cap**, hitting the wall clock both
times, with `pruned_by_bound = 0` at every level transition. §1.3's prediction is
verified. The arithmetic makes it inevitable: a branch is cut when
`used + 1 + bound > cap`, and one change in the bound is 0–2, so at `used = 0`
the test would need `bound > 4`.

> **And the bound is not what makes it intractable — the fingerprint's cost is.**
> Certified-bound computation took 1.32 s of 646 s (0.2%). The cost is
> `_fingerprint` calling `invariants.determinant` at ~80 ms per candidate against
> the ~10 ms its own docstring assumes; level 0 spent **199 s** on it. This is the
> same defect §3.1 flags from the other direction, now measured at 20 crossings
> rather than 254, and it is the single cheapest speedup available anywhere in
> the search.
>
> **Correction — the cost is the defect; the *quality* is not.** An earlier draft
> of this note read "199 s turning ~2,800 flips into 6 distinct fingerprints" as
> a collision disaster, and said a key mapping 2,800 states to 6 classes was
> "erasing knots rather than identifying them". That was wrong, and a dedicated
> experiment refuted it: scored against an independent referee (Jones plus
> `seifert.signature`), those 2,800 states **genuinely are 6 knots up to mirror**
> — the determinant's partition is identical to the true one, with 0 classes
> absorbed, 0 false merges and 0 false splits. `_explore_level` returns 140
> diagrams that are all the same knot, so a handful of classes is what the
> structure permits. The fingerprint was finding knots, not erasing them. Fix its
> speed; do not "fix" its discrimination.

**Consequences, which reorder the plan:**

1. **The 254-letter cube is not an entry point.** It is a quarantined final exam
   and an identification target, nothing more. Every campaign starts from the
   20-letter B5 word.
2. **§11 question 1 is promoted from curiosity to the central open problem.**
   The smallest diagram on which a 5-change witness exists now gates everything,
   because the published answer cannot be borrowed.
3. **A length-monotone search is disqualified** for anything past the second
   change. Controlled inflation is a first-class requirement, not a risk note.
4. M1–M5 are **not** premature — but "reproduce the published witness" is struck
   from M0's deliverables and replaced by "construct an independent one".
5. **Fix the fingerprint before anything else.** It is one import, it dominates
   the only search this project ships, and every later milestone runs on top of
   it. Nothing in M1–M5 should start until it is measured again afterwards.
6. **Change the record format, do not merely compress it.** At M3's stated shape
   a raw record is 844.7 kB, so `10^6` is **844 GB** — impossible — and 14.0 GB
   gzipped, which is 42% of the 31 GiB free on a 93%-full volume. Storing
   `(start word, action list)` and replaying on read is **2.24 kB** per record,
   **2.24 GB** total. Replay is already the trust boundary everywhere else in
   this project; making it the storage format too costs nothing and is the only
   version of M3 that fits.

---

## 0. What this note changes

| # | Note 21 says | This note says | Why |
|---|---|---|---|
| 1 | no target instance | the target is measured and pinned (§1) | it changes almost every downstream decision |
| 2 | objective is `L1000 = 1000*cc + moves` | `L1000` **prefers 6 crossing changes to 5** at target scale; use lexicographic `(cc, moves)` (§2.1) | the one recorded zero-CC leg is 518 moves, i.e. 1,670 braid moves against a 1,033 threshold |
| 3 | certified bounds prune the search | the floor is nearly **constant** (2,2,1,1,0) along the route, so it gives no ordering signal (§1.3, §2.2) | measured; `pruned_by_bound = 0` at every level transition |
| 4 | one flat MCTS over all moves | **two levels**: outer best-first over the crossing-change graph, inner MCTS over zero-CC surgery (§3) | the outer level has transpositions, known labels and a branching factor the network can rank |
| 5 | "scale is the problem" | at the entry representation the target is **5 strands, 20 letters** — capacity is not the blocker (§1.1) | de-risks and re-prioritises the whole programme |
| 6 | augmentation from existing corpora | **interleaved backward generator**: alternate scramble and crossing change from the unknot (§4.1) | produces the exact shape the target needs, with a witness by construction, at any length |
| 7 | scalar conditional `p_solve` | **monotone survival head over the whole CC budget vector** (§5.1) | one rollout supervises every budget; monotonicity architectural, not penalised |
| 8 | tournament `+1` vs `-1/9` on log-likelihood | that objective **is** GRPO and is unbounded below on losers; adopt GRPO's stabilisers (§5.2) | long trajectories make raw weighted log-likelihood diverge |
| 9 | two globals `E_knot` / `G_layout` | **three**, named as in the shipped `ScalableBraidEmbeddingEncoder` (§6.1) | `knot` / `isotopy` / `layout` already exist in code |
| 10 | embedding evaluated by ablation | give it an **operational job**: approximate retrieval of a proof-graph node, measured by recall@k (§3.4) | converts "is it useful" into a number the search reports |
| 11 | single-input network, pair network later | **build the pair network first**; unknotting is the case `d(K, 0_1)` (§7) | avoids a second architecture migration; second input is a cached constant |
| 12 | identification unspecified | Jones is off the hot path at 254 crossings; the fingerprint is the **polynomial-time subset** (§3.1) | measured cost, and note 19's 73% detection trap |
| 13 | "10–100M evaluations" | a preflight throughput projection is a **gate**, on a single-host budget (§8, M0) | `docs/lessons.md`; the offload box is retired |
| 14 | generic leakage groups | the target and its cube are a **named final exam** with an explicit quarantine (§4.3) | the answer is in the repository |
| 15 | success = "solves the knot" | success = **replay-verified witness at the strict cap**, `witness_rate@(cap, dose, seeds)` (§9) | `u <= 6` is free from subadditivity and is not a result |

Everything in note 21 not listed above is carried forward unchanged. In
particular: the four counters, the A/B/C/D outcome classes, "D is never a
failure label", set-valued proof-graph targets, and the rule that a 50:50
minibatch never touches calibration reporting.

---

## 1. The target, measured

### 1.1 Two representations of the same instance

Both live in `unknotdb` and both are pinned artifacts, not literature claims.

| source | strands | letters | crossing changes | note |
|---|---:|---:|---:|---|
| Brittenham–Hermiller B5 word (`tools/audit_brittenham_b5.py`) | 5 | 20 | 2 in this chart, 3 only after reshaping | the entry point |
| Wang–Zhang cube (`outputs/unknotdb-wang-zhang-five-crossing-cube-v0.json`) | 21 | 254 | all 5 in one chart, positions `111,176,192,241,251` | the certificate |

The published chain (`outputs/brittenham-hermiller-k14-k15-k12-chain-v0.json`)
descends through named knots with decreasing upper bounds:

```text
7_1 # m(7_1)  --2 CC-->  14a_18636 (u<=3)  --CC-->  15n_81556 (u<=2)
              --CC-->    12n_412 (u<=1)    --CC-->  unknot
```

That chain is four sub-targets of increasing difficulty with known answers, and
they compose into the headline result. It is the curriculum, not just the proof.

**The single most important structural fact** is recorded in the audit tool: the
tempting shortcut in which all five changes are made inside the 20-letter B5
chart is **false** — all `C(18,3)` completions were enumerated and none reaches a
determinant-1 state. The witness *requires* interleaving crossing changes with
diagram-changing isotopy. Any pipeline of the form "inflate once, flip crossings,
simplify" cannot reach this instance by construction. That is the whole
difficulty, and it is the thing the network has to learn.

### 1.2 What the implemented machinery already says

Measured in this worktree with `rf_knots.invariants`, on the 20-letter B5 word
and on its two-crossing-change successor (0.3 s each):

**M0 ran this properly across both representations and all four chain states.**
The draft's two-row table was measured through `rf_knots.invariants`, which
under-reports (see the trap below); the corrected table is:

| state | knot | repr | det | `sigma` | `tau` | `H_1(Sigma_2)` cyclic | **`u >=`** | `u <=` |
|---|---|---|---:|---:|---:|:--:|:--:|:--:|
| `b5-root` | `7_1 # m(7_1)` | B5 20 | 49 | 0 | 0 | no | **2** | 5 |
| `cube-00` | `7_1 # m(7_1)` | cube 254 | 49 | 0 | 0 | no | **2** | 5 |
| `b5-cc2` | `14a_18636` | B5 20 | 261 | 0 | 0 | no | **2** | 3 |
| `cube-06` | `14a_18636` | cube 254 | 261 | 0 | 0 | no | **2** | 3 |
| `cube-16` | `15n_81556` | cube 254 | 39 | 2 | 0 | yes | **1** | 2 |
| `cube-1e` | `12n_412` | cube 254 | 75 | 2 | -1 | yes | **1** | 1 — **exact** |
| `cube-1f` | unknot | cube 254 | 1 | 0 | — | yes | **0** | 0 |

Four readings, all consequential.

* **The decomposition works.** `connected_summands` splits both the 20- and the
  254-letter root into `s1^7` and `s1^-7`. Subadditivity hands us `u <= 6` free.
* **`K # m(K)` is slice**, so `g_4 = 0` and `tau = s = 0`. Every 4-genus bound
  vanishes; the `u >= 2` is the non-cyclic `H_1(Sigma_2)` obstruction on
  `Z/7 (+) Z/7`.
* **The representations agree**, which is the check that matters: det, `|sigma|`,
  `|tau|` and the cyclic verdict match across the 20-letter word, the 254-letter
  word, and links rebuilt from the chain's own DT codes. Sign flips at masks
  `16`/`1e` are the mirror. `simplify` collapses the cube states to 14/14/15/12/0
  crossings, matching the named knots' crossing numbers.
* **The successor floor is 2, not 0 — the draft was wrong.** `14a_18636` has
  `H_1(Sigma_2) = Z/3 (+) Z/87`, non-cyclic, so Montesinos certifies `u >= 2` and
  the interval is `[2, 3]`, not `[0, 3]`.

> **The trap that produced the wrong number, worth its own line in
> `docs/lessons.md`.** `Invariants.unknotting_lower` is `abs(sigma) // 2`, raised
> to 2 for composites by Scharlemann. It never consults Montesinos or `tau`.
> `rf_knots.lower_bounds.computed_claims` does. Two code paths, similar names,
> different strengths — and the weaker one silently returns the smaller certified
> bound. The root's `2` came from Scharlemann, not from the obstruction the draft
> credited. **Always take the bound from `computed_claims`.**

The fingerprint of the composite is also *ambiguous*: the invariant match reports
`7_1 or 12n_749` for each factor. That is `docs/lessons.md`'s identification trap
arriving on the actual target, not hypothetically.

### 1.3 Consequence: there is no usable admissible heuristic

`rf_knots.unknot_search` prunes with `max(|sigma|/2, |tau|, Montesinos)`.

> **M0: premise refuted, conclusion survives.** The draft claimed the floor is
> "identically zero along the published route". It is not. Measured, it runs
> **2, 2, 1, 1, 0**, so `f = spent + floor` runs **2, 4, 4, 5, 5**. But five
> states take only **three distinct values**, so the floor carries almost no
> *ordering* signal — which is precisely what a best-first priority needs. At
> cap 5 nothing is pruned, correctly, since the route is a valid witness; at
> cap 4 it does prune two of the five. "There is no usable admissible heuristic
> here" therefore stands, for a better reason than the draft gave: not that the
> bound is zero, but that it is nearly constant.

`docs/lessons.md` records the cost side: on a 20-letter positive braid,
branch-and-bound "pruned nothing while paying for a Seifert matrix and a knot
Floer homology at every frontier knot".

**This is the argument for the network, stated precisely.** The search needs a
heuristic; mathematics does not supply one for slice composites; so the heuristic
must be learned, and the certified bound is demoted to what `docs/lessons.md`
says it should be — computed twice, once to know what optimal would be and once
to certify that a found sequence met it, never in the inner loop.

### 1.4 The free baseline is 6, and beating it is the entire result

Subadditivity plus two `T(2,7)` witnesses gives a 6-change witness cheaply, and
`connected_summands` makes that shortcut trivially discoverable. A run reporting
"solved `7_1 # m(7_1)`" has therefore reported nothing.

* The campaign objective is a **replay-verified 5-change witness**.
* The search is conditioned on a **strict cap of 5** from the first node.
* A summand-wise solution is recorded as the incumbent to beat, never as success.

**Scope of the claim, honestly.** A 5-change witness found here would be an
independent, replayable re-derivation of a published upper bound — evidence that
the machinery works at depth five, and not new mathematics. What *would* be new
is a 4-change witness: nothing implemented in this repository rules one out,
since the certified interval is `[2, 5]`. Before treating 4 as open, read
Brittenham–Hermiller for a lower bound stronger than the ones computed here; do
not infer openness from the absence of a bound in our code.

---

## 2. Three defects in the current contract that would sink the target

### 2.1 The objective prefers the wrong answer

The active contract is `L_AB = A*cc + B*moves`, with `L1000` meaning
`A/B = 1000`. Compare two candidate witnesses:

```text
5 crossing changes, 2000 semantic moves  ->  L1000 = 7000
6 crossing changes,  500 semantic moves  ->  L1000 = 6500   <-- wins
```

This is not hypothetical at target scale. The zero-CC leg of the Wang–Zhang
certificate is **518 Reidemeister moves** on a 254-crossing diagram, of which 516
change the diagram.

> **M0 correction.** An earlier draft called this "one of at least six legs".
> That was wrong, and the artifacts say so: the cube makes all five changes in a
> *single* chart, so the corpus records exactly **one** isotopy leg, plus one
> *unrecorded* leg inflating the 20-letter entry word to the 254-letter chart.
> M0 measured that unrecorded leg at a certified floor of **117** semantic moves.

M0 also measured the translation cost, which the draft could only guess at: 516
diagram-changing moves cost **1,548** braid-semantic moves at the measured 3x
blow-up, and **7,639** once the transport needed to bring letters adjacent is
charged (mean cyclic word distance 51.8, max 127, 6,091 total `COMMUTE` moves).
With the inflation leg the totals are **638 / 1,670 / 7,761**.

> **The inversion is now proved, from two independent measurements.** M0's search
> sweep found a shortest replay-verified 6-change witness at **(6, 33)** — so
> `L1000 = 6033`, and any 5-change witness loses whenever it exceeds **1,033**
> semantic moves. M0's replay job independently measured the target's single
> recorded isotopy leg at **1,670 moves at the measured blow-up**, before the
> other four changes are paid for. `1670 > 1033`. The draft projected this
> inversion; M0 closed it with numbers neither job could have produced alone.
> Under `L10` the margin is worse still — a 5-change witness loses past 43 moves.

**Fix.** Compare `(cc, moves)` lexicographically. Where a scalar is required for
a value head or a heap key, set `A > M` with `M` the internal move cap, and
record `A` in every artifact. **M0's measured recommendation: `M = 10_000`,
`A = 100_000`.** Report `L10`/`L1000` alongside for continuity with
note 20's evidence index, and state plainly that numbers under different `A` are
not comparable — the rule the archive already applies to the pre-semantic-move
correction.

### 2.2 The certified-value clamp is vacuous here

`rf_knots.certified_value` clamps a value head by the certified floor. Keep it —
it is free correctness and it matters on the torus rungs — but do not let the
search design assume it carries information. On this target it clamps at 2, then
at 0.

### 2.3 Capacity is not the blocker; the action space is

The entry representation is 5 strands and 20 letters, inside `TIER1`
(`max_len=64`, `max_strands=8`). Reshaping will inflate it, and the published
certificate reaches 21 strands and 254 letters — though that is one particular
chart, not a lower bound on what is needed.

What actually blocks the target is not tensor size but the **fixed action
space**. The README lists "a per-cell action space, so the policy head stops
depending on the strand count at all" as open, with
`research/experiments/strand_shared_head.py` removing the dependence from the
parameters only. Until a per-site policy lands, a checkpoint cannot be evaluated
on a wider braid at all, and the outer search cannot rank 254 crossings.

**The per-crossing / per-site policy head is therefore a hard prerequisite, not
a later stage.** It is also, independently, the highest-leverage head in the
system (§5.3). Add `TIER2 = BraidConfig(max_len=512, max_strands=24, ...)` only
after it exists; adding it first buys nothing.

> **M0: the conclusion is right and the stated reason is wrong.** The action
> space at `(512, 24)` — 25,604 actions — is *cheap*: jitted legality mask 35 us,
> scalar step 110 us, vmap-64 step 24 us per state, and a flat 25,604-way head at
> 103,721 evaluations/s on MPS. Against a 219 us encoder forward at `(256, 21)`
> that is **13% of the per-state cost**, giving ~4,100 sims/s on MPS. Size is not
> the problem.
>
> Three things actually block, in increasing order of importance:
> 1. **Checkpoint transfer**, measured: loading a TIER1 head at TIER2 fails with
>    `size mismatch [1156,96] vs [25604,96]`. This is the half `strand_shared_head`
>    was written to fix, and it is still open.
> 2. **Search memory**: 307 kB of dense per-action statistics per node, so
>    3,255 nodes per GB.
> 3. **The branching factor** — and this is the one that matters. The 254-letter
>    cube has **10,598 legal successors, of which 10,160 (95.9%) are `INSERT`**.
>    The search runs at 39 nodes/s not because actions are expensive but because
>    almost every action adds a cancelling pair.
>
> Point 3 reframes §5.3. The per-site head's first job is not ranking crossing
> changes — it is **deciding where the diagram may grow**. M0.1 already
> disqualified length-monotone search, so `INSERT` cannot simply be banned; it has
> to be *ranked*. Growth is the branching factor, and the branching factor is the
> search.

---

## 3. The search: two levels, not one

Note 21 folds everything into one MCTS over the union of isotopy moves and
crossing changes. That discards the structure the problem has: a witness is five
crossing-change *events* separated by long isotopy runs, and the two levels have
completely different statistics.

| | outer level | inner level |
|---|---|---|
| state | knot type (fingerprint) | concrete diagram |
| edge | one crossing change | one zero-CC semantic move |
| depth | <= 5 | 10^2 – 10^3 |
| ground truth | knot identity, tabulated/witnessed `u` | none |
| transpositions | many, and cheap to exploit | few |
| branching | number of crossings (20 – 254) | small, local |

### 3.1 Outer: best-first over the crossing-change graph

This is note 19 §2.1's depth-`k` generalisation and note 20's heap, made the
primary loop.

* **The node key is the knot, not the diagram** — note 10, Consequence A. Two
  branches reaching the same type are one node.
* **Priority** `f = spent_cc + h(K)`, with `h` the learned remaining-change
  estimate clamped below by the certified floor. `h` is *not* admissible, so this
  is best-first/beam search, not `A*`, and it proves upper bounds only. Say so in
  every artifact.
* **The terminal test** is the generalised DKT step 5: at spent `s`, stop when
  `s + u_upper(K_s) <= cap`. Two grades of terminal, reported separately:
  * **replay-complete** — `u_upper(K_s)` is backed by a program in `unknotdb`
    that replays, so the composed witness replays end to end;
  * **catalogue-composed** — `u_upper(K_s)` is a KnotInfo claim with no program.
    That is a bound with an incomplete witness and must never be filed as one.
    It is exactly the trust boundary `unknotdb` already draws.
* **The fingerprint must stay cheap — but measure which part is expensive.**
  An earlier draft asserted that the Kauffman bracket is the costly piece and
  that "254 crossings are hopeless". **M0 refuted that for this implementation.**
  `rf_knots.invariants.kauffman_bracket` is a Temperley–Lieb state sum:
  exponential in the *strand count* (Catalan `n`), but **linear in word length**
  — 0.8 ms on a 40-letter 5-strand word. The expensive one is the **Alexander
  polynomial**, a Burau determinant, at ~200 ms on the same word. So the axis to
  fear is strands, not letters, and the cheap route to identification is to
  fingerprint the **free reduction** (same group element, same knot), measured at
  38 ms per instance. `tau` remains a confirmation step on a shortlist.
* **The expensive thing is the route, not the invariant.** Measured at 254
  crossings: `seifert.determinant` **0.42 s**, `seifert.signature` 0.30 s,
  Seifert matrix 0.035 s, `simplify("global")` 0.02 s, mod-`p` corank 0.008 s per
  prime, and `seifert.tau` 0.012–0.049 s — the last only because `simplify`
  collapses 254 crossings to 12–15 first, so `tau` still needs a timeout on a
  genuinely large knot. Against that, `invariants.determinant` (the Burau route)
  takes **647 s** to return *the same 49*. That is **1500x** for an identical
  answer.
* **Consequence, and it is a live defect.** `unknot_search._fingerprint` imports
  `invariants.determinant`. **There is no crossover** — M0 measured 8 shapes and
  the Seifert route wins at every one, with agreeing values:

  | letters / strands | `seifert` | Burau | ratio |
  |---|---:|---:|---:|
  | 20 / 5 | 0.0017 s | 0.0425 s | 25x |
  | 100 / 5 | 0.0353 s | 4.70 s | 133x |
  | 100 / 21 | 0.0237 s | 7.87 s | 332x |
  | 254 / 21 | 0.4205 s | 647.16 s | **1,539x** |

  In search terms: one level on the cube costs **45.7 hours** by the Burau route
  and **107 s** by the Seifert one. The docstring's premise is also false —
  `invariants.determinant` costs the same as `alexander_polynomial` (79.7 vs
  82.3 ms); it *is* the polynomial, not "one Burau evaluation". **Caveat for the
  fix:** `seifert.determinant` raised on 3 of 12 synthetic split-link words (0 of
  60 real instances), so this needs a guarded fallback, not a bare swap.
* **Identification is now solved, and it was the bottleneck.** SnapPy is
  installed (`pip install -e '.[identify]'`) and its census lookup names
  `K14a18636` — the first state on the published chain, and one the bundled
  ≤12-crossing table returns `None` for — in **0.02 s**, faster than the 0.42 s
  Seifert determinant. Note 19 §2.2 called identification the bottleneck of the
  DKT pipeline, at 73% detection inside a 3–13 crossing trust window; a census
  lookup is strictly stronger than Jones matching and extends past 13.
  Two limits to state rather than discover: SnapPy's own `determinant` and
  `signature` raise without Sage, so bounds still come from `rf_knots.seifert`;
  and `identify()` correctly returns `[]` on the composite root, since the census
  is of prime knots and a connected sum is not hyperbolic. Composites go through
  `connected_summands` first — which already splits this target unaided.
* **The key should be `(determinant, |signature|)` — and not hyperbolic volume.**
  Measured one crossing change deeper than the level-0 frontier, on 2,525 states
  spanning 36 knots, against an up-to-mirror referee:

  | key | classes | absorbed | false-merge | false-split | ms/state |
  |---|---:|---:|---:|---:|---:|
  | determinant (status quo) | 34 | 2 | 0.0439 | 0.0000 | 1.4 |
  | **`(det, abs sig)`** | 35 | 1 | **0.0001** | 0.0000 | **2.9** |
  | `(volume, det, abs sig)` | 35 | 1 | 0.0001 | 0.0000 | 15.8 |
  | volume, gated on solution type | 24 | 12 | **0.7600** | 0.0000 | 13.0 |
  | volume, naive rounding | 192–203 | 4 | 0.0122 | **0.5811** | 13.0 |

  Adding `|signature|` costs 1.5 ms per state and removes **99.8% of the false
  merges** (12,920 pairs down to 24), with zero false splits, exactly and without
  floats. In a tuple **volume is a measured no-op** — `(volume, det, abs sig)`
  scores byte-identically to `(det, abs sig)` at 5.4x the cost, because where
  volume works it separates exactly what the determinant already separated.
* **Why volume fails here, recorded so it is not retried.** Connected sums are
  **80%** of the level-0 frontier and 59% of level 1 — non-hyperbolic inputs are
  the majority, not an edge case, and §4.1b's torus-knot seeds are non-hyperbolic
  too. Worse, **`volume()` does not raise on them, it lies quietly**: 307
  degenerate states returned plausible values (4.5921, 6.4435), a `T(2,7)`
  summand returned **-1.06e-7**, and the same word varied by up to 1.7e-4 between
  calls — so an identical frontier produced 192 classes on one run and 203 on the
  next. A key that changes between runs is not a key. The summand-wise fallback
  (decompose, per-factor volume, `(det, abs sig)` for non-hyperbolic factors) is
  the only key that reproduces the referee partition perfectly, but at 16 ms per
  state it belongs on a shortlist, never on every candidate flip.
* **Smith normal form is the one real casualty.** `branched_cover_homology` did
  not return in 600 s on any 254-letter state (153 MB of resident integers).
  Note 19 §4.1's mod-`p` corank replaces it: now built, and validated 7/7 against
  known `H_1`, including the discriminating pair `T(2,9)` (det 9, cyclic) versus
  the granny knot (det 9, `Z/3 (+) Z/3`).
  Note 19 §4.1's tractable evaluations — `det` and the `Z/3` rank of the double
  branched cover — are the right hot-path pair, and the `Z/3` rank is still
  unbuilt and cheap.

### 3.2 Inner: zero-CC MCTS

Between two crossing changes the task is pure diagram surgery, and it has the
property the whole project was founded on: **instances can be generated from the
answer**. The inner policy's target is not "reach the unknot" but "reach a
diagram on which a *useful* crossing change is available", so its value head is
`p(a scoring site appears within the internal budget)`.

Specifics that matter and that note 21 omits:

* the move graph is reversible, so the tree is a DAG with cycles; keep a visited
  hash and a potential-based shaping term on word length (note 10, proposal 3 —
  raw state only, so it stays inside the zero-knowledge constraint);
* at the low simulation counts this project can afford, Gumbel/sequential-halving
  root selection is more sample-efficient than PUCT and should be measured
  against it at equal evaluations;
* the inner transposition table is keyed on the exact canonical word, not on the
  fingerprint — inner states are diagrams.

### 3.3 Meet in the middle, and the reusable ball around the unknot

Depth five from one side is expensive; two depth-two/three frontiers are not. The
frontier grown from the unknot side is **the set of knot types with small known
`u`**, and `unknotdb` already stores it with `u_upper_bound` per node. So the
meet-in-the-middle table is not a new artifact — it is the graph, indexed by
polynomial-time fingerprint.

That index is a one-time cost amortised over every future target, including every
one of note 20's 482 open intervals, and it is the concrete reason to keep
`unknotdb` in the loop rather than treating it as a data dump.

### 3.4 What the network is actually for

Four jobs, each with its own gate. This replaces "the network gives policy and
value" with something measurable.

| job | head | gate |
|---|---|---|
| rank crossing-change sites | per-site policy (§5.3) | recall@10 of a witness-realising site on held-out instances |
| reshape a diagram toward a useful site | inner policy | solve rate on backward-generated instances at fixed evaluations |
| decide whether a node is worth expanding | monotone survival head (§5.1) | calibration on the natural distribution |
| find the node again in the proof graph | `knot` embedding | retrieval recall@k against the exact fingerprint, and outer-search speedup at equal evaluations |

The last row is the honest version of note 21's embedding ablation. An embedding
that raises retrieval recall and shrinks the number of expansions to a terminal
has earned its place; one that only produces a pretty space has not.

---

## 4. Data: generate from the answer, interleaved

### 4.1 The interleaved backward generator

The central addition of this note. Note 21 lists seven sources of negatives and
no source of *hard positives*; this is the source of hard positives, and it costs
nothing but CPU.

```text
w <- empty 1-braid
witness <- []
repeat k times:
    apply a type-preserving walk of length l  (braid relations, Markov moves,
                                               stabilisation, free insertion)
    apply one crossing change at a chosen site
    append both to witness
emit (w, reverse(witness))
```

The emitted word is a knot with a **certified `u <= k` witness of exactly the
interleaved shape** — crossing changes separated by long isotopy runs — at any
length and strand count we choose. That is the shape the target requires, and the
shape no random scramble produces.

Properties worth stating, because they drive the design:

* `k` and `l` are independent dials: `k` sets crossing-change depth, `l` sets how
  much reshaping sits between changes. The target needs `k = 5` and `l` large.
* The label is an upper bound, not `u`. Where a certified lower bound meets `k`,
  the instance is **exact** and belongs in the calibration set; where it does
  not, it is a feasible-at-cap-`k` positive and nothing more.
* Composites arise naturally, and among them will be instances where one change
  simplifies both factors — the sub-additive behaviour the target exhibits.
  Training only on prime knots and independently generated composites teaches the
  network that `u` is additive, which is precisely the false belief that stops it
  at 6.
* Long words are free. This is where the variable-length architecture earns its
  keep, and it is a better justification for it than "a million examples".

> **M0 result: built, and the label is far looser than this section assumed.**
> `src/rf_knots/interleaved_generator.py` is implemented and every emitted
> instance carries a `verify()`-passing reverse witness (45 new tests; full suite
> 351 passed). Throughput at the depth-five shape (`k=5, l=24`, 4 strands, length
> 32) is **113 instances/s** on one core, cost linear in `k*l`.
>
> But the requested `k` is **not** the achieved difficulty. Over 1000 instances at
> `k = 5`, the Murasugi lower bound is `{0: 63, 1: 86, 2: 40, 3: 7, 4: 4}` —
> **mean 1.01, never 5** — and **14% of instances are literally the unknot**.
> Zero of 174 identified instances were exact at `k`. The `k` changes partly undo
> one another's effect on knot type, so an unfiltered corpus is mostly easy.
>
> Composites *do* arise as predicted (`3_1#3_1#3_1`, `3_1#6_3`, `5_1#3_1`,
> `10_82#3_1`; 20 composite and 119 prime in a 200-instance sample), so that half
> of the argument survives.
>
> **Mitigation, implemented:** an `accept` hook and a
> `--min-certified-lower-bound` filter scoring the Murasugi bound on the free
> reduction, with ~21% acceptance at `>= 2`. Manifests record
> `min_certified_lower_bound` so filtered and unfiltered corpora cannot be
> confused. **Open consequence for M3:** hard positives cost roughly `1/0.21` the
> generation budget at `>= 2`, and the acceptance rate at `>= 4` or `>= 5` is not
> yet measured — that measurement is now a prerequisite for M3's `10^6` target,
> because the plan needs depth, not volume.

### 4.1b Seed from any representation, not only from the unknot

The walk of §4.1 starts at the empty 1-braid because that makes the reverse path
self-certifying. But nothing in the construction requires it. Start instead at
any representation `R` of any knot `K` already in the proof graph, walk the same
interleaved `k` steps, and emit `R'`. Three things follow, and the third is the
one that matters.

1. **A certified Gordian upper bound, free.** `d(R', R) <= k` by construction.
   This is goal 2's training corpus emitted as a by-product of goal 1's, at no
   extra search cost (§7.4).
2. **An unknotting bound by composition.** `u(R') <= k + u_upper(K)` — but the
   composed witness replays end to end **only if** `K`'s bound is carried by a
   program. Against a catalogue claim it is "catalogue-composed" in §3.1's sense,
   and must be filed as such.
3. **A certified *lower* bound, inherited.** `u` is the graph distance to the
   unknot, so it is 1-Lipschitz under a crossing change (note 10, Consequence B):

   ```text
   |u(R') - u(K)| <= d(R', K) <= k     =>     u(R') >= L(K) - k
   ```

   where `L(K)` is any certified lower bound on the seed. **No invariant of `R'`
   need be computed at all** — the difficulty is inherited from the seed and the
   walk length.

**Why (3) is the fix for §4.1's measured defect.** M0 found the unknot-seeded
label loose to the point of uselessness: at `k = 5` the Murasugi bound came out
`mean 1.01, never 5`, with 14% of instances literally the unknot. The reason is
now obvious from the formula — seeding at the unknot means `L(K) = 0`, so the
inherited bound is `0 - k = 0`. **The generator was not weak; it was seeded at
the one point of the graph that certifies nothing.**

Seed instead where `L` is large and free:

| seed family | certified `u` | cost |
|---|---|---|
| `T(2, 2m+1)` torus knots, `s1^(2m+1)` | `m`, a theorem | a one-letter-repeated word |
| positive braids | `g = (L - n + 1)/2` (note 19 §5b) | free from the word |
| any proof-graph node whose interval is exact | its `u` | already stored |

So `T(2,15)` (`u = 7`) walked at `k = 3` yields instances with certified
`u(R') >= 4` **by construction**, which is precisely the depth M3 needs and could
not reach by walking further from the unknot.

**The same inheritance gives Gordian *intervals*, not just upper bounds.** For any
1-Lipschitz `f` vanishing at the unknot — `|sigma|/2`, `|tau|`, `|s|` —

```text
d(R', R) >= |f(R') - f(R)|
```

so every generated pair carries a two-sided certified interval. That is exactly
the shape `rf_knots.embedding_metric.MetricEmbeddingObjective` already consumes:
it is an interval loss with `lower` and `upper` fields, and until now the lower
side has mostly been zero.

> **Measured. The inheritance works, and the effect is not marginal.**
> 18,000 instances over a 36-cell grid of seed strength against walk depth.
> Acceptance at independently computed certified `u >=`, at `k = 5`:
>
> | seed | `L` | `>=2` | `>=3` | **`>=4`** | `>=5` | inst/s | s per `>=4` |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | `0_1` (the §4.1 control) | 0 | 0.31 | 0.08 | **0.01** | 0.00 | 59.9 | 1.193 |
> | `T(2,7)` | 3 | 0.73 | 0.39 | **0.18** | 0.05 | 51.3 | 0.106 |
> | `T(2,11)` | 5 | 0.94 | 0.71 | **0.52** | 0.26 | 48.9 | 0.039 |
> | `T(2,15)` | 7 | 1.00 | 0.92 | **0.79** | 0.56 | 46.3 | 0.027 |
> | `T(2,21)` | 10 | 1.00 | 1.00 | **1.00** | 1.00 | 43.0 | 0.023 |
> | `P(4,23)` | 10 | 1.00 | 1.00 | **1.00** | 1.00 | 22.1 | 0.045 |
>
> Certified `u >= 4` at depth five goes from **0.014 to 1.000**, and a hard
> positive costs **23 ms** instead of 1.2 s — 52x cheaper. The control reproduces
> M0 (mean 1.11 against M0's 1.01), confirming the diagnosis: **the generator was
> never weak, it was seeded where nothing is certified.**
>
> **The bound is sound and conservative.** Over 18,000 instances the
> independently computed bound fell below the inherited one **0 times**, and
> *exceeded* it **62.2%** of the time, by up to +8. The walk adds difficulty at
> least as often as it preserves it; the inheritance is a floor, not an estimate.
>
> **Hazard 4, measured.** At `k = 5` from `L = 3` the inheritance is exhausted
> (`L - k <= 0`) and acceptance collapses to 0.18. "Depth must come from the
> seed" is not a stylistic preference — it is the shape of the `T(2,7)` row.
>
> **The Gordian side is now populated.** 83.7% of emitted pairs carry a non-zero
> certified lower bound, against almost none before, and **40.6% are exact**
> (`lower = upper`). Distribution over the lower bound:
> `{0: 2934, 1: 8567, 2: 3568, 3: 2210, 4: 365, 5: 356}`.
>
> One implementation subtlety worth keeping: the Lipschitz values are **signed**.
> A knot and its mirror sit at `+3` and `-3`, so the unsigned form would report a
> Gordian distance of 0 between them.

**Four hazards, all of which bite if ignored.**

* **Quarantine moves to seed selection, and must close under the `k`-ball.** A
  seed within `k` changes of the target leaks the answer into training even
  though the emitted word is not the target. Exact closure is expensive; the
  practical rule is to exclude every seed whose identity is in the §4.3 set and
  then screen emitted words against the target's fingerprint.
* **Only certified lower bounds may be inherited.** A catalogue `u` is not a
  theorem, and `L(K) - k` computed from one is not a certified bound. Take `L`
  from `lower_bounds.computed_claims` or an attested certificate, never from a
  table row (§1.2's trap).
* **Record pairs representation-to-representation.** `d(R', R) <= k` is a
  statement about two concrete words and is always sound. Promoting it to a
  knot-level `d(K', K)` inherits the graph's attestation of `R`'s identity, which
  is a separate and weaker claim.
* **The walk erodes the certificate; only the seed supplies it.** Each crossing
  change moves `u` by at most 1, and the certificate must assume every one of
  them moved it *downward*, so the guaranteed floor is `L(K) - k` and it falls
  one unit per step. Walking further never adds certified difficulty — it
  subtracts.

  This does **not** say a long walk produces easy knots. It says a long walk
  produces knots whose difficulty is *unknown*, which for training purposes is
  the same thing until someone pays for it. Both routes reach hard instances;
  only the seeded route knows it did:

  | route | seed `L` | `k` | certified floor | cost of a hard positive |
  |---|---:|---:|---:|---|
  | depth from the walk | 0 (unknot) | 5 | **0** | compute a bound on every `R'`, keep ~2% at `>= 4` |
  | depth from the seed | 7 (`T(2,15)`) | 3 | **4** | free — no invariant of `R'` computed at all |

  The measured distribution makes the first row concrete: unknot-seeded at
  `k = 5`, the Murasugi bound came out `{0: 63, 1: 86, 2: 40, 3: 7, 4: 4}`. Hard
  instances *are* in there — 4 in 200 — but finding them means computing a bound
  on all 200 and discarding 98%, at 12–200 ms each.

  **The tension, and how to set `k`.** The walk is what creates the structure
  worth training on: the interleaved shape, the long isotopy runs, the composites,
  the forced diagram growth. Small `k` gives a tight certificate and instances
  that look too much like their seed; large `k` gives diversity and no
  certificate. So choose the seed to afford the `k` you want:

  ```text
  k <= L(K) - (difficulty floor required)
  ```

  For `k = 5` at certified `u >= 4` you need `L(K) >= 9`. That is not exotic —
  note 19 §5b's hardest rung is a 20-letter three-strand **positive braid** with
  `u = g = 9`, certified and already in this repository, and positive braids
  supply an unbounded family at `u = g = (L - n + 1)/2` for free.

  **The interval narrows for the same reason.** With `u(R') <= k + u_upper(K)`,
  a seed whose interval is *exact* gives `[U - k, U + k]`, of width `2k`. Small
  `k` from a strong exact seed is therefore doubly preferable: it certifies more
  and it brackets tighter.

### 4.2 What one walk yields

A single `k`-change backward walk, with intermediate types recorded, supplies in
one pass:

* `k+1` states with feasible-at-cap labels for every remaining budget;
* the whole descending chain of upper bounds along the path;
* set-valued first-action targets at every node (§5.3);
* `O(k^2)` Gordian pairs — every pair of nodes on the path is at distance at most
  their index gap, and adjacent nodes at exactly 1 unless the types coincide;
* zero-CC positive pairs for the `isotopy` head from every isotopy run.

The Gordian pairs are the reason to build the generator now rather than after
goal 1 (§7.4).

### 4.3 Quarantine: the target is a final exam

The answer is in the repository. Without an explicit quarantine, "the network
found a 5-change witness" is uninterpretable.

Excluded from every training corpus, together with their CC=0 closure and their
one-change neighbourhoods, following the protocol in
`docs/gordian-proof-graph-and-vassiliev.md`:

1. the Brittenham–Hermiller 20-letter B5 word and its 2^5 masked states;
2. the Wang–Zhang 254-letter base word, its five marked positions, all 32 mask
   states, and the 518-move terminal trace;
3. the named chain knots `14a_18636`, `15n_81556`, `12n_412`, and every stored
   representation of them;
4. any composite whose summands are `7_1` and `m(7_1)`.

`7_1` itself stays in training: knowing `u(7_1) = 3` is not knowing the answer,
and withholding it would be a different experiment.

If the chain states are later used deliberately as a curriculum — they are an
excellent one — that run is a **distillation** arm under note 20's rules, capped
as a fraction of native updates and reported in its own column. A distilled
5-change witness demonstrates that the search machinery composes; only a native
one is evidence that it discovers.

---

## 5. Learning contract, corrected

### 5.1 A monotone survival head over the whole budget vector

Note 21 asks for `p_solve(state | budgets)` plus a monotonicity check and a
diagnostic that publishes violations. Build the monotonicity in instead.

The crossing-change budget is a small integer, `b = 0..B` with `B` around 6, so
emit the whole vector at once:

```text
h_b = softplus(head_b(state))                       b = 0..B
logit p_solve(state | b) = -c + sum_{j <= b} h_j
```

Cumulative non-negative increments make `p_solve` non-decreasing in `b` by
construction. Do the same over a small grid of compute doses. This buys three
things note 21's version does not:

* **every rollout supervises every budget.** A run that fails at cap 5 and
  succeeds at cap 6 is one observation of a curve, not a contradictory pair of
  labels — exactly the case note 21 identifies as most valuable and then handles
  with a pairwise penalty.
* **monotonicity violations are zero**, so the diagnostic becomes a unit test
  rather than a published metric.
* the CC budget and the semantic-move budget stop being lumped into one
  "objective budget". They differ by three orders of magnitude and behave
  differently.

Keep note 21's requirement that every record stores `successes/attempts` rather
than a last-attempt bit, and its rule that calibration is measured on the natural
distribution.

### 5.2 The tournament is GRPO; use GRPO's stabilisers

Note 21's `K = 10` equal-budget rollouts from one root, with weight `+1` for the
winner and `-1/9` for the rest, maximising
`sum_i w_i sum_t log pi(a_it | s_it)`, is **GRPO — Group Relative Policy
Optimisation** (Shao et al., *DeepSeekMath*, arXiv:2402.03300; the method behind
DeepSeek-R1). GRPO drops PPO's learned value critic and takes the baseline from
the *group itself*: sample `K` completions from one prompt, subtract the group
mean reward, divide by the group standard deviation, then apply PPO's clipped
ratio and a KL penalty to a reference policy. Note 21 arrived at the same
structure independently — `K` rollouts from one root, weights summing to zero —
which is why the fixes below are named rather than invented.
The centring argument is right and is the best part of that section: a shared
prefix contributes coefficient zero, so credit appears only where trajectories
diverge.

The objective as written is unbounded, though. For a loser, maximising
`-(1/9) sum_t log pi` is maximised by driving `pi -> 0`, which diverges; with
trajectories hundreds of steps long this is not a theoretical worry. Three
standard fixes, all compatible with the centring:

* **standardise the group advantage** — `w_i <- (r_i - mean r)/(std r + eps)` —
  which keeps `sum_i w_i = 0` and bounds the scale;
* **clip the importance ratio** against the behaviour policy that generated the
  group, PPO-style, so a stale group cannot produce an unbounded step;
* **KL-anchor to the previous checkpoint**, which is also what protects the
  retention panel note 20 requires.

On length normalisation, note 21 is right that dividing each trajectory by its
own length breaks prefix cancellation. The fix that keeps both properties is a
**group-constant** divisor — the group's mean length, or its total token count —
applied identically to every member.

Keep the prefix-trie refinement, and compare it against the root-level version at
equal network evaluations as note 21 says. Keep the root version if it wins: it
is simpler and its credit assignment is easier to audit.

Keep note 21 §4.5 unchanged: outside a comparable group, a single failed
trajectory trains legality and nothing else.

### 5.3 The per-site head is the highest-leverage supervised task

Note 19 §6 item 4 has been open since it was written. It should be built first,
for three reasons:

* it *is* the outer search's branching factor — turning 254 candidate sites into
  a top-10 shortlist is worth more than any value-head improvement;
* it has **abundant, cheap, honest labels**: on every backward-generated
  instance the sites lying on the known witness are positives, and note 21's
  set-valued loss `-log sum_{a in A_plus} pi(a|s)` is exactly the right form,
  because several sites usually work;
* it is supervised, so it can be gated before any RL runs: recall@10 on held-out
  identities, reported by word length and strand count.

Absent sites are unexplored, not negative. That rule from note 21 §5.1 is what
makes the loss set-valued in the first place, and it applies here verbatim.

### 5.4 Outcome classes, carried forward

A/B/C/D from note 21 §4.2 are adopted unchanged, with one addition specific to
this target: a run returning a **6-change witness under a cap of 5** is class C
with a recorded incumbent, not class A. It solved a different problem.

---

## 6. Architecture, reconciled with the code that exists

### 6.1 Three globals, named as the code names them

`src/rf_knots/embedding_encoder.py` already emits three normalised heads —
`knot`, `isotopy`, `layout` — and `embedding_metric.py` already trains `knot`
against the crossing-change interval and `isotopy` against the diagram-distance
interval. Note 21's `E_knot`/`G_layout` pair should be read onto that existing
vocabulary rather than introducing a third naming:

| head | invariant under | trained by | consumed by |
|---|---|---|---|
| `knot` | every zero-CC move | CC interval metric, retrieval | outer search key, `h(K)`, Gordian head |
| `isotopy` | relabelling only | diagram-distance interval metric | duplicate detection, inner value |
| `layout` | nothing | policy/value gradients only | inner policy, site ranking |

Note 21's substantive point survives the renaming and matters: a metric loss that
reaches the whole backbone will flatten the representation-specific geometry the
policy needs. Apply the metric losses to `knot` and `isotopy`; let `layout` be
shaped by the policy alone.

### 6.2 Build the pair network first

The second goal is `d(K_1, K_2)`, and `u(K) = d(K, 0_1)`. Building a single-input
network now and a pair network later means migrating checkpoints twice.

Build one encoder `phi`, applied to both inputs, with a symmetric interaction —
architectural symmetry via `[ |e1 - e2|, e1 * e2, e1 + e2 ]`, so `d(a,b) =
d(b,a)` holds by construction rather than by training. For the unknotting task
the second input is the empty word and its embedding is a cached constant, so the
extra cost is the interaction MLP alone. §7 develops this.

### 6.3 Incremental recompute and the throughput gate

Note 21's requirement that a head move recompute only the positional context,
with an equivalence check against full recomputation, is right and should be a
test rather than a note. Add its counterpart for the two-level search: after a
**single crossing change** the diagram changes in one cell, so the encoder should
support an incremental update whose equivalence to full recomputation is also
asserted. The outer search performs one of these per expansion, so this is the
hot path.

The throughput gate from note 21's stage 2 stands, with the shapes this note
needs: forward and backward at `(L, n)` of `(64, 8)`, `(128, 12)` and
`(256, 21)`, reporting evaluations per second, memory and padding fraction — on
the single host that is actually available. The offload machine is retired; a
plan that assumes a cluster is a plan that does not run.

> **M0 ran that gate. Examples/s on the shipped encoder:**
>
> | shape | mode | CPU naive (b=256) | CPU chunked (12) | MPS best |
> |---|---|---:|---:|---:|
> | (64, 8) | fwd | 148 | **3,112** | **16,663** |
> | (128, 12) | fwd | 80 | **2,354** | **9,066** |
> | (256, 21) | fwd | 39 | **1,594** | **4,561** |
> | (64, 8) | fwd+bwd | 129 | **1,142** | **4,432** |
> | (128, 12) | fwd+bwd | 70 | **812** | **2,286** |
> | (256, 21) | fwd+bwd | 35 | **486** | **1,171** |
>
> **MPS** is Metal Performance Shaders, PyTorch's Apple-Silicon GPU backend
> (`--device mps`, as the README's trainer already uses) — the local equivalent
> of `cuda`. It is the only accelerator this project has, so every "feasible"
> verdict below is an MPS verdict. Three properties of it bear on the plan: its
> memory is *unified* with system RAM, so the 16 GB ceiling is shared with
> everything else on the host; unsupported ops fall back to CPU silently unless
> `PYTORCH_ENABLE_MPS_FALLBACK` is watched; and there is no multi-GPU path, so
> the only way to buy more throughput here is a bigger batch or a smaller model.
>
> **Two free multipliers, both found by measurement rather than design.**
>
> 1. **A PyTorch CPU dispatch cliff between batch 12 and 16.**
>    `ResidualWordBlock`'s 1x1 `Conv1d` costs 0.039 ms at batch 8 and 26.271 ms
>    at batch 32 — **674x the time for 4x the work**. Feeding the same 256
>    examples in micro-batches of 12 is numerically exact (max abs difference
>    2.1e-07 across all three heads) and **9–41x faster**. This is a latent
>    property of the shipped encoder, not of this plan, and it costs one loop to
>    claim.
> 2. **Length-sorted buckets.** Padding is 49.9% to `L=64` at the depth-five
>    shape and 21.8% to `L=256` at the larger one; bucketing drops it to
>    **0.4–0.6%**, roughly another 2x, free.
>
> Peak memory across the whole 60-configuration sweep was 1.05 GB RSS and 1.16 GB
> on the MPS driver — nowhere near the 16 GB ceiling. **Memory is not a
> constraint; wall clock and disk are.** Variance is reported honestly in the
> artifact: MPS below batch 32 is queue noise (one cell +/-80%), MPS at batch 256
> is +/-3%, CPU backward drifts +/-40% under host contention.

---

## 7. Goal 2: Gordian-distance witnesses

### 7.1 What generalises for free

The two-level search of §3 generalises with one substitution: the outer terminal
test changes from "is this the unknot" to "is this `K_2`". Everything else — the
fingerprint key, the site head, the survival head, the interleaved generator, the
tournament — is unchanged. The cross-relaxations already written into
`docs/gordian-proof-graph-and-vassiliev.md`,

```text
U(A) <= d_upper(A,B) + U(B)          d_upper(A,B) <= U(A) + U(B)
```

serve simultaneously as search bounds and as consistency losses.

### 7.2 The terminal-test asymmetry is the hard part

Unknotting has an exact, cheap terminal: the empty word. Gordian distance does
not — "is this diagram `K_2`?" is answered in practice by an attested identity or
a replayed zero-CC path to a stored representation of `K_2`, which is exactly
what the shared proof substrate provides and exactly why the second goal needs
`unknotdb` more than the first does.

Design for the consequences rather than discovering them:

* a **`p_same_knot` head is a search heuristic only**. It never certifies. A
  positive triggers a zero-CC search for an actual path; a negative merely
  deprioritises.
* **meet-in-the-middle is not optional here**; it is the primary mechanism. Grow
  frontiers from both endpoints and detect meetings by exact `rep_key`.
* a **finite Vassiliev prefix is not an isotopy certificate**. The sidecar design
  already says this, and it is the most likely place for a false positive to
  enter.

### 7.3 Constraints available for free

Impose them architecturally where possible, as losses otherwise. Each is a
property of a metric that a plain regression head will otherwise violate:

| property | how |
|---|---|
| `d(a,b) = d(b,a)` | symmetric interaction, architectural |
| `d(a,a) = 0` | architectural, and a unit test |
| `d(a,c) <= d(a,b) + d(b,c)` | sampled-triple hinge loss |
| `abs(d(a,c) - d(b,c)) <= 1` when `a`, `b` differ by one change | note 10, Consequence B; self-play emits these pairs free |
| mirror / reversal equivariance | augmentation with recorded lineage |
| `d(K, 0_1) = u(K)` | shared head with the unknotting task |

The 1-Lipschitz constraint is the interesting one. Note 10 observes that *every*
unknotting lower bound in the literature is a 1-Lipschitz function vanishing at
the unknot, and that finding new ones is the honest role for learning here. A
pair head trained under that constraint is a candidate generator for exactly that
search — conjectural, pointing a human at where to look, never a proof.

### 7.4 The data is a by-product of goal 1

Each `k`-change backward walk emits `O(k^2)` distance-bounded pairs (§4.2), and
every crossing change made during any search is a distance-1 pair. The existing
`scripts/build_gordian_embedding_sidecar.py` and the read-only contract in
`rf_knots.embedding_gordian` already define how catalogue bounds enter without
masquerading as proof edges. Goal 2's corpus therefore accumulates while goal 1
is pursued, at no extra search cost — provided the pairs are recorded from the
start, which is why this section is in this note and not a later one.

### 7.5 Gates

1. Recover known Gordian distances on the pairs already in the sidecar, with
   identity-level splits.
2. Beat the trivial bound `d <= u(A) + u(B)` on some pair, with a replayed path.
3. Zero triangle-inequality violations above tolerance on held-out triples.
4. `d(K, 0_1)` agrees with the unknotting head to within the interval width. If
   the two heads disagree, one of them is wrong, and the shared parameterisation
   is what makes that detectable.

---

## 8. Milestones, gates, kill criteria

Every milestone states what would make it stop. `docs/lessons.md`'s rule applies
throughout: project the cost of a sweep, and log inside it, before launching it.

### M0 — Measure the floor (days, no training) — **done; results in §Status**

**What.** Four measurements and no training: whether the published witnesses
replay in this project's move set, what the certified machinery says about each
state on the chain, what today's search achieves at cap 6 and cap 5, and what one
network evaluation costs.

**Why.** Every later milestone's plan is arithmetic on numbers only this stage
produces — and `docs/lessons.md` is explicit that a gate set without a measured
floor measures nothing. It is deliberately the one stage that can refute the rest.

**M0.1 — replay.** Put the Brittenham-Hermiller chain and the Wang-Zhang cube
through `rf_knots`' own verifier; confirm both parse, both replay, and record the
**semantic-move length** of each leg. This decides §2.1's constant `A`.

**M0.2 — certified bounds.** Run the certified bounds on every chain state, from
both representations, and report the interval each one supports.

**M0.3 — the untrained floor.** Run the current search on the 20-letter word at
cap 6, then at cap 5, at three simulation doses and eight seeds. Record what it
does. `docs/lessons.md` is explicit that a gate set without a measured floor
measures nothing.

**M0.4 — throughput.** Benchmark evaluations per second at `(64,8)`, `(128,12)`
and `(256,21)`, and project the cost of every later milestone from those
numbers.

**Kill criterion.** If the published witnesses cannot be replayed in this
project's move set, the representation gap *is* the project, and M1–M5 are
premature.

### M1 — The per-site head (weeks) — **M0 verdict: feasible**

**What.** One shared scorer applied to every candidate action site in a diagram —
one logit per site, no fixed-width output — trained by ordinary supervision to
rank two things: which crossing to change, and where the diagram may grow.

**Why.** The branching factor *is* the search. M0.4 measured 10,598 legal
successors on the target's long word, 95.9% of them `INSERT`, at 39 nodes/s;
depth-5 search over that is hopeless, and turning thousands of sites into a
top-ten shortlist is worth more than any value-head improvement. It is also the
only head whose parameter count is independent of word length and strand count,
so it is what lets a checkpoint transfer across problem sizes at all — M0.4
measured the current head failing exactly there (`size mismatch [1156,96] vs
[25604,96]`). And because the labels come free from generated witnesses, it can
be trained and judged **before** any reinforcement learning is paid for.

Measured cost: 1.1 h generation, then 0.5 h (instance-level) to 30.5 h
(state-level) of MPS training, 115 MB. The binding constraint is the wall clock
of the state-level pass; memory and disk are nowhere near their limits. Note the
rescoped first job from §2.3: the head must rank **where the diagram may grow**,
since 95.9% of legal successors on a long word are `INSERT`.

Per-crossing policy over the raster; set-valued supervision from
backward-generated witnesses. **Gate:** recall@10 >= 0.8 of a witness-realising
site on held-out identities, reported by length and strand count, including a
transfer row at a strand count absent from training.

**Kill criterion.** Recall@10 no better than ranking by a hand-written heuristic
— for instance, sites whose flip reduces free-reduced length — means the head has
learned nothing the search could not have had for free.

### M2 — Depth-`k` outer search with identification (weeks)

**What.** The two-level search of §3: an outer best-first walk over the
crossing-change graph, keyed on knot identity rather than diagram, with an inner
MCTS doing zero-CC diagram surgery between changes, and identification as the
terminal test.

**Why.** The published pipeline this project is measured against is depth **one**
— flip one crossing, then look the result up in a table. This is the depth-`k`
generalisation `u(K) <= k + u(K_k)`, which is the structural advantage rf-knots
has and the reason its witnesses are self-certifying rather than table lookups.
Identification is what turns a reached state into a usable bound; note 19 called
it the bottleneck of the prior work, and SnapPy now closes it.

Reproduce the depth-1 DKT result on knots whose answer is known, then depth-2 and
depth-3 on the chain's own descendants. **Gate:** re-derive the known `u` of a
pre-registered set of tabulated knots, counting replay-complete terminals
separately from catalogue-composed ones.

### M3 — Interleaved generator at scale (weeks)

**What.** Run the interleaved backward generator (§4.1) at scale, seeded from
knots carrying large certified lower bounds (§4.1b), to produce a corpus whose
difficulty is known by construction rather than hoped for.

**Why.** This is the training corpus for M1 and M4, so its quality caps theirs.
M0 measured the unseeded version producing instances of mean certified difficulty
**1.01** at `k = 5`, with 14% of them literally the unknot — a corpus that would
train a per-site head on instances where any site works, and a survival head
calibrated on a distribution that does not contain the target's regime. Note 21's
first risk, "a million rows but little information", is realised here or avoided
here.

`10^5` then `10^6` deduplicated states with witnesses, spanning `k = 1..6`,
lengths to 256 and strands to 21. **Gate:** note 21's four counters reported
separately; dedup ratio measured rather than assumed; identity-level splits
audited by `scripts/audit_embedding_validation_leakage.py`; the §4.3 quarantine
verified by hash.

**Kill criterion.** If deduplicated unique states per CPU-hour make `10^6`
unreachable on the available host, cut the target and say so, rather than
counting augmentations.

> **M0: the kill criterion fires, and the honest cut is the record format.**
> Measured unit costs at M3's *stated* shape (lengths to 256, strands to 21) are
> far worse than at the depth-five shape: generation **7.4 instances/s** (against
> 144.7), **844.7 kB** per raw record (against 30.7 kB), and 200 ms per certified
> filter evaluation (against 11.9 ms).
>
> | | verdict | binding constraint |
> |---|---|---|
> | `10^5` | **feasible** — 3.8 h unfiltered, 1.9 days filtered, 224 MB | CPU under the certified filter |
> | `10^6` | **not feasible as stated** | record format, then CPU |
>
> At `10^6`, raw records are **844 GB** — impossible. Gzipped they are 14.0 GB,
> which is 42% of free disk and irresponsible on a 93%-full volume. Storing
> `(start word, action list)` and replaying on read is **2.24 kB** per record,
> **2.24 GB** total, at 37 replays/s. That is the format M3 should have specified
> from the start, and it is what §7.2 of note 21's replay contract implies anyway.
> The second constraint is real too: **18.75 single-core days** under the
> certified filter.
>
> **Revised M3:** `10^5` at the stated shape, or `10^6` in compact form with the
> filter budget explicitly cut — but not both, and not on this host without
> saying which was given up.
>
> **Update from §4.1b, which may reopen `10^6`.** The seeded generator measured
> the cost split: 81 s generating against **298 s (79%) computing the independent
> certified bound**. A seeded corpus does not pay that 79% at all — its difficulty
> is inherited from the seed, not measured on each instance. That removes the
> "18.75 single-core days under the filter" constraint outright and leaves the
> record format as the only binding one, which §Status consequence 6 already
> solves at 2.24 GB.
>
> **This is a projection, not a measurement, and must not be quoted as one.** The
> seeded rates above are at the depth-five shape; generation at M3's *full* stated
> shape (lengths to 256, strands to 21) was measured only unseeded, at 7.4
> instances/s. One measurement — seeded throughput at the full shape — decides
> whether `10^6` is back on. Run it before re-declaring M3 feasible.

### M4 — Joint training (weeks)

**What.** Train the network end to end on the M3 corpus: the monotone survival
head over the whole crossing-change budget vector, policy by GRPO with its
stabilisers, set-valued targets from the proof graph, and note 20's retention
panel running throughout.

**Why.** This is where the parts become a solver rather than a collection of
heads. The gates are calibration gates because a miscalibrated `p_solve` is worse
than none at all — the outer search trusts it to decide what to expand, so a head
that is confidently wrong at cap 5 wastes the entire compute budget in the one
regime that matters.

Monotone survival head, GRPO with stabilisers, set-valued policy targets, and
note 20's retention panel at a fixed dose. **Gate:** calibration on the natural
distribution (Brier, reliability by budget bucket, ECE); retention >= 0.8 on the
frozen panel; monotonicity violations exactly zero.

> **M0 feasibility.** M4 is **encoder-bound, not environment-bound** — the
> environment is 5–11% of an evaluation.
>
> | | verdict | cost |
> |---|---|---|
> | entry scale (20-letter B5 word) | **feasible** | 1.0e9 evaluations, **0.78 days** on MPS |
> | cube scale `(256, 21)` | **not feasible** | 7.9e9 evaluations, **22.4 days** on MPS |
>
> The entry scale is **29x cheaper** and it is the one that fits. That is an
> independent arithmetic confirmation of §Status's decision to start every campaign
> from the 20-letter word — two different measurements arriving at the same
> instruction.

### M5 — The target (the point)

**What.** A cap-5 campaign against `7_1 # m(7_1)`, run from the 20-letter entry
word with the §4.3 quarantine enforced and the 254-letter cube unused.

**Why.** It is the point of the plan; everything before it is instrumental. And
it is the only stage whose success criterion is a mathematical object rather than
a metric — a replay-verified sequence, or nothing.

A cap-5 campaign on `7_1 # m(7_1)` from the 20-letter entry word, quarantine
enforced, cube unused. **Success:** a replay-verified 5-change witness found
natively, reported with seeds, dose, evaluations and the full path. **Partial
success worth publishing:** a 5-change witness found only with distillation from
the chain, reported as such. **Stretch:** a 4-change witness — which, if it
survives independent replay and a literature check, is new mathematics and should
therefore be treated with corresponding suspicion first.

### M6 — Gordian (after M5, or in parallel once M3 lands)

**What.** The two-representation network of §7 — one encoder applied to both
inputs with a symmetric interaction — producing certified-interval estimates and
replayable witnesses for the Gordian distance `d(A, B)`.

**Why.** It is the stated second goal, and `u(K) = d(K, 0_1)` makes the first a
special case of it rather than a separate system. §6.2 therefore builds the pair
network first, and §7.4 means its corpus accrues during M3 at no extra search
cost — so M6 is mostly the bill coming due for work already done.

Pair head, cross-relaxation, meet-in-the-middle; gates as §7.5.

---

## 9. Metrics

`witness_rate@(cap, dose, seeds)` — the fraction of independent seeds producing a
replay-verified witness that meets `cap` at a fixed simulation dose — is the
headline number, and it is what "high probability" in the goal has to mean.
Targets: `>= 0.95` on the rung ladder at cap equal to the known `u`; `>= 0.5` on
the target at cap 5 before the word "mastery" is used about it.

Everything else is reported alongside, and none of it substitutes:

* **solver** — solved set at fixed evaluations; lexicographic `(cc, moves)`;
  replay-verified rate; results at lengths and strand counts absent from
  training; retention on the frozen panel.
* **value** — Brier, log loss, reliability by budget bucket, ECE, all on the
  natural distribution, with certified-negative and censored-failure rows kept
  apart.
* **representation** — padding and cyclic invariance; retrieval recall@k against
  the exact fingerprint; transfer to unseen lengths; invariant probes as
  diagnostics only; and the one that settles the question — the change in outer
  expansions per terminal at equal evaluations with the embedding disabled.
* **search** — expansions per terminal, transposition hit rate, and the fraction
  of terminals that are replay-complete rather than catalogue-composed.

Report `cc/sr`, never `cc` alone: `docs/lessons.md` records that a run which
abandons hard instances drops them from its own average.

---

## 10. Risks

Note 21's eight risks are adopted. Four are sharpened or new.

> **Which M0 finding is the dangerous one, and why it is not the loudest.**
> M0 returned four negatives. Three announce themselves: the certificate does not
> translate (a job fails), the floor is `0.00` (a search returns nothing), `10^6`
> records are 844 GB (a projection refuses). You cannot proceed past any of them
> by accident.
>
> The fourth is silent. The unknot-seeded generator emitted 1000 instances at
> `k = 5`; every witness verified, every label `u <= 5` was **true**, and nothing
> anywhere failed. Only an independently computed lower bound revealed that the
> mean certified difficulty was **1.01** and that 14% were the unknot. The label
> was true and vacuous at the same time.
>
> This is note 21's risk 1 — "a million rows, but little information" — realised
> and measured rather than anticipated. Its precise form here: **`k` was a dial
> that was not connected to anything.** It counts the changes *made*, not the
> difficulty of the result, because the changes partly undo one another. And
> §4.7's curriculum controller, which steers toward `p_solve ~ 0.5`, cannot steer
> with a dial that does not move the quantity it is regulating.
>
> Left unfixed it damages the learning contract in three specific places, none of
> which would have shown up as an error: the per-site head (§5.3) would rank
> sites on instances where nearly any site works; the monotone survival head
> (§5.1) would be perfectly monotone and confidently wrong at cap 5, the one
> boundary that matters; and MCTS would learn a prior that never spends depth.
> The fix is §4.1b — inherit difficulty from a seed with a certified `L`, so the
> dial is `L(K) - k` rather than `k`.

1. **The objective inverts** (§2.1). Concrete, measured, and it fires on the
   target. Fixed by the lexicographic comparison and by recording `A`.
2. **The summand shortcut.** `connected_summands` makes `u <= 6` easy to find,
   and a network trained on independently generated composites will believe `u`
   is additive. Mitigated by the interleaved generator, which produces
   sub-additive instances, and by the strict cap.
3. **False identification at depth.** The target's own fingerprint is ambiguous
   (`7_1 or 12n_749`). A wrong identification at the outer level produces a
   confidently wrong bound. Mitigated by reporting ambiguity instead of taking
   the first match, and by preferring replay-complete terminals.
4. **Answer leakage.** The answer is in the repository in two forms, plus a named
   chain of intermediate knots. Mitigated only by the §4.3 quarantine and the
   native/distilled split; a run without a verified quarantine hash is not
   evidence about discovery.

And the honest overall risk: the published witness needed a 254-crossing chart to
make all five changes visible in one diagram. If reaching depth five requires
diagrams that large, the inner search's horizon is thousands of moves and the
approach may be out of reach on one host. M0 is designed to find that out in days
rather than months.

---

## 11. Open questions

1. What is the smallest diagram on which a 5-change witness for this target
   exists? The published charts give 20 letters (2 changes realisable) and 254
   letters (all 5). Anything in between is unknown, and is itself a publishable
   measurement.
2. Does Brittenham–Hermiller establish a lower bound stronger than the `u >= 2`
   computed here? Until that is read, "is 4 open?" is unanswered, and M5's
   stretch goal should not be advertised.
3. Is the `Z/3` rank of the double branched cover (note 19 §4.1, still unbuilt)
   a strong enough hot-path discriminator to keep Jones off the outer loop
   entirely?
4. Root-level tournament or prefix-trie tournament, at equal evaluations? Note 21
   poses this; it is cheap to settle, and should be settled before M4.
5. Should `h(K)` predict remaining changes, or predict `p_solve` and be used as
   `-log p`? The second composes additively along a path and therefore behaves
   better as a best-first priority.
6. What is the honest unit of "a million" for this campaign — deduplicated
   semantic states, or outer-level distinct knot types? The second is the smaller
   and more meaningful number, and it is the one the outer search consumes.
7. Does the pair encoder's shared parameterisation cost anything measurable on
   the unknotting task, given the cached unknot embedding? If it does, §6.2's
   pair-first decision should be revisited before M4, not after.
