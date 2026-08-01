# 12 — The serial (Turing-machine) formulation: what was measured

Written after the first two ladder runs. The serial formulation is the one where
the action space stops depending on word length: a **head** points into the word,
the agent sees a window around it, and reaching a distant site costs actions
instead of logits. It is the formulation the macro / metasystem-transition
direction needs, because promoting a fixed action *sequence* to a single action
only keeps the action space `O(1)` if the space was `O(1)` to begin with.

It scored **0** on the first ladder, against 7–8 for the parallel candidates. Then
it scored **9 of 9** on the second. This note is why, and what the difference
implies.

## 1. The defect was the readout, not the formulation

Three causes, found by probing trained checkpoints rather than by reading the
design. In descending order of how much they explain:

**The policy head could not see *where*.** `SerialBraidNet` pooled the window with
mean+max and read every logit off that vector. Mean and max are precisely the two
statistics that discard position — in a formulation whose entire question is "is
the actionable site *at* the head, and if not which way?". Measured on the trained
`serial-w7-head`: cyclically rolling the window contents moved the policy by 0.14
(median max-|Δp|) against 0.40 for a genuinely different state, and that residual
leaked through the convolution's zero padding rather than being represented. The
parallel net had had a *positional* head — `Conv2d(channels, kinds, 1)` — since
the beginning; the serial one dropped it.

**Half the shifts were blind.** The window was `[head, head+w)`, so a right shift
moved into tape the agent could see and a left shift into tape it never had —
presented as two symmetric actions. In the episodes it failed, `serial-w7-head`
played **118 `SHIFT_LEFT` against 2 `SHIFT_RIGHT`**; in the ones it solved, 4 right
against 1 left. It had latched onto the uninformed direction.

**`O(L)` reachability put sites past the search horizon.** One stride of `w/2`
makes an arbitrary site up to 8 plies away, and a ply of repositioning is a ply of
MCTS *depth*. The dominant failure signature is not slowness but substitution:
unable to reach anything, the agent fell back on the always-legal moves. Its
failures ran 69 `STABILIZE_NEG` + 41 `STABILIZE_POS` + 67 `INSERT` against 21
`REDUCE` — it inflated the word because inflating was all it could do at the head.
`serial-w7-window` showed the same cause differently: 204 `REDUCE@+0` against 198
`INSERT(s1+)@+0`, a null 2-cycle burning budget at zero net progress.

**Attribution.** The ablation carrying the *original* single stride, with only the
positional readout and the centred window, also clears the wall — in 22 iterations
against 12. So the stride set is a ~2× training-efficiency gain, and the
**positional readout was the unlock**. Worth recording precisely, because the
first write-up listed three causes without ranking which were necessary.

## 2. What it is worth, once it works

On the same stages and promotion rule as the parallel candidates:

| | highest stage | avg(cc) at stage 8, u=3 |
|---|---:|---:|
| `s-window-128` (serial) | **9** | **3.00** |
| `search-heavy` (parallel) | 8 | 4.00 |
| `wide-net` (parallel) | 8 | 3.00 |
| `u1-puct` (parallel) | 7 | — (0.00 solved) |

Six of the seven arms that reach stage 8 hit the proved unknotting number exactly;
`search-heavy`, the strongest parallel candidate, is the one that does not.

### The A/B objective is inert in the parallel formulation and live in the serial one

This is the interesting result, and it answers a question the design study left
open. Reading `moves` at the two trained edges of `log(A/B)`:

| | moves at A:B=1000:1 | moves at A:B=1:10 | ratio |
|---|---:|---:|---:|
| `search-heavy` (parallel), st. 7 | 10.3 | 10.2 | 1.01× |
| `wide-net` (parallel), st. 7 | 10.2 | 10.2 | 1.00× |
| `u1-puct` (parallel), st. 7 | 10.3 | 10.3 | 1.00× |
| `s-head-128` (serial), st. 8 | 63.0 | 10.0 | **6.3×** |
| `s-head-budget96` (serial), st. 8 | 59.0 | 10.0 | **5.9×** |

The parallel candidates emit the same policy at both ends of the front to two
decimal places — exactly what the Pareto argument predicted (`moves[k] = m₀ + k`
is minimised at the smallest `k` for every positive λ, so the FiLM conditioning has
nothing to condition on). The serial arms respond by 5–6×.

The mechanism is that shifts are charged to the same budget the `moves` metric
reads, so head travel is a real cost and `moves[k]` stops being `m₀ + k`. Which
forces an honest caveat: **serial `moves` and parallel `moves` are not the same
quantity.** The serial number is edits *plus* travel. The dynamic range is real;
whether that is the objective you want is a design decision, not a measurement.

## 3. The register: written registers fail, accumulators are the design

The formulation was proposed with a **memory in the head** — an embedding, order
128 floats. The original build had none: state was `(pgx_state, head)` with `head`
an integer index, nothing carried between plies. A scan machine with no
accumulator. So §2 is a **floor** — the serial formulation cleared the ladder and
beat the parallel nets on crossing-change optimality *without* the memory that
motivated it.

### The cheap version was tried, and it is a clean negative

`serial_registers` gives the head K binary registers with one TOGGLE action each —
the finite control state of a Turing machine, written by the agent, so no gradient
through the memory and no BPTT. A toggle costs a ply, since free writes would make
it an oracle rather than a machine.

```
[s-reg8] stage 1 unknot+6: solved 0.50 after 60 it (capped)
[s-reg4] stage 1 unknot+6: solved 0.83 after 36 it (plateau)
```

`s-reg8` collapsed onto the **exact rung where the pre-fix serial candidates died**.
`s-reg4` cleared it but needed 36 iterations against 12 for the matched
no-register arm, and from a fresh start reached rung 7 where `s-head-128` reached
the equivalent of 9. **The cost is monotone in register count.**

The mechanism is action-space dilution. A TOGGLE never changes the word, so every
toggle is a simulation spent on a branch that cannot make progress; at K=8 that is
8 of 34 actions, roughly a quarter of the search. The general lesson is worth more
than the arm:

> **A register that nothing reads is noise.** Giving an agent somewhere to put
> memory does not give it a reason to. The read side has to come first.

### What reads the register: an automatic accumulator

The design that works keeps the `O(1)` action space and the same search budget, and
changes only *what accumulates over the tape*. The controller still sees a window;
the encoder additionally receives a head-relative scan of the whole word:

| arm | accumulator | fair or oracle |
|---|---|---|
| `s-gru128` | unconstrained GRU-128 | fair |
| `s-fsa32` | learned 32-state soft automaton | fair |
| `s-ff4-p5` | learned 4×4 matrices over 𝔽₅ | fair |
| `s-burau-oracle` | fixed Burau at `t = −1, 1/2` | **oracle** |

The mechanism that makes the algebraic arms more than an arbitrary recurrence is
`SequenceBraidNet.regularization_loss`, which penalises violations of the braid
relations — `σᵢσᵢ⁻¹ = 1`, `σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁`, and far commutation. It pushes
the learned operators toward a representation of `Bₙ`; the experiment below
shows why a penalty does not make that constraint exact.

#### Negative result: retire `s-ff4-p5`, retain the constrained version

The experiment falsified the stronger claim as implemented: a **soft** relation
penalty plus rounded straight-through gradients did not force a representation.
On `T(2,3)+4`, the arm capped after 40 iterations with solve rates `10/12`,
`10/12`, and `1/12` at `A:B = 1000:1`, `10:1`, and `1:10`.  After training,
three of its eight rounded 4×4 matrices were singular, none of the four
positive/negative generator pairs were exact inverses, and none of the three
adjacent braid relations held exactly.  Exact arithmetic after rounding cannot
recover information already destroyed by singular, relation-violating maps.

`s-ff4-p5` is therefore retired from the live leaderboard but kept in code and
artifacts as a negative result.  The idea worth retaining is a **constrained
finite-field representation**:

1. Generate each positive generator directly in `GL(d, 𝔽ₚ)`, never in the full
   matrix algebra, so singular operators are impossible.
2. Define the negative generator as the exact finite-field inverse of the
   positive generator; do not learn it independently.
3. Enforce braid and far-commutation relations by construction or by projection
   onto the exact constraint set after every update.  A soft residual may rank
   alternatives inside that set, but must not define membership in it.
4. Only after the exact `d=4, p=5` version learns should dimension, prime, or the
   `1⊕4⊕5` carrier be increased.

This preserves the original bounded-alphabet, exact-arithmetic motivation while
removing the degeneracy actually observed in the learned operators.

Working in a **finite field** rather than in floats is what makes this a machine
rather than an approximation: arithmetic is exact and closed, so the register is a
bounded alphabet, and the overflow problem of multiplying ~48 matrices disappears
entirely. Constraints: `A ≠ 0`, `δ = −A² − A⁻² ≠ 0` (δ = 0 is the degenerate
Temperley–Lieb algebra), and avoid characteristic 2, which collapses δ.

Measured cost, batch-1 forward — the quantity MCTS is bound by — is 1.3× for the
GRU (its loop is fused in C) and 2.9–4.2× for the hand-rolled scans. Affordable.

Two consequences of having a register at all.

**It cannot represent a knot invariant, and that is a theorem, not a training
problem.** The Markov trace is a left-to-right matrix product — an associative
scan over the whole word. A convolution with receptive field `R` composes at most
`R` letters (`R = 11` here, against `L` up to 48), and a memoryless head composes
one window at a time with no state. So `word → Jones` is outside the function class
of both formulations as built. A register makes it *representable*; whether it is
learnable is then an experiment.

**Where the register has to live.** `game.step` is pure, JAX-jitted, and the
four-way correctness checks depend on it staying that way, so a learned vector
cannot go inside it. The recurrence belongs where the network is already invoked:
`h' = f(h, observation, action)`. The scaffolding exists — `Node.hidden` in
`search.py` is unused, and `Dynamics` in `networks.py` raises `NotImplementedError`
saying the braid environment "needs an action-embedding transition instead". State
is opaque to search and every node keeps its own, so no search change is needed.

The cost is real: the same `(word, head)` reached by two paths carries two
different registers, so the tree stops being a DAG over states and value estimates
become path-dependent. MuZero lives with this.

### Sizing: why `s-ff4-p5` reaches Alexander and not Jones

To *compute* an invariant by scanning, the carrier must hold the running algebra
element, not the final value. The decisive fact is a decomposition rather than a
dimension count. For `B₅`, the Temperley–Lieb algebra `TL₅` has dimension
Catalan(5) = 42, and it **splits into irreducibles of dimension 1, 4 and 5**
(through-strand counts 5, 3, 1; check `1² + 4² + 5² = 42`). The Markov trace —
hence Jones — is a weighted sum of the ordinary traces of those three blocks, the
weights being Chebyshev polynomials in `δ`.

And the 4-dimensional block `V₅,₃` **is** the reduced Burau representation. So:

| carrier | dim | represents | yields |
|---|---:|---|---|
| `s-ff4-p5`, 4×4 over 𝔽₅ | 4 | reduced Burau = `V₅,₃` | Alexander / Conway, and *one of Jones's three blocks* |
| 5×5 | 5 | the largest block `V₅,₁` | the smallest carrier that reaches past Burau |
| block-diagonal 1⊕4⊕5 | 42 entries | all of `TL₅` | **Jones** |

Four evaluation points means four choices of `A ∈ 𝔽ₚ`, so 4 × 42 field elements
for Jones at four points. Jones also needs the writhe correction `(−A³)^{−w}`;
writhe is a sum of signs, already computed in `reference.py`.

Reduction mod `p` loses information — two knots with different Jones can agree at
all four points — so this is a *fingerprint*, not a certificate of inequivalence.
𝔽₅ is fine for asking whether the operators can satisfy the braid relations at all,
which is the learning question; use a larger prime (97, 251) once that is settled,
since four points over 𝔽₅ give only 625 buckets.

For contrast, carrying the invariant as a real vector runs into a different wall:
the Jones **polynomial** has degree growing with word length, so a 48-letter word
gives on the order of 50 coefficients and they grow. No fixed-size real register
can hold it. That is the argument for evaluating at fixed values of the variable —
and, once you do, for evaluating in a finite field rather than in floats.

`t` is the formal variable of the Jones polynomial: `V_K(t)` is a Laurent
polynomial in `t^{1/2}` with integer coefficients — for the trefoil,
`−t^{−4} + t^{−3} + t^{−1}`. "At one value of `t`" means substituting a specific
number for that variable and carrying the resulting element rather than the
coefficient list, which is what makes a bounded register possible at all: a
48-letter word gives on the order of 50 coefficients, and they grow.

Two practical constraints on choosing `t`:

* **Take `|t| = 1`,** i.e. `t = e^{iθ}`. The scan multiplies ~`L` matrices, so any
  other modulus makes entries grow or decay like `|t|^48` and a float register
  overflows or flattens.
* **Avoid the degenerate values.** `V(1) = (−2)^{c−1}` and `V(i) = (−√2)^{c−1}`
  depend only on the component count, so both are constant on knots;
  `V(e^{2πi/3}) = 1` always. `t = −1` gives the knot determinant — integer-valued
  and a good sanity check. Generic `e^{iθ}` with `θ/π` irrational has the most
  separating power.

Note the strand bound is doing real work here. Exact Jones evaluation is #P-hard in
general (Jaeger–Vertigan–Welsh), but at `n ≤ 5` the Temperley–Lieb algebra is
42-dimensional and the whole thing costs `O(L · d²)` — milliseconds.

## 4. Zero human knowledge: what counts, and what already violates it

The experiment is unknotting **with zero human knowledge**, so handing the network
computed invariants is not a fair arm. That standard has a consequence worth
stating, because it is currently being applied unevenly.

AlphaZero's line is between the *rules* of the game and *heuristics* about it.
Rules are given; heuristics must be learned. Audited against that line, the
observation in `env.py` is:

| channel | class | fair? |
|---|---|---|
| letter one-hots, padding plane | raw state | yes |
| phase, strand count, budget, word length, crossing changes so far | raw state | yes |
| `log(A/B)` | task specification | yes |
| Reidemeister / Markov move set | rules of the game | yes |
| **`top_generator` plane** | derived predicate | **no** |
| **`top_count`, `top_count == 1`** | derived predicate | **no** |

Those last two exist because destabilisation legality is a *global* predicate that
a receptive field of 11 cannot evaluate — the comment in `env.py` says so. They are
the same class of object as a polynomial feature: a human-computed answer to a
question the network was supposed to work out. Three channels, and they cover the
move that is 54% of optimal solutions.

So the zero-knowledge baseline has to ablate them, and the honest ordering is:

1. **Fair, and the real experiment.** A learned register (§3). Mechanisms are not
   knowledge: a writable tape, extra head strides, a wider window, macros mined
   from the agent's own traces — all fair, because they change what the agent *can
   do*, not what it is *told*.
2. **Oracle arms, reported as such.** Computed invariants, per-crossing `Δv_n`
   channels ([10 §3](10-invariants-and-representations.md)), and the three
   destabilisation channels. Their job is to **bound the upside before the fair
   version is built**: if handing the agent the invariant does not help, a learned
   register that has to derive it certainly will not. Run them, quote them as
   oracles, never as the headline.
3. **Outside the frame entirely.** Certified lower bounds (`|σ|/2`, `|s|/2`, `|τ|`)
   used for branch-and-bound. These are not features the agent learns from; they
   are how a search result becomes a theorem. Zero-knowledge constrains what goes
   *into* the network, not what verifies its output.

The channel ablation in
[11 §"The falsifiable version"](06-network-growth.md) therefore does double
duty: it tests whether depth growth buys receptive field, and it is the
zero-knowledge audit.

## 5. Reusing this for knot equivalence

**Markov's theorem already makes the environment an equivalence-move graph.** Two
braid closures are the same link iff their words are connected by conjugation and
(de)stabilisation — the existing move set with `CROSSING_CHANGE` disabled.
Unknotting is the special case `target = empty word in B₁`.

What changes is small: the terminal predicate goes from "word empty and `n = 1`" to
"word equals target up to the relevant equivalence" (`equal_up_to_rotation` already
exists in `reference.py`), `CROSSING_CHANGE` is masked off, and pairs are generated
by applying `k` random Markov moves to a source word — so `k` is a known upper
bound on the distance, exactly the grading `GradedGenerator` already does.

The value function must be conditioned on the target. In the parallel formulation
that doubles the observation channels. In the serial formulation it means **two
heads, one per word: a two-tape machine.** That is where this formulation's
advantage stops being incidental.

**The limit is structural.** The agent can *certify equivalence* by exhibiting the
moves and can **never** certify inequivalence: Markov's theorem bounds no path
length, intermediate words may need to grow past `max_len`, and so failure is
evidence of nothing. Inequivalence needs invariants — which is §3 and §4 again.

Which is the point to end on: **the invariant work and the equivalence work are one
project.** Search produces the positive certificates, invariants the negative ones,
and a system that answers "are these the same knot?" needs both halves.
