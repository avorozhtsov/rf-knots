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

## 3. The gap: there is no register

The formulation was proposed with a **memory in the head** — an embedding, order
128 floats. That was never built. The state is `(pgx_state, head)` where `head` is
an integer index, and the observation is a slice of the word. Nothing is carried
between plies. It is a scan machine with no accumulator.

So everything in §2 is a **floor**: the serial formulation cleared the ladder and
beat the parallel nets on crossing-change optimality *without* the memory that
motivated it.

Two consequences.

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

### Sizing, corrected

To *compute* an invariant by scanning, the register must hold the running algebra
element, not the final value. For `max_strands = 5`:

| object | dimension | floats (complex) | fits in 128? |
|---|---:|---:|---|
| reduced Burau (→ Alexander) at one value of `t` | 4 × 4 | 32 | yes, four values fit exactly |
| Temperley–Lieb (→ Jones), `dim TL₅ = Catalan(5) = 42`, one `t` | 42 | 84 | yes, one value |
| Jones at two values of `t` | — | 168 | no; needs ~256 |
| Jones as a Laurent **polynomial** | degree grows with `L` | unbounded | never |

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
[11 §"The falsifiable version"](11-network-growth-branch.md) therefore does double
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
