# Handoff — read this first

Context for continuing the project in a fresh session. Written 2026-07-30.

## What this is

An attempt at RL for knot theory, aimed eventually at a population of agents that
both *propose* and *solve* mathematical problems. It currently does one thing
well: **an AlphaZero agent that unknots braid words, scored against proved
unknotting numbers.**

The experiment is **unknotting with zero human knowledge**. That standard decides
which options are fair arms and which are oracle arms — see
`research/12-serial-formulation.md` §4, which also notes that three channels of the
current observation already violate it.

Two repos, both public. `pgx-mcts-bench` depends on `rf-knots` by path
(`../rf-knots`), so they must sit side by side.

| repo | branch | contents |
|---|---|---|
| [rf-knots](https://github.com/avorozhtsov/rf-knots) | `main` | the environment, the generator, the research notes |
| [pgx-mcts-bench](https://github.com/avorozhtsov/pgx-mcts-bench) | `serial-fixes` | training, search, sweeps, the ladder |

Start with, in this order:
1. `rf-knots/docs/representation.md` — how a knot is encoded, the move set, the
   Reidemeister/Markov correspondence
2. `rf-knots/research/12-serial-formulation.md` — the most recent findings, the
   head-register question, and the zero-knowledge scope
3. `pgx-mcts-bench/docs/braid-training.md` — what is actually being trained
4. `rf-knots/research/README.md` — the design study, with a list of the positions
   that measurements have since revised

## What works

**The environment.** Cyclic braid words, 104 tests. Verified four independent ways
on every step of random rollouts: differential against a list-based reference,
exact braid-group equality via the faithful Artin representation, the seam-move
decomposition, and closure component count. `rf-knots selfcheck` runs them at
scale (5792 steps, zero disagreements).

**The solver, both formulations.** Exactly optimal crossing changes against
theorems: 1.00 for `u(T(2,3)) = 1`, 2.00 for `u(T(2,5)) = 2`, 3.00 for
`u(T(2,7)) = 3` and `u(T(3,4)) = 3`.

**Search dominates capacity.** The sharpest result in the project: 128 simulations
reach the top of the old ladder, 16 simulations reach stage 0, and 7.7× the
parameters (`wide-net`, 372K) buys nothing that 2× the simulations
(`search-heavy`, 48K) does not. Any capacity work has to establish a
capacity-bound regime first — `research/11-network-growth-branch.md`.

**The serial (Turing-machine) formulation.** `O(1)` action space in word length.
It scored 0 on the first ladder and cleared the whole thing on the second; the
difference was a readout bug, not the formulation.

## Where the numbers are

Reports regenerate from their own JSON and describe **the rungs that run actually
saw**, so an old report labels itself historical rather than being relabelled with
today's stage list. Regenerate any of them with:

```bash
uv run pgx-mcts-bench braid-ladder-merge artifacts/<run>
```

| artifact | rungs | what it is |
|---|---:|---|
| `artifacts/ladder-run/` | 10 | first ladder. **Historical** — every serial candidate scored 0 here, pre-fix |
| `artifacts/serial-screen/` | 10 | the serial grid after the readout fix. Comparable to `ladder-run` by rung name |
| `artifacts/ladder-17/` | 17 | current ladder, 8 arms, resumed |
| `artifacts/memory-run/` | 17 | the written-register arms, fresh |
| `artifacts/central-benchmark/` | 17 | **running now**: 7 arms including the sequence encoders |

### 17-stage ladder (`artifacts/ladder-17/ladder.md`)

| candidate | top rung | reached |
|---|---:|---|
| `s-window-128` (serial), `search-heavy`, `u1-puct`, `wide-net` | **16** | `T(3,5)+4` |
| `s-head-128` | 11 | `T(3,4)+4` |
| `s-head-1stride` | 10 | `T(3,4)+2` |
| `s-head-budget96` | 9 | `T(3,4)+0` |
| `u3-uct` | 8 | `T(2,5)+8` |

**Four arms cleared all 17 rungs, so the ladder is saturated at the top again and
needs extending before it can discriminate further.**

## What does not work, and why

**The learned proposer.** Over 8 seeds, indistinguishable from a uniform-random
generator (+0.15, 95% CI [−0.06, +0.35]). Replaced by a programmable graded
generator. A 3-seed version said +0.40 with CI [+0.08, +0.72] — that was seed
selection, and it is why anything about the Scrambler needs 8 seeds.

**Agent-written head registers — a clean negative.** `serial_registers` gives the
head K binary registers with one TOGGLE action each: the finite control state a
scanning head is missing, written by the agent, so no gradient and no BPTT.

```
[s-reg8] stage 1 unknot+6: solved 0.50 after 60 it (capped)
[s-reg4] stage 1 unknot+6: solved 0.83 after 36 it (plateau)
```

`s-reg8` collapsed onto the exact rung where the *pre-fix* serial candidates died.
`s-reg4` cleared it but needed 36 iterations where the matched no-register arm
needed 12, and reached rung 7 fresh where `s-head-128` reached the equivalent of 9.
**Cost is monotone in register count.** The mechanism is action-space dilution: a
TOGGLE never changes the word, so 8 of 34 actions are branches MCTS cannot make
progress on. The lesson generalises — *a register that nothing reads is noise*, and
memory needs a mechanism that makes it pay.

**The A/B objective is inert in the parallel formulation** — a property of the
formulation, not the domain. Parallel candidates emit the same policy at both ends
of `log(A/B)` to two decimal places, as the Pareto argument predicted. Serial
candidates respond by 5–6× in moves, because head travel is charged to the budget
the metric reads. Caveat: that makes serial `moves` edits *plus* travel, so the two
numbers are not the same quantity.

**Climbing does not fully protect the rungs below.** Stage mixing at `decay=0.5`
reduces the forgetting-shaped residual but does not remove it — the regressions
table in `artifacts/ladder-17/ladder.md` shows `T(3,4)+8` failing for two arms
after they climb past it.

**GPU rental would not help yet.** MCTS is batch-1 latency bound: 605 µs per
simulation, 463 µs of it the forward pass on a 48K-parameter network. Kernel-launch
overhead makes a GPU worse. Batch the leaf evaluations first (7.8× measured on this
laptop), then rented CPU is the cheap win — ~$85 for a sweep that takes this machine
two weeks.

## Running now

```bash
uv run pgx-mcts-bench braid-ladder \
  --only s-head-128,s-reg4,s-reg8,s-gru128,s-fsa32,s-ff4-p5,s-burau-oracle \
  --max-iterations 25 --eval-games 16 --retro-games 6 --workers 4 \
  --output artifacts/central-benchmark
```

Seven arms on one action space and one search budget, differing only in what
accumulates over the tape. The four encoder arms receive a head-relative full-tape
scan instead of a window:

| arm | accumulator | fair or oracle |
|---|---|---|
| `s-head-128` | none (window only) | fair — the baseline |
| `s-reg4`, `s-reg8` | agent-written binary registers | fair |
| `s-gru128` | unconstrained GRU-128 | fair |
| `s-fsa32` | learned 32-state soft automaton | fair |
| `s-ff4-p5` | learned 4×4 matrices over 𝔽₅ | fair |
| `s-burau-oracle` | fixed Burau at `t = −1, 1/2` | **oracle** — known algebra enters the model |

`SequenceBraidNet.regularization_loss` penalises violations of the braid relations
(`σᵢσᵢ⁻¹ = 1`, `σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁`, far commutation), which is what forces the
learned operators to be an actual braid-group representation rather than an
arbitrary recurrence.

Measured cost, batch-1 forward — the quantity MCTS is bound by:

| arm | forward | vs window-only | 128-sim decision |
|---|---:|---:|---:|
| `s-head-128` | 687 µs | 1.0× | 88 ms |
| `s-gru128` | 867 µs | 1.3× | 111 ms |
| `s-fsa32` | 2,014 µs | 2.9× | 258 ms |
| `s-ff4-p5` | 2,164 µs | 3.2× | 277 ms |
| `s-burau-oracle` | 2,872 µs | 4.2× | 368 ms |

## What to do next

1. **Extend the ladder.** Four arms cleared all 17 rungs. Next rungs by `u`:
   `T(2,11)` (u=5), `T(2,13)`/`T(3,7)` (u=6). Raise `generator_max_crossings` from
   10 to 16; word lengths stay well inside `max_len = 48`.
2. **Certified lower bounds** (`|σ|/2`, `|s|/2`, `|τ|`) with branch-and-bound. The
   step that turns output into theorems where `u` is *unknown*. Unaffected by the
   zero-knowledge constraint — bounds verify output, they are not features.
   Highest value on this list.
3. **Jones needs a bigger carrier than `s-ff4-p5` has.** `TL₅` has dimension
   Catalan(5) = 42 and decomposes into irreducibles of dimension **1, 4, 5**
   (1² + 4² + 5² = 42). The 4-dimensional block *is* the reduced Burau
   representation, so a 4×4 carrier reaches Alexander/Conway and one of Jones's
   three blocks. Jones needs all three: add a 5×5 arm and a block-diagonal
   1⊕4⊕5 arm. Four evaluation points = four choices of `A ∈ 𝔽ₚ`, so 4 × 42 field
   elements. Require `A ≠ 0` and `δ = −A² − A⁻² ≠ 0`, and avoid characteristic 2.
   𝔽₅ is fine for learning; use a larger prime for fingerprint strength.
4. **Batch the MCTS leaf evaluations.** 7.8× measured, no rental needed.
5. **Knot equivalence as a two-tape machine.** Markov's theorem already makes the
   environment an equivalence-move graph; the change is small and reuses the head.
   No delimiter is needed — two cyclic words plus one tape-select register.
   `research/12-serial-formulation.md` §5.
6. **Hard unknot diagrams** — `u = 0` but needing many moves and a temporary
   increase in crossing number. Where the science is.

## Process notes that were learned the hard way

- **8 seeds minimum** for anything about instance difficulty. 3 gave a false
  positive that survived two rounds of reporting.
- **Read the trained checkpoint, not the design doc.** All three serial defects
  were found by probing weights and action histograms. None was visible in the
  code's intent, and all three were described in the module docstring as handled.
- **Never let a report hardcode what it is reporting.** The ladder header said "Ten
  stages … capped at 25 iterations" through two changes to the stage list and one
  to the promotion rule. Reports now derive their description from the rungs the
  results recorded, so an old run labels itself historical instead of being
  silently relabelled with today's ladder.
- **Extract tricky logic into pure functions and test it.** Doing that to the
  resume and promotion rules immediately caught two bugs that produced plausible
  numbers rather than errors.
- **A promotion threshold makes the metric at promotion nearly meaningless.** It
  reports where the network happened to be when it crossed the bar. Weights are now
  snapshotted either side of every rung.
- **Resume on stage identity, not index.** Inserting a rung silently repoints every
  index-keyed checkpoint at a different stage, with weights that never saw it.
- **Time each phase separately** and project the total from the first completed
  job.
- **Never pipe a background run through `tail`** — it buffers until exit.
- **Check the instance family has dynamic range before building machinery for it.**
- Negative results belong in commit messages and in the notes, not in dead code.
