# Handoff — read this first

Context for continuing the project in a fresh session. Written 2026-07-29.

## What this is

An attempt at RL for knot theory, aimed eventually at a population of agents that
both *propose* and *solve* mathematical problems. It currently does one thing
well: **an AlphaZero agent that unknots braid words, scored against proved
unknotting numbers.**

The experiment is **unknotting with zero human knowledge**. That standard decides
which of the open options are fair arms and which are oracle arms — see
`research/12-serial-formulation.md` §4, which also notes that three channels of the
current observation already violate it.

Two repos, both public, both pushed:

| repo | branch | contents |
|---|---|---|
| [rf-knots](https://github.com/avorozhtsov/rf-knots) | `main` | the environment, the generator, the research notes |
| [pgx-mcts-bench](https://github.com/avorozhtsov/pgx-mcts-bench) | `serial-fixes` | training, search, sweeps, the ladder |

`pgx-mcts-bench` depends on `rf-knots` by path (`../rf-knots`), so they must sit
side by side. **Note the branch has moved** from `braid-unknot-env` to
`serial-fixes`.

Start with, in this order:
1. `rf-knots/docs/representation.md` — how a knot is encoded, the move set, the
   Reidemeister/Markov correspondence
2. `rf-knots/research/12-serial-formulation.md` — the most recent findings, and
   the zero-knowledge scope
3. `pgx-mcts-bench/docs/braid-training.md` — what is actually being trained
4. `rf-knots/research/` — the design study, and where it was revised by evidence
   (the README lists the revisions)

## What works

**The environment.** Cyclic braid words, 104 tests. Verified four independent
ways on every step of random rollouts: differential against a list-based
reference, exact braid-group equality via the faithful Artin representation, the
seam-move decomposition, and closure component count. `rf-knots selfcheck` runs
them at scale (5792 steps, zero disagreements).

**The solver, both formulations.** Exactly optimal crossing changes against
theorems: 1.00 for `u(T(2,3)) = 1`, 2.00 for `u(T(2,5)) = 2`, 3.00 for
`u(T(2,7)) = 3` and `u(T(3,4)) = 3`.

**The ladder discriminates.** Highest stage cleared spans 0 to 9. The sharpest
result across the whole project: **search dominates capacity.** 128 simulations
reach stage 8, 16 reach stage 0, and 7.7× the parameters (`wide-net`, 372K) buys
nothing that 2× the simulations (`search-heavy`, 48K) does not.

**The serial (Turing-machine) formulation.** `O(1)` action space in word length.
It scored 0 on the first ladder and 9 of 9 on the second; the difference was a
readout bug, not the formulation. It beats the strongest parallel candidate on
crossing-change optimality at stage 8 (3.00 against 4.00, optimum 3).

## What does not work, and why

**The learned proposer.** Over 8 seeds, indistinguishable from a uniform-random
generator (+0.15, 95% CI [−0.06, +0.35]), and *worse* than random on some
initialisations. Replaced by a programmable graded generator. A 3-seed version of
the same measurement said +0.40 with CI [+0.08, +0.72] — that was seed selection,
and it is why anything about the Scrambler now needs 8 seeds.

**The A/B objective is inert in the parallel formulation** — and this turned out to
be formulation-specific rather than a property of the domain. Parallel candidates
emit the same policy at both ends of `log(A/B)` to two decimal places, exactly as
the Pareto argument predicted (`moves[k] = m₀ + k`, minimised at the smallest `k`
for every λ). Serial candidates respond by **5–6× in moves**, because head travel
is charged to the same budget the metric reads. Caveat: that makes serial `moves`
edits *plus* travel, so the two numbers are not the same quantity.

**GPU rental would not help yet.** Measured: MCTS is batch-1 latency bound, 605 µs
per simulation of which 463 µs is the forward pass on a 48K-parameter network. A
GPU's kernel-launch overhead makes that worse. Batch the leaf evaluations across
parallel games first — 7.8× on this laptop, measured. Then rented CPU is the cheap
win (~$85 for a sweep that takes this machine two weeks).

## Running now

`pgx-mcts-bench artifacts/serial-screen/` — six serial arms over the *old*
ten-stage ladder, 4 of 6 complete. Results so far, highest stage cleared:
`s-window-128` 9, `s-head-128` 8, `s-head-1stride` 8, `s-head-budget96` 8,
`s-head-256` ≥8, `s-w11-128` ≥7. It finishes on its own; nothing to intervene on.

Its numbers are comparable to `artifacts/ladder-run/ladder.md` because both used
the same stages and promotion rule. **The stage list and the promotion rule have
since changed**, so anything run from here is not directly comparable to either —
resume handles it (see below) but the tables do not.

## What changed in the curriculum, and why it matters for resuming

Four changes, all driven by one measurement: re-evaluating a finished candidate's
weights on the rungs it had already passed. `s-window-128` promoted `T(2,3)+4` at
4.18 crossing changes against an optimum of 1, and eight stages later — having
never trained there again — still scored 1.17. Transfer from harder stages improves
the easier ones substantially and converges none of them.

1. **Training mixes the cleared stages** (`--mix-decay`, geometric back from the
   frontier; 0 reproduces the old frontier-only rule). Evaluation stays pinned.
2. **Promotion has two exits**, recorded separately: reaching `u(K) + tolerance`,
   or plateauing on it. Solve rate alone measured feasibility, not the objective.
3. **Every promotion re-evaluates the rungs below** and logs regressions.
4. **17 stages, graded finely in scramble and coarsely in `u`.** Every `+0` stage
   promoted in 2 iterations at exactly the proved number; every `+4` overshot. The
   knot is nearly free and the scramble is the difficulty. `T(2,7)` is dropped
   (`u = 3`, same as the harder `T(3,4)`); `+2` and `+8` rungs added.

**Resume is keyed on stage identity `(source, scramble)`, not index.** Inserting a
rung silently repoints every index-keyed checkpoint at a different stage. Old
checkpoints resume correctly and restart at the first newly-inserted rung they have
not actually done.

**Weights are snapshotted either side of every stage** at
`checkpoints/<candidate>/stage<NN>-{before,after}.pt`, which is what makes the
retrospective analysis above possible at all.

To continue, with both formulations racing on the 17-stage ladder:

```bash
cd pgx-mcts-bench
uv run pgx-mcts-bench braid-ladder --max-iterations 60 --eval-games 12 \
    --workers 6 --output artifacts/ladder-17
```

Seed its checkpoint directory from `artifacts/serial-screen/checkpoints/` and
`artifacts/ladder-run/checkpoints/` to continue rather than restart.

## What to do next

1. **Certified lower bounds** (`|σ|/2`, `|s|/2`, `|τ|`) with branch-and-bound. This
   is the step that turns output into theorems for knots where `u` is *unknown*.
   Unaffected by the zero-knowledge constraint — bounds verify output, they are not
   features. Highest value of anything on this list.
2. **Batch the MCTS leaf evaluations.** 7.8× measured, no rental needed, and it is
   the prerequisite for a GPU ever helping.
3. **A learned head register.** The serial formulation was proposed with a memory in
   the head and it was never built, so every serial number so far is a floor. It is
   also what makes a knot invariant *representable* at all — a memoryless head
   cannot compute a Markov trace, which is a theorem rather than a training problem.
   Sizing and where it has to live: `research/12-serial-formulation.md` §3.
4. **Knot equivalence as a two-tape machine.** Markov's theorem already makes the
   environment an equivalence-move graph; the change is small and reuses the head
   machinery. `research/12-serial-formulation.md` §5.
5. **Hard unknot diagrams** — `u = 0` but needing many moves and a temporary
   increase in crossing number. Where the science is, and what DeepMind's
   2.6M-diagram corpus is made of.
6. **Network growth**, but only after establishing a capacity-bound regime exists.
   Plan, including what would kill the branch: `research/11-network-growth-branch.md`.

## Process notes that were learned the hard way

- **8 seeds minimum** for anything about instance difficulty. 3 gave a false
  positive that survived two rounds of reporting.
- **Read the trained checkpoint, not the design doc.** Every one of the three
  serial defects was found by probing weights and action histograms. None was
  visible in the code's intent, and all three were described correctly in the
  module docstring as things that had been handled.
- **Extract the tricky logic into pure functions and test it.** Doing that to the
  resume and promotion rules immediately caught two bugs that produced plausible
  numbers rather than errors.
- **A promotion threshold makes the metric at promotion nearly meaningless.** It
  reports where the network happened to be when it crossed the bar. Snapshot both
  sides or the trajectory is unrecoverable.
- **Time each phase separately** and project the total from the first completed
  job. Two invisible bugs (an 18× over-generous BFS bound, and 42 runs recomputing
  identical values) cost ~25 minutes twice before per-phase timing existed.
- **Never pipe a background run through `tail`** — it buffers everything until
  exit. Nine minutes of zero output came from that.
- **Check the instance family has dynamic range before building machinery for it.**
  The Pareto check takes seconds and would have prevented the entire conditioning
  apparatus being built for what turned out to be a degenerate objective — in one
  formulation.
- Negative results belong in commit messages, not in dead code. Roughly 580 lines
  were removed once measurements killed the options they implemented.
