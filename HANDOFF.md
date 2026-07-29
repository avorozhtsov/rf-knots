# Handoff — read this first

Context for continuing the project in a fresh session. Written 2026-07-29.

## What this is

An attempt at RL for knot theory, aimed eventually at a population of agents that
both *propose* and *solve* mathematical problems. It currently does one thing
well: **an AlphaZero agent that unknots braid words, scored against proved
unknotting numbers.**

Two repos, both public, both pushed:

| repo | branch | contents |
|---|---|---|
| [rf-knots](https://github.com/avorozhtsov/rf-knots) | `main` | the environment, the generator, the research notes |
| [pgx-mcts-bench](https://github.com/avorozhtsov/pgx-mcts-bench) | `braid-unknot-env` | training, search, sweeps, the ladder |

`pgx-mcts-bench` depends on `rf-knots` by path (`../rf-knots`), so they must sit
side by side.

Start with, in this order:
1. `rf-knots/docs/representation.md` — how a knot is encoded, the move set, the
   Reidemeister/Markov correspondence
2. `pgx-mcts-bench/docs/braid-training.md` — what is actually being trained
3. `pgx-mcts-bench/docs/ladder-plan.md` — the current experiment and how to resume it
4. `rf-knots/research/` — the design study, and where it was revised by evidence

## What works

**The environment.** Cyclic braid words, 104 tests. Verified four independent
ways on every step of random rollouts: differential against a list-based
reference, exact braid-group equality via the faithful Artin representation, the
seam-move decomposition, and closure component count. `rf-knots selfcheck` runs
them at scale (5792 steps, zero disagreements).

**The solver.** On generated instances it reaches **exactly optimal crossing
changes** — 1.00 against `u(T(2,3)) = 1`, 2.00 against `u(T(2,5)) = 2`, 3.00
against `u(T(2,7)) = 3`. Those are theorems (Milnor conjecture, Kronheimer–
Mrowka), not search results, which is the point of using torus knots.

**The curriculum.** 0.951 ± 0.029 on frozen anchors with zero collapses in 6
seeds, against 0.375 for an untrained control.

## What does not work, and why

**The learned proposer.** Measured over 8 seeds, indistinguishable from a
uniform-random generator (+0.15, 95% CI [−0.06, +0.35]), and *worse* than random
on some initialisations. Replaced by a programmable graded generator. A 3-seed
version of the same measurement said +0.40 with CI [+0.08, +0.72] — that was
seed selection, and it is the reason anything about the Scrambler now needs 8
seeds before it is said out loud.

**The A/B objective is inert on current instances**, and this is a property of
the environment rather than a bug. An exhaustive Pareto check found
`moves[k] = m₀ + k` on every instance tested: a crossing change *is* a move and
never *saves* one, so `λ·k + moves[k] = m₀ + k(λ+1)` is minimised at the smallest
`k` for every positive `λ`. The three ratios should produce the same policy.
**The open question** is whether hard unknot diagrams break this — `u = 0` but
needing many moves and a temporary increase in crossing number, where one
crossing change might collapse a tangle worth twenty Reidemeister moves. The
front-check code exists; run it on those.

## Running now

A ladder run: 9 candidates × 10 stages, `artifacts/ladder-run/`. Checkpointed
after every cleared stage, so this resumes it:

```bash
cd pgx-mcts-bench
uv run pgx-mcts-bench braid-ladder --max-iterations 25 --eval-games 12 \
    --promote-at 0.8 --workers 5 --output artifacts/ladder-run
```

Early signal: `search-heavy` cleared all 10 stages in 164s. **The ladder is
probably too easy to discriminate** and likely needs extending — larger `p` in
`T(p,q)`, or more scramble depth. Check `ladder.md` before drawing conclusions.

## What to do next

1. **Read the ladder result.** If several candidates reach stage 9, extend the
   stages before comparing anything.
2. **The serial (Turing-machine) candidates are the point of that run.** Action
   space `O(1)` in word length rather than `O(L)`; three bugs in their path have
   been fixed and none of them has yet produced a number.
3. **Hard unknot diagrams** — the outstanding question above, and also where the
   science is (they are what DeepMind's 2.6M-diagram corpus is made of).
4. **Certified lower bounds** (`|σ|/2`, `|s|/2`, `|τ|`) with branch-and-bound.
   This is the step that turns output into theorems for knots where `u` is
   *unknown*, rather than confirming ones where it is known.
5. Macros: mine frequent subsequences from solution traces, promote them to
   actions, retrain. A macro is real iff the objective drops.

## Process notes that were learned the hard way

- **8 seeds minimum** for anything about instance difficulty. 3 gave a false
  positive that survived two rounds of reporting.
- **Time each phase separately** and project the total from the first completed
  job. Two invisible bugs (an 18× over-generous BFS bound, and 42 runs
  recomputing identical values) cost ~25 minutes twice before per-phase timing
  existed.
- **Never pipe a background run through `tail`** — it buffers everything until
  exit. Nine minutes of zero output came from that.
- **Check the instance family has dynamic range before building machinery for
  it.** The Pareto check takes seconds and would have prevented the entire
  conditioning apparatus being built for a degenerate objective.
- Negative results belong in commit messages, not in dead code. Roughly 580 lines
  were removed once measurements killed the options they implemented.
