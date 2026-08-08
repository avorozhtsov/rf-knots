# rf-knots

> **Continuing this work?** Start with [what this project is](#what-this-project-is) below, then
> [docs/rungs.md](docs/rungs.md) for what the curriculum is made of. The operational state of the
> training runs — what is running, on which machine, and what is next — lives in
> `../pgx-mcts-bench/HANDOFF.md`, next to the runs it describes.

Braid-word unknotting environments for [Pgx](https://github.com/sotetsuk/pgx), built as the
foundation for a population self-play system where agents both *propose* and *solve* problems
from a fixed, machine-verifiable family.

## What this project is

**The task.** Every knot can be written as a braid word. Untying it means rewriting
that word to the empty one, using the braid-group and Markov moves — and where those
are not enough, paying for a **crossing change**, the move that cuts a strand and
passes it through. The fewest crossing changes any sequence needs is the knot's
**unknotting number** `u(K)`, an open invariant for most knots past ten crossings.

**Why it is a good RL problem.** Instances are generated *from* the answer: scramble
the unknot with `K` legal moves and you have a problem whose solution you already
know, at any difficulty, for free and without a labelled dataset. Every move
preserves the knot type, so a solution is machine-checkable and no reward model is
needed. On torus knots `u` is a theorem, so an agent's answer can be scored against
truth rather than against another agent.

**The research question**, which is not "can a network untie knots" — it can:
*where do hard instances come from, and does training against a proposer that
searches for them transfer to instances no random generator would emit?* That is
the claim [research/09](research/09-vs-learning-to-unknot.md) argues is worth
making, and it is why the project is a propose/solve league rather than a solver.

**What is here versus next door.** This repository is the mathematics and the
environment: the encoding, the legal moves, the instance generator, knot
invariants, and the research notes. Training, search and the curriculum ladder
live in [pgx-mcts-bench](https://github.com/avorozhtsov/pgx-mcts-bench), which
depends on this repository by path; its `HANDOFF.md` is the operational state of
the training runs.

The design study is in [research/](research/); start with [research/README.md](research/README.md).

## The environment

`BraidUnknot` is a two-player, zero-sum game:

```
phase 0 (Scrambler, K plies)   start from the empty 1-braid, whose closure is the unknot.
                               apply K type-preserving moves. the closure is STILL the unknot.
phase 1 (Simplifier, M plies)  sees only the resulting word, not the move history.
                               wins iff it reaches the empty 1-braid.
payoff                         +1 / -1, zero-sum.
```

Ground truth is exact and free at every difficulty, because instances are generated *from* the
answer rather than labelled after the fact. `K` is the difficulty dial.

A braid word on `n` strands is stored as a left-compacted `int32[L]` array: `+g` is
sigma_g, `-g` is sigma_g^-1, `0` is padding. Moves are the braid-group relations (free
reduction/insertion, far commutation, the braid relation) plus the Markov moves (conjugation by
cyclic rotation, and (de)stabilisation). The crossing change is implemented and masked off during
the scramble phase; it is the unknotting move for the later unknotting-number mode.

## Install

```bash
uv sync --extra dev --python 3.12
```

## Use

Action-space layout and shapes:

```bash
uv run rf-knots rules --tier tier1
```

Verify the invariants that make generated instances trustworthy — JAX kernels against a
list-based reference, closure components, and exact braid-group equality via the Artin
representation:

```bash
uv run rf-knots selfcheck --tier tier0 --episodes 200
```

Scramble from the unknot and solve it exactly by breadth-first search:

```bash
uv run rf-knots demo --tier tier0 --seed 0
```

Throughput and the random-play baseline a trained agent has to beat:

```bash
uv run rf-knots bench --tier tier1 --batch 1024
```

How hard a K-move scramble really is, measured by exact optimal solution depth:

```bash
uv run rf-knots calibrate --tier tier0 --samples 64
```

Identify the knot a braid word closes to, and compute its invariants:

```bash
uv run rf-knots knot "1,1,2,2,1,1,2,1,-2,-1,2,1,-2,-2,-1,2,1,-2"
```

That word is a rung of the training ladder, recorded there as eighteen crossings
with an unknown unknotting number. It is the seven-crossing knot `7_5`, whose
unknotting number has been 2 since the knot tables. Every rung is worked out in
[docs/rungs.md](docs/rungs.md), with the invariants in
[docs/rungs-invariants.md](docs/rungs-invariants.md).

The bundled table uses KnotInfo identifiers as canonical names: for example,
`12n_570`. Its committed metadata records the KnotInfo database URL, retrieval
date and snapshot hash, together with the corresponding Spherogram identifier
(`K12n570`) used to obtain each braid. Runtime lookup accepts either spelling.

## Layout

| Path | Contents |
|---|---|
| `src/rf_knots/config.py` | static shape and rule parameters; `TIER0` / `TIER1` presets |
| `src/rf_knots/actions.py` | flat action-space layout, encode/decode |
| `src/rf_knots/braid.py` | JAX kernels: rewrites, legality masks, closure diagnostics |
| `src/rf_knots/env.py` | the Pgx `Env` and `State` |
| `src/rf_knots/generator.py` | source knots and the complexity grade the ladder climbs |
| `src/rf_knots/invariants.py` | Alexander, Jones, determinant, genus and unknotting bounds |
| `src/rf_knots/knot_table.py` | naming a knot, against the bundled table in `data/` |
| `src/rf_knots/reference.py` | slow pure-Python oracle: Artin representation, BFS solver |
| `src/rf_knots/render.py` | ASCII and SVG pictures of a braid and its closure |
| `src/rf_knots/torus.py` | the diagram as a `position × strand` raster: strand-count-agnostic input |
| `src/rf_knots/rollout.py` | random play, scramble generation, batched benchmarking |
| `docs/representation.md` | how a knot is encoded and what may be done to it |
| `docs/rungs.md` | every ladder rung, its rationale, and the knot it really is |
| `docs/lessons.md` | process notes: the mistakes that cost time, and what to do instead |
| `research/13-directions.md` | the big next moves, unscheduled |
| `research/18-raster-representation.md` | why the strand count is the remaining size dependence, and what a picture fixes |
| `research/experiments/` | the parameter-matched representation probes behind note 18 |
| `scripts/` | one-off builders for the committed data files |
| `research/` | design study behind all of this |

## Status

M2 of the roadmap in [research/08-roadmap.md](research/08-roadmap.md). The environment is
complete and tested; training runs in `../pgx-mcts-bench`, which depends on this repo by path.

What is established, on generated torus-knot instances scored against proved unknotting numbers:

* **Exactly optimal crossing changes** — 1.00 against `u(T(2,3)) = 1`, 2.00 against
  `u(T(2,5)) = 2`, 3.00 against `u(T(2,7)) = 3` and `u(T(3,4)) = 3`. Those are theorems
  (Milnor conjecture, Kronheimer–Mrowka), which is the point of using torus knots.
* **A staged ladder discriminates.** Highest stage cleared spans 0 to 9 across candidates.
  Search dominates capacity: 128 simulations reach stage 8, 16 simulations reach stage 0,
  and 7.7× the parameters buys nothing that 2× the simulations does not.
* **The serial (Turing-machine) formulation works** — `O(1)` action space in word length,
  a head the agent must move. It cleared the full ladder and beat the strongest parallel
  candidate on crossing-change optimality, once its policy head was made positional.
  See [research/12-serial-formulation.md](research/12-serial-formulation.md).

* **The ladder's hardest rungs were not what they claimed.** The rungs were graded on the
  length of the braid word they were generated from, which is not an invariant. `R(3,18)#0`
  is the seven-crossing knot `7_5` with `u = 2`; `R(3,22)#0` is the unknot. 19 of the 23
  distinct rung knots now have an exact unknotting number, against 6 before. See
  [docs/rungs.md](docs/rungs.md).

Open, in rough order of value: the remaining certified lower bounds — `|σ|/2` is now computed
per knot in `rf_knots.invariants`, but Rasmussen `|s|/2` and `|τ|` are not, and none of them is
wired into a branch-and-bound, which is what turns a search result into a theorem; a learned
head register; hard unknot diagrams; knot equivalence as a two-tape machine.
