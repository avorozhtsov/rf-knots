# rf-knots

> **Continuing this work?** Read [HANDOFF.md](HANDOFF.md) first.

Braid-word unknotting environments for [Pgx](https://github.com/sotetsuk/pgx), built as the
foundation for a population self-play system where agents both *propose* and *solve* problems
from a fixed, machine-verifiable family.

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

## Layout

| Path | Contents |
|---|---|
| `src/rf_knots/config.py` | static shape and rule parameters; `TIER0` / `TIER1` presets |
| `src/rf_knots/actions.py` | flat action-space layout, encode/decode |
| `src/rf_knots/braid.py` | JAX kernels: rewrites, legality masks, closure diagnostics |
| `src/rf_knots/env.py` | the Pgx `Env` and `State` |
| `src/rf_knots/reference.py` | slow pure-Python oracle: Artin representation, BFS solver |
| `src/rf_knots/rollout.py` | random play, scramble generation, batched benchmarking |
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

Open, in rough order of value: certified lower bounds (`|σ|/2`, `|s|/2`, `|τ|`) with
branch-and-bound, which is what turns a search result into a theorem; a learned head register;
hard unknot diagrams; knot equivalence as a two-tape machine.
