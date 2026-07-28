# rf-knots

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

M1 of the roadmap in [research/08-roadmap.md](research/08-roadmap.md). The environment is
complete and tested; training (M2) reuses `../pgx-mcts-bench` with an environment swap.
