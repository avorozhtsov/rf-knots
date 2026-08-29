# rf-knots

**Learning representations of knots by teaching neural networks to transform
and compare them.**

RF Knots is the research home for neural knot manipulation. The current system
trains agents to unknot braid representations; the longer-term goal is to learn
models that can also:

- establish that two representations describe the same knot;
- estimate **Gordian distance**, the minimum number of crossing changes (CCs)
  needed to pass from one knot type to another; and
- produce embeddings from which both crossing-change distance and
  representation distance can be recovered.

Here, representation distance means the number of zero-CC transformations
between two diagrams or braid words: Reidemeister moves, braid relations,
Markov moves, and stabilization/destabilization. It is deliberately kept
separate from Gordian distance, because changing a diagram without changing
its knot type is not the same operation as changing the knot itself.

The repository provides the mathematics, encodings, legal moves, invariant
checks, generators, and [Pgx](https://github.com/sotetsuk/pgx) environments.
MCTS training loops and large policy runs live in
[pgx-mcts-bench](https://github.com/avorozhtsov/pgx-mcts-bench), which depends
on this package. Full-representation embedding models and their metric-training
loop live here because they are part of the RF Knots model architecture.

> **Continuing this work?** Read [docs/rungs.md](docs/rungs.md) for the knot
> curriculum. The operational state of training runs lives in
> `../pgx-mcts-bench/HANDOFF.md`, next to the code and artifacts it describes.

## The three-project system

| Repository | Responsibility |
|---|---|
| **rf-knots** | Knot representations, legal transformations, invariants, environments, model-facing features, and the neural-knot research agenda. |
| [**pgx-mcts-bench**](https://github.com/avorozhtsov/pgx-mcts-bench) | MCTS games and training infrastructure, including the braid objectives `L10` and `L1000`. |
| [**unknotdb**](https://github.com/avorozhtsov/unknotdb) | A replayable proof graph of representations, witnesses, invariants, and sidecar data. It uses RF Knots models to preprocess representations and exports graph-derived supervision for future models. |

The intended feedback loop is:

```text
RF Knots model + pgx-mcts-bench search
                    │
                    ▼
          replay-verified witnesses
                    │
                    ▼
             unknotdb proof graph
                    │
                    ▼
       graph-derived training examples
                    │
                    └──────────────► next RF Knots model
```

## Graph-derived embedding training

The repository owns the full-representation encoder, the crossing-change and
diagram-distance objectives, augmentation, streaming/materialized data loaders,
training, checkpoints, and evaluation. Unknot DB remains a read-only evidence
source during training: it exports versioned SQLite sidecars and never owns
model or optimizer state.

Install the optional Torch dependency and run the canonical trainer against
explicitly pinned sidecars:

```bash
uv sync --extra embeddings --extra dev --python 3.12

uv run python scripts/train_embedding_metric.py \
  --pairs ../unknotdb/outputs/unknotdb-embedding-pairs-v4.sqlite \
  --geodesic ../unknotdb/outputs/unknotdb-embedding-geodesic-pairs-v3.sqlite \
  --hard-cases ../unknotdb/outputs/unknotdb-embedding-hard-cases-v4.sqlite \
  --gordian artifacts/gordian-fks-v0/gordian-supervision-v0.sqlite \
  --output artifacts/embedding-run/report.json \
  --checkpoint artifacts/embedding-run/checkpoint.pt \
  --device mps --batch-size 512 --steps 1954
```

Every run records the exact source hashes, model and optimizer state, seed,
throughput, learning curve, and protected metrics. Generated sidecars,
checkpoints, and reports remain under ignored `artifacts/`; only code and
versioned contracts belong in Git.

The current implementation includes an infinite deterministic pair stream,
replayable zero-CC augmentation, compact pre-materialized shards, Gordian-bound
supervision, and a graph-derived validation/quarantine protocol. Validation and
test identities are isolated together with their known CC=0 closure and
one-CC graph neighborhood; unresolved actions and interval upper bounds are not
silently converted into exact labels.

The first one-million-example Apple MPS calibrations prove that the pipeline
trains at roughly 500--600 pairs/s, but they do **not** yet establish a useful
hard-knot embedding. Aggregate validation loss improved while the protected
Wang--Zhang and long-witness cases remained largely collapsed. No checkpoint is
promoted on that evidence. The next experiment is a fixed-validation,
checkpoint-continuation tournament in which several branches consume separate
deterministic million-pair streams and only the best complete checkpoint
continues.

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
| `src/rf_knots/seifert.py` | certified lower bounds: signature, tau, and the Montesinos `u ≥ 2` obstruction |
| `src/rf_knots/unknot_search.py` | branch-and-bound over crossing changes, pruned by those bounds |
| `src/rf_knots/certified_value.py` | the certified bound as a floor on the objective, for clamping a value head |
| `src/rf_knots/verified_bounds.py` | a bounds ratchet that stores the move sequence and replays it on read |
| `src/rf_knots/rollout.py` | random play, scramble generation, batched benchmarking |
| `src/rf_knots/embedding_encoder.py` | variable-length full-braid encoder with separate knot, isotopy, and layout heads |
| `src/rf_knots/embedding_metric.py` | interval-aware crossing-change and diagram-distance objective |
| `src/rf_knots/embedding_augmentation.py` | infinite replayable pair stream and zero-CC augmentation |
| `src/rf_knots/embedding_shards.py` | compact resumable materialized training shards |
| `scripts/train_embedding_metric.py` | canonical embedding training entry point |
| `scripts/build_embedding_validation_protocol.py` | graph-neighborhood validation and test quarantine |
| `docs/representation.md` | how a knot is encoded and what may be done to it |
| `docs/rungs.md` | every ladder rung, its rationale, and the knot it really is |
| `docs/lessons.md` | process notes: the mistakes that cost time, and what to do instead |
| `research/13-directions.md` | the big next moves, unscheduled |
| `research/18-raster-representation.md` | why the strand count is the remaining size dependence, and what a picture fixes |
| `research/experiments/` | the parameter-matched representation probes behind note 18 |
| `scripts/` | one-off builders for the committed data files |
| `research/` | design study behind all of this |

## Status

The braid environment is at M2 of the roadmap in
[research/08-roadmap.md](research/08-roadmap.md) and is complete and tested.
MCTS policy runs live in `../pgx-mcts-bench`; the newer embedding-metric branch
lives here and remains an unpromoted calibration experiment.

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

* **The certified lower bounds are computed, and wired into a search.** `|σ|/2` was
  claimed but silently returned nothing without `spherogram`, which was not installed;
  it is now exact (validated on 2870 tabulated determinants), `|τ|` is computed from knot
  Floer homology, and a non-cyclic `H₁(Σ₂(K))` certifies `u ≥ 2` where both of those are
  zero. `rf_knots.unknot_search` uses them as an admissible heuristic, so a sequence that
  meets the bound **determines** `u` rather than bounding it.
* **The ladder's worst calibration gap is closed.** `R(3,18)#0` — `7_5`, `u = 2` — stood in
  the ratchet at 6 crossing changes. The branch-and-bound finds 2 in five seconds, meets the
  certified bound, and stores a witness that replays through the reference implementation.

Open, in rough order of value: Rasmussen `|s|/2`, which needs Khovanov homology and has no
offline backend here; a per-cell action space, so the policy head stops depending on the strand
count at all (`research/experiments/strand_shared_head.py` removes the dependence from the
*parameters*, which is the half that blocks checkpoint transfer); a learned head register; hard
unknot diagrams; knot equivalence as a two-tape machine.
