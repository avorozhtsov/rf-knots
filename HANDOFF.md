# Handoff — read this first

Context for continuing the project in a fresh session. Written 2026-08-01.

## What this is

RL for knot theory, aimed at a population of agents that both *propose* and
*solve* mathematical problems. It currently does one thing well: **an AlphaZero
agent that unknots braid words**.

Two repos, both public, both pushed, both on `main`. `pgx-mcts-bench` depends on
`rf-knots` by path (`../rf-knots`), so they must sit side by side.

| repo | contents |
|---|---|
| [rf-knots](https://github.com/avorozhtsov/rf-knots) | environment, generator, research notes |
| [pgx-mcts-bench](https://github.com/avorozhtsov/pgx-mcts-bench) | training, search, the ladder |

Read in this order:
1. `rf-knots/docs/representation.md` — the encoding, the move set, Reidemeister/Markov
2. `rf-knots/research/12-serial-formulation.md` — the most recent findings
3. `rf-knots/research/README.md` — the design study and where evidence revised it

## The ladder has two halves, and they are scored differently

**Rungs 0–16, the calibration set.** `unknot`, then torus knots up to `T(3,5)`.
Every `u` is a theorem (Milnor conjecture / Kronheimer–Mrowka), so the gap to
truth is measurable. Promotion can exit on `objective` — reaching `u + tolerance`.

**Rungs 17–30, the challenge set.** Random **mixed-sign** braid words, `u` unknown,
marked `UNKNOWN_UNKNOTTING = -1`. Promotion there can only exit on `plateau` or at
the cap: there is no theorem to reach.

The change happened because the labelled families are the *structured* ones. Every
torus knot and positive braid is fibred, chiral, positive-signature, and satisfies
`u = g₃ = g₄`. An agent can learn "reduce monotonically, crossing changes always
pay", be right on all of them, and have learned nothing that transfers. Worse, ten
of the sixteen positive-braid rungs were torus knots renamed — on two strands a
reduced positive word is `σ₁^c`, so `P(2,11)` *is* `T(2,11)`.

Old rung records are in `artifacts/archive-positive-braid-rungs/` — the only
evidence anything here ever cleared u = 8, 9, 10.

## The reference is now a ratchet, not a theorem

`artifacts/bounds.jsonl` — append-only claim log, `bounds.py` folds it on read.
A knot's `u` is **the fewest crossing changes anyone has ever used**, with the
holder recorded; it moves the moment someone beats it. Seeded from 263 claims
across every checkpoint, giving 23 knots a standing record. `artifacts/bounds.md`
is the rendered table.

Two things it already showed:

* **On eighteen labelled knots the record equals the theorem exactly.** The
  ratchet independently reproduced the Milnor conjecture.
* **`P(3,20)#0` sits at 11 against a theorem of 9.** Not a contradiction — an
  upper bound can be loose — but it means nobody found the optimal sequence there.
  That gap is invisible on unlabelled knots, which is the calibration set earning
  its keep.

Runs claim automatically via `--bounds artifacts/bounds.jsonl`. **The currently
running jobs were started without it**, so they are not claiming; add it on the
next restart.

## The finding worth chasing

**The ranking on structured knots and unstructured knots disagree.**

| arm | labelled rungs | unlabelled rungs |
|---|---|---|
| `search-heavy` | exactly optimal at every `T(3,4)`/`T(3,5)` rung | 4.00 on `R(3,14)#0` |
| `u1-puct` | +1.42, +3.92, +6.00 — worst in the field | **3.00** on `R(3,14)#0`, holds 3 of 6 records |

`u1-puct` is the worst arm where `u = g` and a greedy positive-braid strategy
works, and the best where it does not. Visible at rungs 23 and 27. If it survives
a second seed it says the labelled ladder was rewarding a heuristic that does not
generalise — which is the whole reason the challenge set exists.

**Nothing here has more than one seed.** That is the single biggest weakness of
every table in this project.

## Running now

| where | what |
|---|---|
| local, 4 slots | **leaders**, open-ended on unlabelled rungs: `u1-puct`, `wide-net`, `search-heavy`, `s-head-256`, `s-reg4` |
| local, 4 slots | **climbers**, `--stop-after 16`: the other eleven, climbing to the top of the calibration set then exiting |
| server, 3 containers | `s-burau-oracle`, `s-head-1stride`, `s-reg8` |

Queue scripts live in the session scratchpad (`queue-lead.py`, `queue-climb.py`,
`jobs-*.jsonl`). One job per candidate, process-group isolation so SIGTERM kills
the tree, start rate-limited to one per 20s — a gate on `getloadavg` cannot see a
job launched five seconds ago and will happily burst-start eight onto a full
machine.

The split exists because **slowest-first only works when jobs terminate.** The
leaders are on rungs that can only end at the cap, so they held all eight slots
for nine hours and eight arms never started. Bounded targets for the climbers fix
that.

### The server

`locuscanvas.com` / `89.169.108.199`, user `artemvorozhtsov`, key `~/.ssh/id_ed25519`,
passwordless sudo. Full details in `pgx-mcts-bench/artifacts/oracle/locuscanvas_log.md`.

**It runs the user's production stack.** `locuscanvas-postgres` and
`locuscanvas-persona-backend` have been in restart loops since the machine last
booted — a permissions failure on `/var/run/postgresql`, unrelated to the training
containers and untouched. Training is capped at `--cpus=1.2` with
`OMP_NUM_THREADS=1` each so it cannot starve the web services.

## Open, in rough order of value

1. **A second seed.** Every number in every table is one seed. `--stop-after 16`
   makes a clean replication cheap: 15 arms over the calibration set only. The
   jobs file (`jobs-seed1.jsonl`) already exists and was never run.
2. **Add `--bounds` to the running jobs** so the ratchet accumulates instead of
   needing to be re-seeded by hand.
3. **Store the unknotting sequence as the witness.** `bounds.py` currently records
   the knot's defining word, so a bound can be attributed but not re-verified.
   Until then these are trusted claims, not checkable ones.
4. **Run `braid-ladder-rescore`.** Recorded `cc` is measured once, at promotion,
   with the weights of that moment. The rescore re-measures with current weights
   and has already shown drift larger than the gaps between adjacent rows —
   `s-gru128` moved 2.10 → 3.33 on one rung. It was killed by a restart and never
   completed.
5. **`s-burau-oracle` has not cleared rung 9 in nine hours.** Meanwhile
   `s-head-1stride` — plain window, worst stride set, no accumulator — went from
   rung 1 to 14 on the same box. If the oracle caps, that is evidence against the
   whole whole-tape-accumulator direction, and it applies with more force to
   `s-fsa32`, `s-gru128` and `s-ff4-p5`, which are *learning* what it is handed.
6. **Certified lower bounds** (`|σ|/2`, `|s|/2`, `|τ|`) with branch-and-bound —
   what turns an upper bound into `u(K) = n`. Unaffected by the zero-knowledge
   constraint: bounds verify output, they are not features.
7. **Batch the MCTS leaf evaluations.** 7.8× measured on this laptop, and the
   prerequisite for a GPU ever being worth renting.

## Process notes learned the hard way

- **8 seeds minimum** for anything about instance difficulty. 3 gave a false
  positive that survived two rounds of reporting.
- **Read the trained checkpoint, not the design doc.** Every serial defect was
  found by probing weights and action histograms, none was visible in intent.
- **Diagnose before fixing.** I called `u1-puct`'s 14.00 crossing changes a FiLM
  conditioning bug. It was two different measurement artefacts — a stale recorded
  value, and a search-depth failure that 128 simulations fixed with the network
  untouched. Only 9 of 177 rungs inverted, which is what ruled out a systematic
  fault before any code was written.
- **Never let a report hardcode what it is reporting.** The ladder header claimed
  "Ten stages" through two changes to the stage list. Reading `len(STAGES)` is the
  same bug one level down — it relabelled a historical ten-rung run with today's
  rungs. Reports now derive everything from the results themselves.
- **Resume on stage identity, not index.** Inserting a rung silently repoints
  every index-keyed checkpoint at a different stage.
- **A promotion threshold makes the metric at promotion nearly meaningless.** It
  reports where the network happened to be when it crossed the bar.
- **`cc` is conditional on solving and anti-correlated with the solve rate.** An
  arm that abandons hard instances drops them from its own average, so failing
  more can look cheaper. `cc/sr` is the honest column.
- **A gate on `getloadavg` cannot throttle a burst** — it is a one-minute decaying
  average and lags a launch by about a minute. Rate-limit starts instead.
- **`pkill` on a `uv run` wrapper orphans its child.** Two launches accumulated
  instead of replacing one another and load hit 45.
- **`grep "[m]y-pattern"` still matches your own shell command line**, which cost
  several rounds chasing three processes that were my own `ps` invocations.
- **List a shared host's running services before putting load on it.** `nproc`,
  `free` and `uptime` said "idle"; it was running production.
- Negative results belong in commit messages and notes, not in dead code.
