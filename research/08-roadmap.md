# 08 — Roadmap

Ordered so that each milestone produces something falsifiable, and so that the expensive parts
are only reached after the cheap parts have proven the mechanism works.

## M0 — League harness on rational sums (CPU, ~1–2 weeks)

Family: `S = Σ_{n≥1} p(n)/q(n)`, `deg q ≥ deg p + 2`, no poles at positive integers.
Verifier: sympy — compare candidate closed form to high-precision numeric evaluation, plus
symbolic simplification. Planted solutions: generate `q` from chosen roots so the answer is known.

Build:
- [ ] problem bank with invariant fingerprint + novelty dedup
- [ ] round loop: 5 agents × 2 proposals × k attempts each
- [ ] scoring rule from [01](01-game-design.md) (own-solve gate, `4p(1−p)`, novelty, budget)
- [ ] 2PL IRT fit; `θ` and `b` tracked per round
- [ ] **frozen anchor set** and anchor-only reporting
- [ ] collapse detectors: mean `b_j` flat, novelty → 0, `p_j` → 0 or 1, proposal entropy drop

Agents at this stage can be trivial (templated generators + sympy solvers with varied budgets) —
the point is to validate the *measurement instrument*, not to learn anything.

**Exit criterion:** with deliberately handicapped agents, the league's `θ` on anchors rises and
the collapse detectors fire on synthetic bad-behaviour injections.

## M1 — `braid_unknot` pgx environment — **done**

- [x] state `int32[L]` (int8 was not worth the casting noise), moves 0–7 per
      [03](03-knot-env-pgx.md), all masks vectorized
- [x] `num_players=2`, Scrambler/Simplifier phases, budget `K`, cap `M`
- [x] **differential test** against a list-based Python reference — every move, every mask bit
- [x] **exact braid-group check** via the Artin representation (faithful, so equality in `B_n` is
      decidable): every braid-group move provably preserves the group element
- [x] **component invariant**: every state stays a 1-component closure, verified on every step
- [x] random-play baseline: 1.6% of `K=6` tier-0 scrambles undone, 0.1% at `K=12` tier-1
- [ ] positional policy head `[L, M]` — deferred to M2, where the network lives
- [ ] SnapPy / sage.knots cross-check on ≤10-crossing tables — deferred to M3, where real knot
      tables enter; the Artin + component checks cover the move set itself

**Exit criterion — met.** 12.5k verified steps with zero disagreements (`rf-knots selfcheck`);
141k env steps/s at batch 8192 **on CPU** (Apple silicon), ≈4.7k games/s, so the GPU target is
comfortably clear.

Notes from building it:

* Starting from the **empty 1-braid** is forced. The closure of the identity braid in `B_n` is the
  `n`-component unlink, so `n=1` is the only trivial start state that is a knot. The
  component-count invariant then does real work: it is what rules out ever wandering into a link.
* `PASS` is always legal. Pgx requires a non-empty mask, and it doubles as "scramble less" /
  "resign", which is meaningful once the budget-efficiency reward from [01](01-game-design.md) lands.
* The BFS oracle needs a **growth cap**, not just a depth cap. Insertions dominate the branching
  factor (`2·(n−1)·(L+1)` of them per node); without bounding how far the word may lengthen, depth
  6 does not terminate. Sound but incomplete: a returned path is valid, `None` never means
  "unsolvable".
* Random scrambling produces mostly self-cancelling junk (`s1^-1 s1 s1 s1^-1 …`). Exact BFS puts
  numbers on it: `K=3,4,5` random scrambles have mean optimal solution depth `2.56, 3.16, 3.96`,
  i.e. **~0.7 moves of difficulty per move spent**, and never harder than `K`. That is the
  difficulty floor and the concrete form of the original objection — a random proposer is cheap
  but uninformative. It is exactly what a trained Scrambler has to beat, and it gives M2 a
  falsifiable target.

## M2 — AlphaZero on Scrambler vs Simplifier (~1 month, 1 GPU)

Swap the env into `../pgx-mcts-bench`. Reuse the training loop, arena protocol, fixed-compute
methodology, and the `U1`–`U5` exploration comparison.

- [ ] tier-0 smoke run (laptop) → tier-1 run, ≥5 seeds
- [ ] curriculum on `K` driven by arena win-rate
- [ ] MuZero ablation: can a learned dynamics model absorb the Reidemeister/Markov rules?
- [ ] `U1`–`U5` rerun — sparse-reward long-horizon domain, different from 6×6 Go

**Exit criterion:** Simplifier beats a strong Scrambler at `K` well beyond what random or greedy
crossing-reduction handles; a curve of `K` vs win-rate over training.

## M3 — First mathematical output (~1 month)

**The decisive experiment** (see [09-vs-learning-to-unknot.md](09-vs-learning-to-unknot.md)):
does training against an *adversarial* generator produce a Simplifier that transfers to instances
no random generator would emit? This, not a win rate, is the paper.

- [ ] train Simplifier-vs-random-Scrambler and Simplifier-vs-trained-Scrambler at **matched compute**
- [ ] evaluate both on held-out instances neither generator produced
- [ ] report the transfer gap

- [ ] point the trained Simplifier at the standard prime knot tables (≤12 crossings) and at the
      hard-unknot-diagram corpora; record unknotting-number **upper** bounds
- [ ] compute **lower** bounds out-of-env: `|σ(K)|/2`, Rasmussen `|s(K)|/2`, Nakanishi/Alexander
      obstructions; cache per knot type
- [ ] report every knot where upper = lower (unknotting number *determined*)
- [ ] publish the Scrambler's hard-instance corpus with generation budgets — this is a dataset
      contribution independent of the RL claims

**Exit criterion:** either a bound matching published state of the art, or an honest table showing
the gap. Both are results.

## M4 — Promote to the 5-agent league (~2–3 months, multi-GPU)

Now the proposer is a policy over *scrambling strategies* and, optionally, an LLM emitting move
programs. PopuLoRA-style: one frozen base model, 5 LoRA adapters, weight-space mutation/crossover
as the PBT replacement step.

- [ ] difficulty head `d_φ(instance) → b̂` trained on M2/M3 outcome matrices; use it to amortize
      the proposer's search (see [02](02-alphazero-backprop.md))
- [ ] the M0 harness now drives real agents
- [ ] capacity growth schedule from [06](06-network-growth.md), triggered on measured saturation
- [ ] ablation that justifies the whole project: **population of 5 vs. 2 vs. 1, compute-matched.**
      This is the paper's central experiment — it is the claim nobody has tested on a
      mathematical family with an exact verifier.

## M5 — Bridge to L-functions (exploratory)

Different game: agents propose *conjectures* about LMFDB statistics; scoring is
(a) survives numerical testing on a held-out conductor range, (b) is not implied by a known
statement, (c) other agents fail to construct a counterexample. The verifier is a **refuter**,
not a solver. Treat as research, not engineering; do not schedule it.

---

## Decision log — things to revisit

| Decision | Chosen | Revisit if |
|---|---|---|
| encoding | braid words | `L`-overflow rate >1%, or braid relations prove too weak → switch to grid diagrams |
| first game | 2-player Scrambler vs Simplifier | if the Scrambler trivially wins at all `K`, rebalance `M/K` before abandoning |
| ground truth | planted (generate from the unknot) | never replace with majority vote |
| proposer reward | `4p(1−p)` × own-solve × novelty / budget | if problem-space coverage stalls, add an explicit diversity bonus in fingerprint space |
| population size | 5 | 3 if compute-bound; the argument in [01](01-game-design.md) needs ≥4 solvers per item |
| net | size-invariant (transformer/CNN over the word) | only grow depth/width, never re-architect for a new `L` |
