# 04 — Related work (annotated)

## A. LLMs that propose *and* solve, trained by RL

The answer to "has anyone tried this" is **yes, heavily, since mid-2025**. Nobody has done it on
a fixed mathematical family with an exact verifier and a population of 5.

* **Absolute Zero / AZR** — [arXiv:2505.03335](https://arxiv.org/abs/2505.03335).
  One model plays proposer and solver; tasks are code (deduction / abduction / induction); the
  code executor is the verifier. The canonical reference for "zero external data" self-play RLVR.
* **R-Zero** — [arXiv:2508.05004](https://arxiv.org/abs/2508.05004).
  Challenger + Solver, challenger rewarded for driving solver success toward ~50%. Important for
  us because of its **documented failure**: pseudo-label accuracy falls 79% → 63% by iteration 3,
  and the two-model variant collapses after iteration 3. Their ground truth is solver majority
  vote — which is exactly what a planted-solution family lets us avoid.
* **PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play** —
  [arXiv:2605.16727](https://arxiv.org/abs/2605.16727).
  The closest paper to your idea. Teacher and student **LoRA adapters on a shared frozen 7B base**;
  cross-evaluation between sub-populations replaces self-calibration; LoRA weight-space
  mutation/crossover as the PBT replacement step. Key finding, quoting the abstract's claim:
  the single agent "self-calibrates to generating easy problems it can reliably solve", whereas the
  population "enters a co-evolutionary arms race" — and even the *weakest* population member beats
  the compute-matched single-agent baseline. **This is the empirical case for 5 > 1, and the
  cheapest way to get 5 agents (5 LoRA adapters, one base model).**
* **Search Self-play (SSP)** — [arXiv:2510.18821](https://arxiv.org/abs/2510.18821).
  Proposer/solver co-evolution for search agents; adds retrieval-grounded verification.
* **Propose, Solve, Verify (PSV)** — [arXiv:2512.18160](https://arxiv.org/pdf/2512.18160).
  Self-play where a **formal verifier** provides the signal. Closest in spirit to using a
  mathematical checker rather than a code executor.
* **Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain** —
  [arXiv:2603.02218](https://arxiv.org/abs/2603.02218) (Liu, Qi, Du, He).
  The theory paper for this whole area. Thesis: loops plateau because they synthesize more data
  without increasing *learnable information*. Three necessary ingredients they identify —
  **asymmetric co-evolution** (proposer/solver/verifier alternate in strength), **capacity growth**
  (parameters and inference budget must grow with the information demand — see [06](06-network-growth.md)),
  and **proactive information seeking** (novel external task sources). Read this before writing
  any code; it constrains the design more than anything else in this list.
* **Towards Understanding Self-play for LLM Reasoning** — [arXiv:2510.27072](https://arxiv.org/abs/2510.27072).
* **SPELL** — [arXiv:2509.23863](https://arxiv.org/abs/2509.23863) (long-context self-play).

## B. RL / ML for knots — the target domain already has results

* **The unknotting number, hard unknot diagrams, and reinforcement learning** —
  [arXiv:2409.09032](https://arxiv.org/abs/2409.09032) (Applebaum, Blackwell, Davies, Edlich,
  Juhász, Lackenby, Tomašev, Zheng; DeepMind + Oxford), published in *Experimental Mathematics*
  ([link](https://www.tandfonline.com/doi/full/10.1080/10586458.2025.2542174)).
  RL agent finds minimal-length unknotting crossing-change sequences on diagrams up to **200
  crossings**; determined unknotting numbers for **57k knots**; **43 knots of ≤12 crossings whose
  unknotting number was previously unknown**; produced a dataset of **2.6M hard unknot diagrams**,
  most under 35 crossings. This is the benchmark to beat and the proof the domain is fertile.
* **RL unknotter, hard unknots and unknotting number** —
  [arXiv:2603.07955](https://arxiv.org/abs/2603.07955) (Dranowski, Kabkov, Tubbenhauer).
  Agent learns move proposals **and a value heuristic** over Reidemeister moves — i.e. policy +
  value, AlphaZero-shaped. Recovers the upper bound 3 for `u(4₁ # 9₁₀)` via "diagram inflation",
  and describes a **self-improving workbook-driven extension** that systematically improves
  unknotting-number upper bounds across the prime-knot list. Read this one closely: it is the
  nearest neighbour to the loop you want, minus the population and minus the proposer game.
* **Learning to Unknot** — [arXiv:2010.16263](https://arxiv.org/abs/2010.16263)
  (Gukov, Halverson, Ruehle, Sułkowski). Braid words + Markov moves + RL; the source of the
  encoding recommended in [03](03-knot-env-pgx.md).
* **Untangling braids with neural networks** — [arXiv:2206.05373](https://arxiv.org/abs/2206.05373).
* **The Unbearable Hardness of Unknotting** — de Mesmay, Rieck, Sedgwick, Tancer, SoCG 2019
  ([PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol129-socg2019/LIPIcs.SoCG.2019.49/LIPIcs.SoCG.2019.49.pdf)).
  Deciding whether a *given diagram* can be untangled with ≤k crossing changes, or with ≤k
  Reidemeister moves, is **NP-hard**. Good news for us: the search problem is genuinely hard at
  the diagram level, so there is real work for an agent to do.
* **NP-hard problems naturally arising in knot theory** — [arXiv:1809.10334](https://arxiv.org/abs/1809.10334).
* Complexity background: unknot recognition ∈ NP (Hass–Lagarias–Pippenger) and ∈ coNP
  (Lackenby); not known to be in P.

## C. The eventual target: ML on L-functions / LMFDB

* **Murmurations of Elliptic Curves** — He, Lee, Oliver, Pozdnyakov,
  [arXiv:2204.10140](https://arxiv.org/abs/2204.10140), *Experimental Mathematics* 34 (2025) 528–540.
  Found *by* running ML experiments on LMFDB data: averages of Frobenius traces `a_p(E)` oscillate
  with frequency set by the conductor, with shape depending on rank parity.
  [Quanta coverage](https://www.quantamagazine.org/elliptic-curve-murmurations-found-with-ai-take-flight-20240305/).
* **Machine learning the vanishing order of rational L-functions** —
  [arXiv:2502.10360](https://arxiv.org/abs/2502.10360).
* **Murmurations, Mestre–Nagao sums, and CNNs for elliptic curves** —
  [arXiv:2603.17681](https://arxiv.org/abs/2603.17681).
* **BSD invariants and murmurations** — [arXiv:2603.04604](https://arxiv.org/abs/2603.04604).
* Murmurations proved for modular forms (Bober, Booker, Lee, Lowry-Duda) and Dirichlet characters
  (Lee, Oliver, Pozdnyakov).

**Important structural observation.** The LMFDB work is *not* a propose/solve game. It is
supervised learning + human pattern recognition on a curated database. There is no move tree, no
verifier, no self-play. So the LMFDB endgame is not reachable by scaling the knot game — it needs
a different mechanism (see [07](07-domain-choice.md), phase 3).

## D. Infrastructure

* **pgx** — [GitHub](https://github.com/sotetsuk/pgx),
  [paper](https://arxiv.org/abs/2303.17503), [API docs](https://www.sotets.uk/pgx/api/).
  JAX-native, jit-able `step`, multi-agent via `current_player`, custom envs by implementing
  `_init` / `_step` / `_observe` and `num_players`.
* `../pgx-mcts-bench` — your existing AlphaZero/MuZero + `U1`–`U5` exploration comparison on 6×6 Go.
  The training loop, arena protocol and multi-seed methodology transfer directly.

## The gap

| | proposer game | exact verifier | move-tree / MCTS | population > 2 | open math target |
|---|---|---|---|---|---|
| AZR / R-Zero / SSP | ✅ | ✅ (code) | ✗ | ✗ | ✗ |
| PopuLoRA | ✅ | ✅ (code) | ✗ | ✅ | ✗ |
| DeepMind unknotting | ✗ | ✅ | ✅ | ✗ | ✅ |
| Dranowski et al. | partial (workbook) | ✅ | ✅ | ✗ | ✅ |
| **this project** | ✅ | ✅ | ✅ | ✅ | ✅ |

Nobody occupies the bottom row. That is a real, defensible research contribution — and each
column is individually de-risked by an existing paper.
