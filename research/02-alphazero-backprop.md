# 02 — What "backpropagation like AlphaZero" means here

There are two different loops, and conflating them is the most common way this kind of project
goes wrong.

## Loop A (inner): MCTS value backup over a real game tree

This is what AlphaZero's backpropagation actually *is*: after a simulation reaches a leaf,
the leaf value `v` is propagated up the path, updating `N(s,a)` and `W(s,a)`, and the visit
distribution `π ∝ N^(1/τ)` becomes the training target for the policy head. It requires:

* a deterministic, cheap, exactly-known transition function `s, a → s'`;
* a terminal reward;
* a fixed, maskable action space.

**A knot-simplification environment has all three.** The move sequence (Reidemeister/Markov moves,
crossing changes) *is* a game tree. This is where your `../pgx-mcts-bench` code drops in
essentially unchanged — including the `U1`–`U5` exploration-rule comparison, which becomes a
real experiment again in a new domain (single-agent-ish, sparse-reward, long-horizon: exactly
where PUCT-vs-UCT differences should show up more sharply than in 6×6 Go).

An LLM emitting a chain of thought does *not* have loop A in this sense — there is no exact
transition model over "reasoning states". You can bolt MCTS onto token generation, but the value
backup is then over a learned/heuristic model and loses AlphaZero's guarantees. Be explicit about
which one you are building.

## Loop B (outer): population / league update

This is *not* MCTS. It is population-based RL:

* per-round advantages from the scoring rule in [01](01-game-design.md), centered within the round
  (= GRPO group baseline);
* policy-gradient update per agent;
* periodic exploit/explore: reseed the weakest agent from the strongest with perturbation
  (PBT), or evolve LoRA adapters in weight space (PopuLoRA's operator set).

There is no value backup across rounds. Rounds are not a tree.

## The bridge: which quantity is the "value"

If you want something that genuinely rhymes with AlphaZero across both loops, make it the
**difficulty estimate**:

```
b_j   (item difficulty)   ←  estimated from the round's outcome matrix          [loop B]
V(s)  (state value)       ←  estimated from MCTS backup inside one solve        [loop A]
```

and then *train a difficulty head* `d_φ(problem) → b̂` on the round outcomes. This head is what
the proposer conditions on: it can then search for high-`b̂` instances offline, without paying
for 20 solver rollouts per candidate. That is the analogue of AlphaZero's value network amortizing
the tree search, and it is the piece that makes the proposer's search tractable.

Concretely:

```
proposer policy π_prop  --generates--> candidate instance
   guided by MCTS over the *scrambling* move tree
   with leaf value = d_φ(instance) · own_solvability(instance)
```

Both the proposer and the solver then run the *same* MCTS machinery over the *same* move tree,
in opposite directions. That symmetry is the elegant core of the design and it is what makes the
Scrambler-vs-Simplifier formulation in [03](03-knot-env-pgx.md) worth building first.

## Recommended reuse of pgx-mcts-bench

| pgx-mcts-bench piece | reuse |
|---|---|
| AlphaZero self-play + training loop | as-is, swap the env |
| MuZero (learned dynamics) | keep for a later ablation: can a net learn Reidemeister moves? Genuinely interesting — the rules are simple but the state is a graph |
| `U1`–`U5` exploration rules | as-is; new domain = new evidence |
| arena protocol | becomes Scrambler-vs-Simplifier match play |
| fixed-compute methodology, multi-seed protocol | keep, it is the most valuable part |
