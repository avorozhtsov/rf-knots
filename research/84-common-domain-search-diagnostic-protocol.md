# Common-domain full-search diagnostic

2026-09-21. Frozen after completed greedy diagnostic82/83, before MCTS scoring.
Use its exact16 TRAIN roots and three frozen checkpoints, L10/L1000, capacities
48/12, CCcap2 and128 native plies (Q) or semantic moves per path (RF).
Seed20261042. One run per condition; no TEST promotion claim or retuning.

Q254/Q304: existing native NeuralMCTS, eight simulations per controller
decision, temperature0, no root noise, original checkpoint-derived search
settings otherwise. Preserve native value predictions and inference cache;
count actual forward samples/batches and cache hits/misses. Apply CCcap2 and
PASS exclusion at EVERY tree transition, not only executed roots. Preserve
head/controller state. Record the actually executed path AND the best replayed
successful path encountered anywhere during search. A solution found in an
unselected branch still counts as a discovered proof. Stop execution on native
termination or128 decisions; do not stop merely on the first speculative proof.

RF joint57: existing continuation cost MCTS, eight total tree simulations,
128-move continuation horizon, widening4, cpuct1.5, embedding-prior temperature1,
stochastic policy continuations with their existing exact-braid repeat filter.
Set insertion candidate limit1056 (=2*48*11), which covers every legal insertion
in this short-input envelope; ordinary actions are also fully offered. The
untrained scalar value head remains unused. Report actual policy-state neural
samples separately from cached tree embedding samples; they are NOT FLOPs.
Retain all continuation transcripts and replayed incumbents using the existing
cost-MCTS auditor. The shared semantic catalogue excludes PASS.

These are different search algorithms: eight Q simulations PER DECISION is not
eight RF tree expansions TOTAL. Neither equal compute nor a causal MCTS-only
ablation is claimed. In particular RF also introduces stochastic continuations
and repeat avoidance versus greedy82, while Q introduces value-guided lookahead.
Compare success sets, costs on common successes, and actual work together.
Do not attribute any gain solely to embeddings, architecture, or MCTS.

Save96 immutable per-condition outputs, input/source hashes, real process and
committed progress. Resume only matching contract; no duplicate writers. Verify
every counted proof from its original root, CC/move/cap accounting, and native
controller replay for Q speculative best witnesses. Failures have no invented
L cost and are not training negatives. Existing training69 and evaluations70/74
remain unchanged; this small diagnostic informs the next search/architecture
step, not replacement of the independent embedding experiment.
