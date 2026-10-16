# Root-budget-16 multi-output value bootstrap (engineering stage)

Prospective contract, 2026-10-03. This is the first **neural training** step
for the separate protocol256 value architecture, not a value-guided MCTS test.

Use the immutable 32,768-step long raster actor from protocol261 as the
frozen shared encoder and policy. Extract all first 2,560 attempt outcomes
from `artifacts/raster-budgeted-value-teacher-unselected-v1`: 128 TRAIN-owned
representations, L10/L1000, ten independent search seeds, exactly 16 root
simulations per attempt, unchanged distance-prior/rollout teacher. Independently
check every source SHA, task identity, root split and any found witness by
full replay, resources, CC/moves and objective cost. The best witness of
**the same attempt** supplies its paired CC and moves. An unsolved attempt
supervises only the bounded solve event, never costs or an action negative.

This stage uses **root-query-group ownership**: all examples are root states
whose original query group is TRAIN and excluded from the pre-existing heldout
roster. It does not use intermediate states as input examples. Historical
trajectory-wide invariant ownership rejected 227 raw successful attempts;
that conservative rule remains appropriate for trajectory policy training,
but would turn a successful root search into a false failure label for the
root-level teacher event. Changing the split unit is an explicit new pilot,
not a silent reinterpretation of the old S10 audit. An independent exact
query-state collision check against saved DEV/TEST roots is required before
training. No claim about generalization to intermediate nodes, or to budgets
other than 16, follows from this stage.

Freeze the raster encoder and existing policy; train only the new
`BudgetedCostValueHead(embedding_dim=64,width=128,reference_simulations=1024)`.
Inputs: the encoder's knot/layout/isotopy vectors, remaining CC/moves at the
root, lambda, and `local_simulations=16` encoded as `log1p(16)` by the head.
Outputs: `p_solve` for this exact frozen 16-simulation teacher, and paired
conditional `cc_remaining`/`moves_remaining` for that attempt's selected
verified witness. Use the existing masked BCE + two normalized smooth-L1
cost terms, with no cost gradient on failures. One class-conditional constant
baseline must be reported separately. Train a fixed 512 AdamW updates,
batch 64, LR 0.001, weight decay 0.01, gradient clip 1, seed 20261004,
checkpoint every 64 updates with optimizer and RNG; do not select a
checkpoint on TRAIN loss.

Next, collect independent DEV outcomes from the same frozen teacher on
validation-owned roots at 16 simulations and predeclared seeds, then measure
probability calibration/Brier/log loss versus frequency constants and paired
CC/move errors on successful attempts. Extend to several *fresh local*
budgets and internal TRAIN states under a new contract before interpreting
the budget input or wiring the head into AlphaZero-style leaf MCTS. TEST150
is not used for selection. No claim of heuristic superiority follows here.
