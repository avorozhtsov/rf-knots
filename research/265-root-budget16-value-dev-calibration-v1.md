# Fixed-budget root value: independent DEV calibration

Prospective evaluation contract, frozen before any teacher result on these
roots, 2026-10-03. The protocol264 head completed its fixed 512 updates;
its checkpoint is selected by the preregistered final step, not by DEV.
The head predicts only the frozen actor's **root** search event at local
budget 16. This does not validate other budgets or an internal-node leaf
evaluator.

Use all 16 historical validation roots from
`artifacts/mcts-tournament-dev-panel-v1/panel.json` and all 16 independent
long validation roots from `artifacts/mcts-tournament-long-dev-panel-v1/panel.json`,
each at L10 and L1000. At each of the 64 root/objective conditions run four
fresh, independent seeds `20271210..20271213` of the same frozen 32,768-step
actor and protocol139 search: 16 simulations, insertion candidates 4,
widening batch 4, c_puct 1.5, temperature 1, distance tree prior, cached CPU
float32 inference, and the saved resource caps/horizon. This is 256 attempts.
Save resumable, SHA-pinned shards; verify every found witness by full replay,
start identity, caps and paired CC/moves. No reference witness is supplied to
the teacher. DEV outcomes never update model parameters or choose among head
checkpoints.

Evaluate the frozen step-512 head on the same 64 root/objective inputs with
`local_simulations=16`. Compare Brier score and binary log loss against
TRAIN-only constant success frequencies, separately by lambda and short
versus long panel; report actual observed solve frequencies and bin counts.
On successful attempts only, compare paired CC and moves errors from that
attempt's selected solution against TRAIN-only conditional constant means.
Report both costs from the **same** outcome, never independent marginal
minima. A failed bounded search has no CC/move label. Report CPU and neural
evaluations for the teacher, and inference cost separately.

The teacher is a rollout/distance-prior search; this is not AlphaZero-style
value-guided MCTS. Regardless of calibration outcome, budget dependence and
internal-node transfer require new multi-budget, fresh-node supervision.
TEST150 is excluded from model or checkpoint selection. No claim of
heuristic superiority is authorized by this calibration.
