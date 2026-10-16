# AlphaZero-style multi-output leaf head for cost MCTS

2026-10-02. The user's original requested direction is **AlphaZero-style
iterative policy/value-guided tree search with a multi-output head**, not a
long-rollout search with a single success-probability scalar. Preserve this
requirement in every implementation and comparison; a one-output value head
does not satisfy this protocol. Prospective design amendment. This is not a
change to the running 248 pair, the completed protocol53 collection, or any
hash-frozen checkpoint. No value-guided search or new training is claimed here.

## Output contract

For an exact braid state `s`, remaining crossing-change and move caps
`(r_cc, r_moves)`, objective weight `lambda` (10 or 1000), and an explicitly
allocated local compute budget `b`, the network must return a **policy prior over legal moves**
and the following separate value outputs from the same shared representation:

- `p_solve`: probability that a fresh run of the specified frozen continuation
  procedure, rooted at `s`, finds a complete solution within `(r_cc, r_moves, b)`;
- `cc_remaining`: estimated remaining crossing changes **conditional on the
  bounded search finding a complete solution**;
- `moves_remaining`: estimated remaining semantic moves for that **same
  bounded-search solution distribution**.

The two cost outputs must not be separate marginal minima: the minimum-CC
path and the minimum-move path can be different, and their two minima may not
form any feasible path. Each independent bounded search attempt supplies one
outcome: failure, or its own cheapest fully replayed solution under the fixed
`lambda*CC + moves` rule. Train the paired outputs from the SAME successful
attempt and its SAME selected suffix, never by mixing minima from different
attempts. Their intended estimand is the conditional mean CC/moves of that
attempt's best found solution, not the mathematical minimum. Condition the
outputs on `lambda` and remaining resources. A joint feasible `(CC,moves)`
distribution with a failure bin is also valid and preferable if point estimates prove inconsistent or poorly
calibrated. A stored best-known path is an upper bound, not certified optimum.

`B_root` is the number of simulations allocated to the whole root search;
`b` is the number of additional simulations explicitly allocated to a fresh
continuation rooted at `s`. They are not interchangeable. In particular,
`B_root - simulations_used` is **not** `b` for an internal tree node: the
remaining root simulations may visit other nodes. Nor does the number of
previous visits to `s` specify its future budget. If `b` is an input, the
search must choose and record `b` for each use of this head; it must not infer
it from depth or total root visits. The value contract deliberately starts a
fresh continuation at `s`, because predicting the result of resuming an
already partly built subtree would additionally require that subtree's
statistics as input.

Use node-relative resource units: the costs predicted are additional CC and
additional semantic moves *after* `s`; `(r_cc, r_moves)` are the caps left at
`s`, not the original root caps. Encode `b` on a logarithmic scale (for
example `log1p(b) / log1p(B_ref)` for a fixed documented `B_ref`), and use
the same encoding for TRAIN and inference. Pin what one simulation means and
record neural evaluations and CPU time separately; equal simulation counts
need not imply equal compute. `lambda` may change which completed path is
selected and must remain an explicit input, while the paired CC/move outputs
must still refer to that same path.

Two viable implementation levels must not be conflated. The first, simpler
pilot fixes one local `b=b0` for every evaluated leaf and leaves `B_root` only
as a search-control parameter; this needs no budget-conditioned head. A later
budget-conditioned head must train on independently restarted attempts from
TRAIN states at several predeclared local budgets `b`, with success/failure
and best replay-verified `(CC,moves)` recorded at each cutoff. Varying
`B_root` at the root while always labeling an internal node with the root's
leftover budget does not generate valid supervision. If the continuation
procedure itself changes with checkpoint, freeze and version that procedure
for each data-collection round so that `p_solve` has a reproducible target.

For adaptive AlphaZero-style tree allocation, an internal node has no known
future local budget. The provisional *forecast* for a node on the currently
selected root-to-node path is `b_eff = 1 + (B_root - t - 1) * product(q_e)`,
where `t` is the number of completed root simulations and, for each selected
edge `e`, `q_e = (N_e + alpha*P_e) / (sum_siblings N + alpha)`. `N_e` counts
completed simulations that traversed that edge; `P_e` is its normalized
policy prior; `alpha > 0` is a pinned smoothing pseudocount. The `1` is the
current selected simulation. This is a real-valued estimate of future visits,
not a promise that MCTS will allocate them. Use `log1p(b_eff)` in the value
tower. Do not use current depth, root budget remainder alone, or past visits
`N_e` alone as the node's future budget. Train the head on integer fresh-node
budgets surrounding the expected range, then check whether interpolation at
`b_eff` is calibrated. Audit this budget forecast by snapshotting TRAIN/DEV
trees, continuing them without intervention, and comparing forecast versus
actual subsequent node visits by depth, objective and resource caps. If the
forecast or resulting value is poorly calibrated, use fixed `b0` rather than
silently treating `b_eff` as observed compute. The concrete provisional
estimator and unit tests are in `scripts/budgeted_cost_value_head_v1.py` and
`tests/test_budgeted_cost_value_head_v1.py`; it is not yet wired into MCTS.

For exploration, keep two distinct numbers. `b_eff` remains the honest visit
forecast used for calibration and reporting. Only when selecting a tree branch,
allow an optimistic value query at
`b_plus = min(B_root-t, max(2, (1+epsilon)*b_eff + delta))`, with the floor 2
reduced to the available root remainder when that remainder is below 2.
Predeclare and compare `delta` in `{0,1,2,3,4}` and `epsilon` in `{0,0.05}`;
do not silently add both or tune them on TEST. Recompute after every completed
root simulation. A higher budget is a *hypothetical* opportunity for the
branch, not an observed visit or a training label. Report calibrated value at
`b_eff` separately from exploratory value at `b_plus`. Since conditional CC
and move means can move in either direction with budget, check monotonicity of
the combined anytime-solution utility, not of either cost output alone. Test
whether the credit actually redirects visits to branches with subsequent
verified solution gain; compare solved fraction, CC, moves, L10/L1000 and CPU
at identical total root budgets. Uniform inflation need not improve ranking.

For a leaf reached after `used_cc, used_moves`, the provisional root-relative
value corresponding to the existing terminal return is

```
p_solve * (1 - (lambda*(used_cc + cc_remaining)
                 + used_moves + moves_remaining)
             / (2*(lambda*root_cc_cap + root_move_cap + 1)))
```

The AlphaZero-style loop is: select a leaf by tree statistics and policy prior,
evaluate that leaf ONCE with the multi-output head, expand its legal moves,
back up the resulting cost-aware value, and repeat. A long stochastic policy
continuation is not the default leaf evaluator in this variant. Search-derived
policy targets and value targets are formed only under the frozen, audited
training contract; this note does not claim that self-play in a single-agent
cost problem can be copied unchanged from Go.

Only a fully replayed terminal solution can become an incumbent. A neural leaf
estimate influences tree selection/backups but is never recorded as a solution.
Terminal solutions use their exact CC/move counts. A `p_solve` failure label
at a pinned cutoff is a valid zero for **that bounded event**; it is not a
proof of unsolvability or a negative action label. `cc_remaining` and
`moves_remaining` losses are masked when no complete solution was observed.

## Data and comparison gate

The saved 248/149 raster checkpoints contain policy and embedding weights but
no trained `p_solve`/paired-CC/move head. Reuse one immutable SHA-pinned
checkpoint only as an encoder/policy initialization. Add a new value head in a
separate model version; do not reinterpret an embedding distance or an
untrained scalar as any of these three outputs. First bootstrap its labels
from fresh, replay-verified TRAIN-only searches by a frozen teacher procedure,
with independently repeated attempts and explicit local cutoff `b`. Failed
attempts supervise only `p_solve`; successful attempts supervise `p_solve` and
the paired CC/move costs of the same selected solution. Known solutions can
provide additional cost-path supervision but not a `p_solve=1` label for a
teacher search that never found them. Calibrate on disjoint DEV groups before
using this head for value-guided collection or comparing tournament variants.

Collect multiple independent TRAIN attempts under the same pinned policy,
resources, seeds rule and compute budget `B`; keep failures and their cutoff
explicit. For every successful suffix, replay all actions to the empty
one-strand braid, check legal actions and resources, and verify TRAIN ownership.
Keep both lambda values separate. Fit and calibrate success probability and
paired conditional CC/move estimates on TRAIN/DEV only; report errors by CC,
length, strands and budget. Do not use TEST150 to select architecture or weights.

Compare the value-guided tree to the frozen rollout-based protocol53-style
tree on the same roots, legal moves and comparable neural evaluations, action
scores, CPU time and simulations. Report solution fraction, CC, moves, L10 and
L1000 separately. A short adequacy pilot must demonstrate sufficient complete
path diversity and nontrivial DEV search benefit before joining protocol254's
eight-way tournament-policy training sweep. Do not spend a multi-day budget
merely because the new head fits TRAIN labels.
