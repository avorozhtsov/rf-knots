# Existing checkpoint selected; value-head bootstrap smoke, not value training

2026-10-02. The user requested proceeding from the best available saved weights
without waiting for 248 to finish, and explicitly required a separately trained
multi-output value head and a node-specific simulation-budget input. This is a
new experiment namespace and does not modify or stop 248.

The selected immutable initialization is the 248 `long` checkpoint at step
32,768, SHA256 `9b8ebee990c1986eec3cda30a00105da9819892dd8649dcc438468a197685cba`.
The saved checkpoint hash was recomputed. On the latest available common fresh
general DEV report for both arms at that step, mean interval violation was
0.242108 for `long` versus 0.581826 for `control`; historical DEV was 0.504127
versus 0.505187. This is a DEV metric selection only, not evidence that this
checkpoint has the best search performance. TEST150 was not used. The checkpoint
has policy/embedding weights but no trained bounded-success/CC/move value head.

`scripts/budgeted_cost_value_head_v1.py` adds a separate untrained head from the
existing encoder's three global vectors. Outputs are bounded success
probability and paired remaining CC/moves, conditioned on node-relative
remaining resources, cost weight and `log1p(local_simulations)`. A censored
failure trains only the success output. Four unit tests pass. This head is not
yet integrated into MCTS or trained on data.

`scripts/bootstrap_cost_value_teacher_v1.py` is a reproducible small CPU data
adequacy check, using the frozen checkpoint and two cost weights of **one**
TRAIN representation. It ran ten independent searches at L10 and ten at L1000,
16 tree simulations each. Source/input/checkpoint hashes and per-attempt
shards are retained under `artifacts/raster-budgeted-value-teacher-smoke-v1`;
the manifest is `completed`. The existing search runner replayed every
continuation, and the new runner checked TRAIN ownership per continuation.
Results: 17/20 searches found a complete path, but only three searches
contained a TRAIN-admissible solved continuation (one at L10, two at L1000).
Their observed pairs were (7 CC, 59 moves), (7, 62), and (7, 64), respectively.
These are **not** enough distinct representations or admissible successes to
train/calibrate the head or rank tournament variants. The seventeen raw solves
must not be presented as seventeen usable training positives. The three
unsolved searches are bounded failures, not action negatives.

For a selected internal node, `estimate_local_simulations` provisionally
forecasts future visits from the root's remaining simulations and smoothed
edge-visit shares along the selected path. Its value is *estimated*, not the
node's guaranteed allocation. It has unit tests, but no empirical calibration
or integration yet. The exact formula and required future-visit audit are in
protocol256. The next collection version should choose multiple TRAIN roots
with verified eligible path diversity (for example, candidate condition70/71
has a short 13-letter representation and four recorded distinct costs), pin a
new roster and search budget, and run an independent ownership/replay audit
before using any label. Do not expand a costly sweep from this one-root smoke.
