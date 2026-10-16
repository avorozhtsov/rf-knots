# Nine-arm rollout-tournament screening pilot, frozen before results

2026-10-03. This is a short exploratory optimizer screen, **not** a test of
AlphaZero value-guided MCTS. No arm may be promoted from TRAIN loss alone.

## Immutable inputs and separation

- Starting model/objective: the long arm at 32,768 steps,
  `artifacts/raster-long-general-training-continuation-v1/long/step-032768-9b8ebee990c1986e.pt`,
  SHA256 `9b8ebee990c1986eec3cda30a00105da9819892dd8649dcc438468a197685cba`.
  On the saved fresh-general DEV interval-violation series, this checkpoint
  had 0.2421 versus 0.2548 at 36,864; this is only a metric-based
  initialization choice, not evidence of better search.
- Tournament paths: the independently audited first 2,560 shards of
  `artifacts/raster-budgeted-value-teacher-unselected-v1`, exactly S=10 at
  16 simulations, 128 TRAIN representations × L10/L1000. Teacher manifest
  SHA256 `358f81da2ff6b6e5c2eee17d3c61355a60db5506559b811b20f06a63a0a418ea`.
  Only complete, admitted, fully replayed paths are compared; no censored
  search is a negative action.
- An independent pre-outcome screening panel is frozen at
  `artifacts/mcts-tournament-dev-panel-v1/panel.json`, SHA256
  `45f83bfe89655531208ee9c8a9775cd2d2a084085bee8de8681779c3da1e40f9`.
  It has 16 distinct validation-owned representations (8 historical
  validation, 8 size-shift), each at L10 and L1000; known solutions and
  historical TRAIN exclusion were checked. Word lengths are only 18–45.
  **This small short/medium panel cannot establish long-diagram utility**;
  a longer independent DEV replication will be required before large CPU
  promotion. TEST150 is not used for selection.

## Matched optimizer contract

Run arms A–I sequentially in a new `mcts-tournament-nine-arm-pilot-v1`
artifact, each for 144 optimizer updates on native MPS, with a fresh AdamW
state from the same fixed model/objective. All arms use the same seed and
root-balanced schedule over the 132 cost-different TRAIN groups (one pass
over every group plus the same first 12 repeat groups), the same metric
draws, known-solution set-positive draws and old relative-loss schedule.
Optimizer LR/decay/clipping and base loss weights match protocol149.
Checkpoint every 16 updates with model, objective, optimizer and Python,
Torch-CPU and MPS RNG states; resume only from the SHA-checked saved state.

Tournament hyperparameters fixed from TRAIN-only profiles: `K=4`, F's
near-best margin `2` objective units, I's gap scale `lambda` and cap `4`.
The common tournament coefficient is `0.002`; H doubles it only in updates
73–144. G's separately marked high-CC objective has half that coefficient
when present. The control A executes a zero-gradient tournament slot to
match network work. Every update allocates 16 policy action-term slots,
including zero-coefficient padding; G splits them 8/8 when its extra signal
is available. Oversized groups use the stratified coefficient sampler with
seed fixed by group/update. Actual action scores, CPU/wall time and effective
distinct groups must still be reported, not inferred from optimizer steps.

**D is a stochastic full-suffix variant**, not exact full-group D: all
cancelled full-suffix coefficients form its sampling population, but 16
terms are used per update. A one-group real-network gate found that one
small sample has high gradient variance (four-draw means: cosine 0.477 at
16 terms, 0.541 at 32, 0.828 at 64, 0.941 at 128). Consequently D's
ranking is exploratory and must be replicated with exact/full-work or a
validated lower-variance estimator before any claim about the exact
full-suffix objective. This limitation is preferable to silently spending
31 times E's compute at equal steps.

A real-network MPS engineering preflight executed one optimizer step for
every arm without saving trained weights. At tournament weight 0.01, B/E
raised total gradient norm from control A's 1.95 to approximately 8.6–8.9;
the weight was therefore reduced before this contract. At 0.002, all nine
preflight steps completed with 16 tournament slots and finite parameters;
total pre-clip norms were A 1.953, B 2.616, C/E/F/G/H/I approximately
2.587, D 1.960. This is a scale sanity check, not a separate per-task
gradient-cosine audit or a learning result.

The pilot's first operational segment is 16 updates of A, then a SHA and
manifest check; continue with `--resume` under the same fixed contract.
After all nine models have actual checkpoints, run the same excluded DEV
search matrix: all 32 saved conditions, one fixed seed `20271003`, 64 tree
simulations per condition, `insertion_candidates=4`, `widening_batch=4`,
`c_puct=1.5`, `temperature=1.0`, horizon equal to each saved move cap,
distance tree prior and cached CPU float32 inference under the unchanged
protocol139 search implementation. This is an internal screening search,
**not** the canonical `pgx-mcts-bench` non-neural superiority comparison.
Replay every found solution. Compare solved starts,
CC/moves and L10/L1000 at matched simulations, neural evaluations and
measured time. Advance no more than two candidates, and do not infer a
winner from this short panel alone.

## First checkpoint and diagnostic, 2026-10-03

The bounded first segment completed A's first 16 optimizer updates and saved
`A-step-0016.pt`, SHA256
`2cae55d3d4fef55ba8cee08206783a07e08ed77bcd3690119286fc5fcee667ed`.
The checkpoint contains finite model/objective tensors and AdamW step 16.
The recorded pre-clipping gradient norm at update 16 was 9,560, which first
looked alarming. This update is also an old-relative-loss update. For
context, the frozen 248 long arm's 544 saved 64-step observations have
median pre-clipping norm 8.63, 90th percentile 23,333 and maximum 35,319;
large intermittent norms were therefore already present under the same
clipping contract. This is not by itself evidence of a new nonfinite state
or a failed optimizer step. Continue the exact saved v1 contract with
`--resume` while monitoring subsequent checkpoints; do not change its
source or overwrite this checkpoint.
