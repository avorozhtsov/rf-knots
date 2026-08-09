# rf-knots research

The active research contract charges only semantic knot operations:

`L_AB = A * crossing_changes + B * semantic_moves`.

Controller-local head shifts, tape writes, and memory-state changes consume a
separate internal budget and never contribute to `moves`. Historical notes,
leaderboards, and checkpoints from before this correction are preserved under
[`../archive/pre-semantic-moves-v1/`](../archive/pre-semantic-moves-v1/) and are not
numerically comparable with current work.

## Active map

| File | Purpose |
|---|---|
| [00-glossary.md](00-glossary.md) | Exact terminology and metrics for current experiments |
| [01-game-design.md](01-game-design.md) | Population propose/solve design principles |
| [02-alphazero-backprop.md](02-alphazero-backprop.md) | Search and backup formulation |
| [03-knot-env-pgx.md](03-knot-env-pgx.md) | Braid environment design |
| [04-related-work.md](04-related-work.md) | Related work |
| [05-compute-budget.md](05-compute-budget.md) | Compute model and batching constraints |
| [06-network-growth.md](06-network-growth.md) | Capacity growth after a measured capacity limit |
| [07-domain-choice.md](07-domain-choice.md) | Domain choice |
| [08-roadmap.md](08-roadmap.md) | General roadmap |
| [09-vs-learning-to-unknot.md](09-vs-learning-to-unknot.md) | Research contribution relative to prior unknotters |
| [10-invariants-and-representations.md](10-invariants-and-representations.md) | Invariants, representations, and oracle controls |
| [12-serial-formulation.md](12-serial-formulation.md) | Current two-budget serial-controller contract |
| [13-directions.md](13-directions.md) | Short map of immediate directions |
| [16-scientists-collaboration.md](16-scientists-collaboration.md) | From-scratch roster and five-arm collaboration protocol |
| [17-scalable-braid-raster.md](17-scalable-braid-raster.md) | Scalable architecture families and gates |
| [17-text-mediated-sharing.md](17-text-mediated-sharing.md) | Longer-horizon theory/language sharing proposal |
| [18-raster-representation.md](18-raster-representation.md) | Active masked raster representation |
| [19-superseding-the-rl-unknotter.md](19-superseding-the-rl-unknotter.md) | Hard-knot upper-bound campaign and evidence standard |

Implementation references are in [`../docs/`](../docs/). Generated experiment
artifacts live in the sibling `pgx-mcts-bench` repository. A result enters an
active note only after its metric, evaluation set, compute, seeds, checkpoint
hashes, and solved-set differences are recorded.
