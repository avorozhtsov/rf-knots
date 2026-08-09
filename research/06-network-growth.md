# Network growth

The current roster starts with architectures whose parameter shapes already share
work across positions or strands. We do not grow a network merely because a knot
is hard: first establish a capacity-bound regime with matched training data and
search compute.

## Capacity gate

For one admitted scientist, compare:

1. the current model with increased simulations per move;
2. a deeper model created by zero-initialized residual block stacking; and
3. a wider model created by function-preserving channel duplication followed by
   conservative fine-tuning.

Immediately after migration, old and expanded networks must agree within a stated
numerical tolerance. Then train both from the same replay snapshot and compare
paired held-out solved sets, capped L1000, throughput, and memory. Growth is
justified only if the expanded model wins after total compute is matched and the
gain persists on larger-strand or longer-word tasks.

Vertical stacking increases reasoning depth. Horizontal widening increases state
capacity. Either can be applied after the first curriculum segment, but the grown
scientist begins a new checkpoint lineage and its additional training compute is
charged to its arm.

Historical ladder measurements and detailed Net2Net notes are archived in
[`../archive/pre-semantic-moves-v1/research/06-network-growth-full.md`](../archive/pre-semantic-moves-v1/research/06-network-growth-full.md).
