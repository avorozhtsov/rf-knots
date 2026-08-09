# Serial formulation: current contract

A serial scientist observes a local braid window and may move a private head,
read/write private tape, or update finite control between semantic knot actions.
Those internal operations make the architecture more expressive, but they are not
part of the mathematical solution.

The experiment therefore maintains two clocks:

- the scientific objective charges crossing changes and semantic braid moves;
- the internal-step budget limits head shifts, tape writes, and controller-state
  changes without adding them to `L_AB`.

The observation includes the requested objective ratio, remaining semantic
objective budget, and remaining internal budget. These features enter the shared
body and all value heads. The policy, crossing-change, semantic-move, and
`p(solve)` losses can all train the shared representation.

Stable current serial families are `window-local` and `cyclic-memory`; the latter
adds a full-word cyclic encoder and persistent tape. Both are trained from scratch
under the same source-disjoint curriculum as raster and strand-graph scientists.

The historical serial ladder—including results that charged head travel as
`moves`—is preserved in
[`../archive/pre-semantic-moves-v1/research/12-serial-formulation-full.md`](../archive/pre-semantic-moves-v1/research/12-serial-formulation-full.md).
Those objective values and Pareto comparisons are not active evidence.
