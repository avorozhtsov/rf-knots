# Nine-step experiment plan

Status: **registered protocol; steps 1–2 complete, step 3 running**.

The active, detailed plan is the “Recommended nine-step plan” in
[`research/16-scientists-collaboration.md`](../research/16-scientists-collaboration.md).
The roster decision is now resolved: `raster-axial` remains the fourth scientist;
`raster-routed` is an unadmitted development architecture.

## Execution record

- The semantic-move objective and verifier contract are implemented.
- The prior partial foundation run was stopped resumably because it used the old
  local raster roster and a target-biased 1:3 objective mixture.
- Foundation sampling is now neutral: L10 and L1000 each have weight 1.
- `raster-routed` now has a full global raster, shared routed row-pair policy,
  dynamic torus seam, dilated recurrent block, distinct content/workspace masks,
  and layer-wise objective/budget conditioning.
- Structural and roster tests passed before the learning gate started.
- The registered matched admission compares `window-local`, `raster-axial`, and
  `raster-routed` from random initialization under identical seed and MCTS
  simulation counts, while reporting parameter count and wall time separately.

## First admission result

The local raster cleared the four-strand stage at 10/10 and optimal crossing cost.
The window baseline reached that stage but solved 0/10. The routed raster solved
10/10 on its first rung, but used 0.3 unnecessary crossing changes on average and
therefore failed the known zero-crossing objective after six iterations. Its wall
time was 932 seconds for that rung, versus 711 seconds for all four local-raster
rungs.

The exact-state adaptive `F=8` continuation kept solve rate at 10/10 but worsened
mean crossing changes from 0.3 to 0.6. This rejects the slower-objective-learning
rescue at the declared dose. The first big experiment therefore retains
`raster-axial` as the fourth scientist and keeps `raster-routed` as a separate
capacity research direction.

The corrected four-candidate, three-seed foundation pretrain stopped after 11 of
12 jobs once the three-scientist selection became invariant to the unfinished
job. It used the neutral 1:1 objective mixture. Its v2 manifest embeds full candidate specs,
the Git base, dirty-status hash, and executable-source hash.

No long-run checkpoint may be selected from the development gate. Foundation
the selected checkpoints are stored in the separate semantic-v1 artifact root and
frozen manifest provenance.
