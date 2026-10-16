# Fixed UnknotDB graph-branch translation sample v1

2026-10-13. This is **candidate supply**, not F-label admission or an MCTS/training run. Pin proof snapshot `/Users/artemvorozhtsov/projects/unknotdb/outputs/unknotdb-dgkt-witness-coverage-v1-final.sqlite` SHA256 `e31028def491caab7317ab63eb9829a1e63e1fa9a512f2ccf4daac6d178ffd20` and matching routing sidecar `/Users/artemvorozhtsov/projects/unknotdb/outputs/unknotdb-dgkt-routing-v1.sqlite` SHA256 `673e0fcbae9c7b582e44fed8e92e09ec0716331e80a1c82a81c48bd7c9b77001` (`meta.proof_sha256` matches). Both were accessed read-only.

Selection was score-blind with respect to RF/PGX outcomes. Start from the 22,214 graph nodes in research/352 with two distinct encoded first actions, different target nodes and different `edge.l1000_cost + target.l1000_score`. Exclude graph-optimal score zero, then divide by graph-optimal score into `[1000,3000)`, `[3000,5000)`, `[5000,7000)`, and `>=7000`; eligible counts are respectively 5,052, 12,238, 4,474 and 233. Within each band choose the four smallest SHA256 strings of `20261013:<node_id>`. Do not replace a selected node after seeing route or translation failure. For each selected node, consider pairs of outgoing edges with different encoded first actions, different targets and different graph L1000 costs; reject any edge whose continuation through `routes.next_l1000_edge` repeats a graph node, then choose the smallest SHA256 of `20261013:<node_id>:<lower_edge_id>:<higher_edge_id>` among remaining pairs. Graph routes must terminate at score zero and sum exactly to the stated L1000 score. This cycle filter removes trivial graph detours before any F claim.

| Band | Node | Edge pair | Graph L1000 route costs |
|---|---:|---:|---:|
| 1–2k | 106255 | 35021, 158909 | 2009, 1004 |
| 1–2k | 6132 | 33576, 118411 | 3009, 2008 |
| 1–2k | 9009 | 82345, 110690 | 4004, 2004 |
| 1–2k | 51535 | 21480, 84376 | 4004, 2004 |
| 3–4k | 7151 | 34260, 116595 | 5011, 3007 |
| 3–4k | 65707 | 16089, 104019 | 4005, 5006 |
| 3–4k | 82922 | 77993, 99138 | 3003, 3005 |
| 3–4k | 61145 | 15025, 105752 | 5006, 3004 |
| 5–6k | 54793 | 9566, 82623 | 6006, 5005 |
| 5–6k | 79520 | 2963, 108442 | 5007, 5009 |
| 5–6k | 42861 | 27518, 107843 | 7009, 5007 |
| 5–6k | 71484 | none after cycle filter | — |
| >=7k | 58409 | 69479, 119509 | 9019, 7015 |
| >=7k | 88914 | none after cycle filter | — |
| >=7k | 11394 | 36900, 118798 | 8016, 7015 |
| >=7k | 63897 | 120071, 154518 | 9032, 8009 |

The graph-side read-only route walk found 14/16 selected nodes with a cycle-free candidate pair and 2/16 with none. It verified edge connectivity, absence of repeated node IDs, terminal graph score zero and edge-cost sums for the listed pairs. **It did not decode or replay semantic macro programs, unquotient mirror/origin coordinates, translate actions to canonical PGX, establish TRAIN-owned distinct invariant query groups, apply heldout/viewed exclusions, or check paired L10/L1000 caps.** Zero new F roots are admitted. In particular, the tiny cost differences at nodes 82922 and 79520 may be unsuitable after PGX replay; do not replace them.

Next bounded gate: for these 16 nodes only, use a separately versioned read-only decoder to replay both full UnknotDB routes including chirality/origin bookkeeping, translate exact root and each semantic action to canonical PGX, replay there, compute exact root ownership and viewed/TEST exclusions, and require root first-successor divergence and strict admissible L10 or L1000 cost difference. Log source/checkpoint SHA, wall/CPU and failure reason per selected node. Stop this source if translation fails generally or fewer than four **distinct** new TRAIN query groups survive; even four only permit a separate prospectively frozen small MCTS pilot, not bank admission without independent audit. No database mutation, cloud purchase, commit, merge or push; preserve old 402-root bank and active R2 trainer unchanged.
