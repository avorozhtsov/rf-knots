# 15 — Recombination triad attempt

This note records one concrete attempt under the recombination direction in
[13](13-directions.md#21-composing-networks-into-larger-creatures). It is an
experiment, not itself a direction.

## Fixed window/scan/tape triad

The composite `s-triad-wst` combines three independently trained, initially frozen
towers whose roles are deliberately different:

* `s-window-128` supplies positional local editing and may act anywhere in its
  seven-cell window;
* `s-scan-gru` supplies a forced full-necklace scan and global recurrent summary;
* `s-tape4` supplies a persistent four-symbol annotation tape transported through
  braid rewrites.

The experiment is pinned to the completed-rung snapshots visible when it was
approved on 2026-08-01. These identities are part of the candidate definition,
not replaceable aliases for "latest":

| tower | completed rung | checkpoint SHA-256 |
|---|---:|---|
| `s-window-128` | 18 | `e6d4285f7b268f123f683d80e6cfcf7daef0b1d8145f7b559054ce19e6a8949c` |
| `s-scan-gru` | 10 | `ed49509639dc198575c0ec919b000716bfb983e7c0db21d38cfc65d6214ab0b3` |
| `s-tape4` | 8 | `ddf87bd8a019bc721567123cfd5d62596298cfe9363e1c188dfa88e5fd18697f` |

The child inherits **no rung credit** from them. Before training, it is evaluated
from rung 0 upward; the first rung that fails the ordinary solve-rate, worst-ratio,
or known-unknotting-number criterion is its recommended training start.

The frozen two-seed sweep fixed that start at **rung 10, `T(3,4)+2`**. Both
seeds cleared rungs 0--7 at the exact known crossing optimum. At rung 8 their
pooled `A:B=1000:1` cost was 42 crossing changes over 19 solves, or 2.21 against
`u=2`, within the 0.25 tolerance; pooled solve rate across all ratios was 67/72.
Rung 9 was exact at 3.00. At rung 10 the two crossing-dominant means were 3.50
and 3.25 against `u=3`, averaging 3.375 and failing the tolerance. The deployed
ladder still starts the child at rung 0 so these clearances are reproduced in its
own log; rung 10 is where useful mixer training is expected to begin.

## Fusion design

Raw logits cannot be averaged directly: each independently trained softmax head is
free to choose its own additive offset and temperature. For each tower, centre its
represented logits and divide by their RMS. For action `a`, let `S(a)` be the
towers that actually represent it, and let a zero-initialized router produce
scores `z_i(a)` from the concatenated penultimate features. The fused logit is

```
F(a) = sum_{i in S(a)} softmax_{S(a)}(z(a))_i * normalized(F_i(a)) + R(a)
```

where the residual head `R` is also initialized to zero. Thus the initial policy
is exactly the average of the available normalized opinions: an action gets no
automatic advantage merely because three towers rather than one can name it.
Missing actions contribute nothing rather than a misleading zero logit. The same
features feed zero-residual fusion of the advanced solve-probability, conditional
crossing-change, and conditional-move estimates.

The shared semantic action space is the union of the parents' actions. In
particular, **shift while preserving the tape** is distinct from shifts that write
symbols 0 through 3: writing symbol 0 erases a mark and is not the non-tape
parents' old shift. Freeze all three towers and their BatchNorm statistics during
the first mixer-only phase. Only after the frozen frontier and mixer warm-up are
measured may the policy/value heads be unfrozen at a smaller learning rate; trunk
unfreezing is a separate ablation.

## Failure criteria

The attempt fails if frozen averaging fails earlier than the weakest parent; if a
mixer improves the training prefix but loses solve rate on either parent's
specialty instances; or if inference cost is high enough that matched-compute
search performs better with one parent. Compare the triad against all three pinned
parent snapshots, not against subsequently promoted replacements.
