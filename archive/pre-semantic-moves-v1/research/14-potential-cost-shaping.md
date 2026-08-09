# Potential-based objective-cost shaping

**Status (2026-08-02): implemented and tested as an opt-in experiment; not enabled
for the production ladder.** It makes the critic's regression problem better
conditioned and showed a small aggregate gain, but did not robustly remove the
ratio inversions that motivated it.

## Construction

Let `lambda = A/B`, `H` be the Simplifier budget, and

```
C(s) = lambda * crossing_changes(s) + moves_used(s)
W    = (lambda + 1) * H
Phi(s) = -2 C(s) / W
```

for nonterminal states, with `Phi(terminal) = 0`. The Simplifier receives

```
r'(s,a,s') = r(s,a,s') + Phi(s') - Phi(s)
```

and the Scrambler receives its negative. Because the search discount is one and
the initial potential is zero, every complete trajectory keeps exactly its old
return. Solves still return `1 - 2 C(final)/W`; failures still return `-1`.

Merely inserting these edge rewards and converting the old leaf value would
cancel algebraically inside MCTS and change nothing. The experiment therefore
trains the scalar head on the shaped remaining-return target

```
V_target'(s) = terminal_return - Phi(s).
```

This stays in `[-1,1]`: on a solve it is `1` minus normalized *remaining* cost;
on a failure it is `-1` plus normalized cost already paid.

## Controlled protocol

The source was the completed rung-21 `s-window-128` checkpoint, evaluated on
`R(3,12)#0+4`. For each of two independent seeds:

* one unmodified checkpoint generated 24 self-play episodes at 128 simulations;
* control and treatment received exactly the same 422--429 positions, minibatch
  sample sequence, optimizer state, 128 updates, and policy targets;
* only the scalar value targets differed;
* both clones were evaluated at temperature zero and without root noise on the
  same 12 Markov-equivalent depth-4 representations at `1000:1`, `10:1`, and
  `1:10`; and
* every solved trajectory was replayed into a complete semantic witness.

The treatment's shaped-target MAE fell from `0.121/0.122` to `0.052/0.051` in
the two seeds. The terminal-target controls fell from `0.062/0.064` to
`0.045/0.049`.

## Results

All costs below are conditional on solved episodes. `objective` is the actual
`lambda * crossings + moves` optimized at that ratio.

| seed | arm | ratio | solved | crossings | moves | objective | inversions |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20260802 | terminal control | 1000 | 12/12 | 5.000 | 18.250 | 5018.250 | 18 total |
| 20260802 | shaped treatment | 1000 | 12/12 | **4.333** | 25.000 | **4358.333** | 17 total |
| 20260802 | terminal control | 10 | 12/12 | 4.500 | **16.583** | **61.583** | |
| 20260802 | shaped treatment | 10 | 12/12 | 4.500 | 16.667 | 61.667 | |
| 20260802 | terminal control | 0.1 | 12/12 | 5.167 | 17.083 | 17.600 | |
| 20260802 | shaped treatment | 0.1 | 12/12 | **4.250** | **16.000** | **16.425** | |
| 20260803 | terminal control | 1000 | **12/12** | **5.750** | 19.333 | **5769.333** | 27 total |
| 20260803 | shaped treatment | 1000 | 11/12 | 5.909 | **19.000** | 5928.091 | 27 total |
| 20260803 | terminal control | 10 | 12/12 | 4.583 | 17.583 | 63.417 | |
| 20260803 | shaped treatment | 10 | 12/12 | 4.583 | 17.583 | 63.417 | |
| 20260803 | terminal control | 0.1 | 11/12 | **4.636** | **16.182** | **16.645** | |
| 20260803 | shaped treatment | 0.1 | **12/12** | 4.667 | 16.333 | 16.800 | |

Pooled over the two independently trained clones, shaping changed inversions
from `45` to `44`. It improved mean objective at `1000:1` (`5393.792` to
`5109.087`) and `1:10` (`17.143` to `16.613`), and was effectively tied at
`10:1` (`62.500` to `62.542`). But the high-ratio improvement came entirely from
one seed; the other seed lost one solve and became worse on solved-case cost.

## Verdict

Keep the implementation as an opt-in research switch, but do not turn it on for
the ladder yet. It demonstrably teaches the intended remaining-return quantity,
and the first seed's trajectory improvement is too large to dismiss. However,
the cross-ratio dominance pathology is essentially unchanged and the performance
gain is not seed-stable. Exact accumulated cost was not the main missing signal.

The next deciding experiment would use at least five training seeds and a larger
shared replay buffer, and would compare this scalar shaped critic against the
already implemented factorized `(p_solve, crossings, moves)` critic. The latter
matches the causal structure directly and may exchange counterfactual costs
between ratios rather than merely re-centering one scalar.

Machine-readable artifacts:

* `../../pgx-mcts-bench/artifacts/potential-shaping-s-window-r21-20260802/`
* `../../pgx-mcts-bench/artifacts/potential-shaping-s-window-r21-seed2-20260802/`

## Five-seed factorized-critic comparison

The proposed deciding experiment was run with seeds `20260802` through
`20260806`.  Each seed used 64 shared self-play games (1,025--1,080 positions),
256 paired optimizer updates, and 20 shared depth-4 representations evaluated at
all three ratios.  The shaped-scalar and factorized arms began from identical
checkpoint and optimizer states.  Every solved evaluation was replayed and
stored as a verified semantic witness.

| arm | ratio | solved | crossings | moves | objective |
|---|---:|---:|---:|---:|---:|
| shaped scalar | 1000:1 | **92/100** | **4.598** | 20.380 | **4618.207** |
| factorized | 1000:1 | 90/100 | 4.944 | **20.356** | 4964.800 |
| shaped scalar | 10:1 | **98/100** | **4.633** | 17.408 | **63.735** |
| factorized | 10:1 | 96/100 | 4.760 | **17.385** | 64.990 |
| shaped scalar | 1:10 | **100/100** | **4.700** | **16.670** | **17.140** |
| factorized | 1:10 | 95/100 | 5.053 | 17.053 | 17.558 |

The solve comparison is paired: there were nine shaped-only solves and no
factorized-only solves.  On jointly solved cases, factorized objective
win/tie/loss counts were `3/71/16`, `3/85/8`, and `3/84/8` from high to low
ratio.  Restricting the inversion comparison to the 84 representations solved
by both arms at every ratio avoids rewarding failures: shaped scalar produced
136 cross-ratio dominance inversions and factorized produced 140.

This rejects switching ladder MCTS to the current factorized critic.  It is not
evidence that the decomposition itself is wrong.  All 5,288 training positions
in this comparison came from solved self-play episodes, so `p_solve` saw no
negative labels and reached a misleading mean in-buffer Brier score of about
`1.1e-6`.  The next factorized experiment, if pursued, must first construct a
calibration buffer with failures and hard near-failures, then validate held-out
solve calibration before allowing the composed value into search.

Full reports, checkpoints, witnesses, summary, and checksums:

* `../../pgx-mcts-bench/artifacts/factorized-critic-five-seed-r21-20260802/`
