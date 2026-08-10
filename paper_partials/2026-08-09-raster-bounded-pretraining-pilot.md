# Raster-bounded foundation-pretraining pilot

Date: 2026-08-09. Status: **rejected after the preregistered seed-71 pilot**;
seeds 72 and 73 were not run.

## Candidate and protocol

`raster-bounded` is exactly the `raster-axial` scientist with one search-side
change: every exact Simplifier state receives the certified lower bound

\[
u(K)\geq\max(|\sigma(K)|/2,|\tau(K)|,
2\;\mathbf 1[H_1(\Sigma_2(K))\text{ non-cyclic}]).
\]

If `C` is the semantic episode cap and `c_spent` is the charged cost already
paid, the network value is clamped to the valid solver ceiling

\[
1-2\min((c_{spent}+A\,u_{floor}(K))/C,1).
\]

The candidate has the same axial raster, 64 channels, four residual blocks,
optimizer, auxiliary heads, L10/L1000 mixture and training schedule as
`raster-axial`. The run used seed 71, `F=(5,8,12,16)`,
`F_old=(1,2,4,8)`, simulations `(64,128,256,512)`, 10 evaluation attempts per
objective and an 80% promotion threshold. The executable-source hash and full
candidate spec are in the run manifest.

The certified backend was an explicit prerequisite. Before training, it was
informative on five of the six canonical foundation sources (83.3%), with exact
crossing-change floors `0,1,1,1,2,2`. The first attempted run also exposed and
led to a fix for tuple/list manifest normalization on resume; the results below
come only from the clean v2 run.

Artifact:
`pgx-mcts-bench/artifacts/current/semantic-moves-v1/foundation-pretrain-raster-bounded-v2-20260809`.

## Pretraining results

`SR` is the pooled held-out solve rate across L10 and L1000. `cc` is conditional
on solving and is compared with the theorem-known optimum `u`. `Info` counts
search value evaluations with a nonzero certified floor; `Bind` counts those for
which the floor actually changed the network value. `F` is the cumulative
training-dose cap for that attempt.

| Rung | Source | F | Sims | Iterations | SR | cc / u | Info / evaluations | Bind / evaluations | Result |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | unknot + 2 | 5 | 64 | 2 | 80% | 0 / 0 | 12,807 / 17,154 (74.7%) | 13,054 / 17,154 (76.1%) | promoted |
| 1 | T(2,3) | 5 | 64 | 4 | 100% | 1 / 1 | 78,756 / 99,978 (78.8%) | 3,525 / 99,978 (3.5%) | promoted |
| 2 | P(3,4)#0 | 5 | 64 | 2 | 100% | 1 / 1 | 3,267 / 5,631 (58.0%) | 989 / 5,631 (17.6%) | promoted |
| 3 | P(4,5)#0 | 5 | 64 | 5 | 0% | unsolved / 1 | 108,029 / 230,874 (46.8%) | 236 / 230,874 (0.1%) | capped |
| 3 | P(4,5)#0 | 8 | 128 | 8 | 100% | 2 / 1 | 269,595 / 389,637 (69.2%) | 1,251 / 389,637 (0.3%) | capped: non-optimal |
| 3 | P(4,5)#0 | 12 | 128 | 12 | 100% | 2 / 1 | 343 / 6,658 (5.2%) | 2,402 / 6,658 (36.1%) | capped: non-optimal |
| 3 | P(4,5)#0 | 16 | 128 | 16 | 100% | 2 / 1 | 208 / 7,597 (2.7%) | 3,573 / 7,597 (47.0%) | rejected: non-optimal |

Across completed attempts, the search requested 757,529 certified values. The
floor was informative 473,005 times (62.4%) and binding 25,030 times (3.3%).
Informative methods were signature 439,204 times, non-cyclic double-cover
homology 33,716 times and tau 85 times. The clean pilot occupied 67.9 wall-clock
minutes on a machine simultaneously running the remaining foundation roster.

The rung-0 informative values are not false claims that the source unknot is
knotted. Ten independently generated root representations were checked and all
had floor zero. Informative descendants occur after an MCTS branch has already
paid a crossing change and thereby changed the current knot; `spent +
u(current)` remains an admissible branch-specific lower bound.

## Matched seed-71 comparison

The ordinary `raster-axial` seed used the same source order and adaptive
schedule. Both candidates promoted rungs 0--2. On the decisive first
four-strand knot:

| Candidate | Base attempt | Adaptive retry | Final four-strand result |
|---|---|---|---|
| raster-axial | F=5, 64 sims, 50% SR, capped | F=8, 128 sims | 100% SR, **1/1 cc**, promoted |
| raster-bounded | F=5, 64 sims, 0% SR, capped | F=8,12,16 at 128 sims | 100% SR, **2/1 cc**, never promoted |

The bounded arm therefore demonstrated four-strand solving capacity, but it did
not demonstrate foundation-pretraining quality. A lower bound can rule out an
over-optimistic value; it does not provide the policy route that realizes the
bound. Here the altered MCTS targets converged to a reliable but dominated
two-crossing solution and never recovered the one-crossing solution found by the
matched baseline.

## Decision

Do not finish the three-seed pretraining and do not add `raster-bounded` to the
scientist roster in its current form. Preserve the latest promoted checkpoint
(through rung 2) and the failed progress checkpoint as diagnostic artifacts,
not as released pretrained scientists.

A future version should be a separate experiment: use the certified floor for
branch-and-bound only after a verified incumbent exists, or add an explicit
bound-gap training target. Either design must first show that it preserves the
matched `raster-axial` one-crossing solution on `P(4,5)#0`.

## 256-simulation rescue

The original adaptive rule did not raise search above 128 simulations once solve
rate reached 100%, even though the L1000 result remained non-optimal. We therefore
ran the missing 256-simulation test on the exact post-F16 `stage03-after` weights.

| Test | Training | L1000 | L10 | Result |
|---|---:|---:|---:|---|
| Frozen checkpoint, 256 simulations | none | 100% SR, **2 cc + 6 moves** | 100% SR, **1 cc + 7 moves** | no search-only rescue |
| Warm-start rescue, 256 simulations | 16 iterations, 128 optimizer steps each | 100% SR, **2 cc + 6 moves** | 100% SR, **1 cc + 7 moves** | no training rescue |

The frozen evaluation used the same ten fixed attempts per objective and seed as
the original rung evaluation. The training rescue took 831.1 seconds. Because a
stage snapshot stores network weights but not optimizer or replay state, this was
explicitly a fresh AdamW optimizer and fresh replay buffer rather than a
bit-for-bit continuation; auxiliary value and encoder training were active from
its first update.

The result strengthens the rejection: doubling MCTS from 128 to 256 neither
changed the frozen L1000 route nor allowed a full new training dose to learn the
known one-crossing solution. The machine-readable report is
`pgx-mcts-bench/artifacts/current/semantic-moves-v1/raster-bounded-256-rescue-20260809/report.json`.
