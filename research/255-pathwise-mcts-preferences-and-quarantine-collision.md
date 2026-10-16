# Pathwise MCTS preferences recover 15 conditions without weakening quarantine

2026-10-02. User identified that discarding 15 of 19 cost-varying conditions
because any one continuation crosses a non-TRAIN fingerprint is poor training
efficiency. This is a confirmed *whole-condition eligibility* loss, not evidence
that MCTS produced only four useful comparisons. No frozen55/248 source, data,
checkpoint or running worker was changed; this is a new preference overlay for
future training, not a resumed or modified 248 experiment.

## Cause and exact scope

In the completed collection, 256 root/weight conditions contain 1,494 saved
continuation episodes. The old supervision builder admits only complete,
replayed TRAIN-owned successful paths and a TRAIN-owned reference witness for
positive targets. Yet after constructing cost preferences it clears the ENTIRE
preference group whenever `all_episodes_train_owned` is false, even if multiple
admitted complete solutions of different cost remain. There are 19 conditions
with >1 admitted unique success path and >1 distinct cost; 15 fail the
whole-condition gate and only four retain relative preferences. Among those
15, twelve have an episode entering validation, nine have an episode entering
test, and one has an episode entering the `(49,0)` quarantine (overlap between
validation and test categories is possible across episodes). These categories
must not be summed to 15.

`scripts/build_pathwise_mcts_preferences.py` constructs a NEW overlay from the
same frozen collection, audit and v1 targets. It replays the permitted witnesses
and refuses any witness whose path fingerprint belongs to validation/test,
quarantine or an excluded held-out query group. It recomputes cost preferences
only among these complete TRAIN-owned successes, preserving the original
positive target set and all four old admitted preference groups exactly.
Incomplete/censored continuations never become negative action labels. It
does not change the old partition, old source, or the current training run.

The first preparation attempt stopped because in-memory tuples were compared
with JSON-loaded lists. That was a serialization-only false mismatch; no
preference output was committed. After normalizing the comparison, an initial
overlay was produced but its source pin became stale during formatting; it was
preserved as `artifacts/diverse-mcts-pathwise-preferences-v1` and is explicitly
UNVERIFIED. The corrected separate output is
`artifacts/diverse-mcts-pathwise-preferences-v2`.

| Measure | Frozen v1 | New v2 |
| --- | ---: | ---: |
| Conditions with admitted relative preferences | 4 | 19 |
| Aggregated state/action preference terms | 514 | 2,859 |
| Newly recovered conditions | — | 15 |
| Newly recovered terms | — | 2,345 |

These terms are correlated path/state observations, not independent knot types
or 2,859 new MCTS runs. `scripts/audit_pathwise_mcts_preferences.py` separately
replays all 63 source-witness occurrences, rechecks TRAIN ownership and all
input/shard hashes, recomputes winner/loser rational coefficients and common
state/action cancellation, verifies the unchanged candidate pool and all old
admitted preferences, and checks exact coverage/counts. Its result is
`artifacts/diverse-mcts-pathwise-preferences-v2/independent-audit.json` with
`status=passed`. Ruff passed for both new scripts; eleven relevant existing
MCTS supervision/collection tests passed. This certifies the overlay's data
integrity, not a trained model or downstream search improvement.

## The `(49,0)` collision and Jones diagnostic

The quarantine key means `(determinant, absolute knot signature)`, not cost,
coordinates or CC count. Condition51 has five saved MCTS continuation episodes;
one episode (simulation8) reaches `(49,0)` after its 35th full-path move, at a
13-letter, four-strand knot state. The key is deliberately over-broad: unrelated
knots can share it. Direct exact Jones-polynomial computation using the local
`rf_knots.invariants.jones_polynomial` distinguishes this state from the
20-letter Brittenham-Hermiller B5 representative of `7_1 # m(7_1)`:

- protected B5 representative: 15 Jones terms, exponents -7..7;
- condition51 state: 10 Jones terms, exponents -3..6;
- the two complete polynomials are unequal.

This proves the encountered state is NOT isotopic to that particular protected
B5 target knot, because Jones is invariant under knot isotopy. It does NOT
prove the state is outside the deliberately protected one-crossing-change
neighbourhood, the other named chain knots, or all 32 Wang-Zhang mask states.
The reproducible replay and full polynomial coefficients are saved in
`artifacts/quarantine-jones-collision-v1.json` (SHA256
`63b4fb0be3c97ec6920cde7c08c557f55b860c2d73d29e5152d663bbfcc425dc`);
the probe's source hash and pinned collection/shard hashes are included there.
Therefore no existing quarantine/split rule was relaxed. The pathwise fix
already recovers condition51's safe comparison while excluding its one
quarantined episode. A Jones-refined global quarantine requires a separately
versioned full protected-set/near-neighbour audit, not just one counterexample.

## Why this matters but does not yet solve MCTS learning

The four old groups were not mathematically zero-gradient: the earlier
TRAIN-only diagnostic119 measured large scheduled relative-policy gradients,
sometimes dominating the metric gradient. The actual failure was tiny target
diversity plus a conservative whole-condition veto. This overlay increases
diversity and eligibility but is NOT yet wired into a trainer, so current248
still uses the old four groups. Next, use the verified overlay as an explicitly
new input in protocol254's matched CPU pilot, measure relative gradient share
and diversity per root, and compare downstream DEV search before investing a
large CPU budget. Do not use observed TEST150 for selection.

The proactive contribution gate for every future MCTS-to-policy pipeline is:
(1) count collected complete successes, cost-varying TRAIN-owned groups, and
distinct divergence contexts *after* every filter, not only before filtering;
(2) inspect actual draw frequency, weighted encoder-gradient norms and
cosines against metric/demo tasks, with clipping effects; (3) test whether
the relative objective improves on held-out DEV groups and whether a matched
policy ablation changes DEV search success/CC at equal search compute; and
(4) compare gain per collection/training/search CPU hour. A positive gradient
is only evidence that a term changes weights; it is not evidence that it helps
the search. If eligibility collapses, a few groups dominate updates, or DEV
search does not improve, revise collection/targets/weighting before extending
training. This check should be automatic before a multi-day run, not prompted
by the user after it starts.
