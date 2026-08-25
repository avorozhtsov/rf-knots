# Single-knot mastery: v1 exhaustion and v2 interim evidence

Status: preliminary live snapshot, 2026-08-16 00:15 UTC. The source table is
[`data/2026-08-16-single-knot-mastery.csv`](data/2026-08-16-single-knot-mastery.csv).

## Result

None of the five scientists had produced a strict challenge success, a genuine
upper-bound improvement, or a replay certificate at this snapshot. This is a
negative upper-bound-search result, not an absence of learning: the v2
scientists produced 196 relaxed training successes and 302 replayable training
witnesses, but those outcomes do not satisfy the stricter challenge target and
must not be reported as improvements.

The terminal strand-graph run is the clearest exhaustion result. It introduced
all 240 challenges, executed 5,854 challenge attempts and 11,708 native
training updates, and ended with all 240 challenges at `gave-up-threshold` and
an empty heap. Its last non-null frozen-panel retention was 0.71875, below the
0.80 target. The final program-state SHA256 is
`40a18f397eed295cef156229853be1dbab75c1723f45bd0324f69a682565f413`.

The raster-axial v1 scientist remained a useful control: at its exact
group-160 checkpoint it retained 0.96875 of the frozen panel, but had no strict
success. An external, provenance-writing guard verified that checkpoint and
froze the service.

Among the v2 candidates, combined-dual and cyclic-memory are the productive
continuations. Combined-dual had contributed 148 replayable witnesses and
cyclic-memory 129, while maintaining retention 0.90625. Jones contributed only
25 witnesses and remained at maximum rehearsal 0.50 with retention 0.50. At
its exact group-100 checkpoint, Jones had 207 heap nodes, 9 hard timeouts, only
5 relaxed successes, and no strict success. This is evidence of poor retention
and throughput under the present protocol, but it does not by itself prove
that the 257,890-parameter Jones model is intrinsically too small. Jones was
frozen at that verified group-100 checkpoint; any width claim requires a
controlled paired width/retention ablation.

## Snapshot table

| scientist | protocol | introduced | step | native updates | attempts | strict successes | relaxed successes | hard timeouts | supported failures | evidence | heap | max observed lineage depth | rehearsal | latest retention | disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| simple-strand-graph | v1 | 240 | 2,927 | 11,708 | 5,854 | 0 | n/a | n/a | n/a | 0 | 0 | 3 | 0.30 | 0.71875 | terminal; frozen at group 240 |
| simple-raster-axial | v1 | 160 | 1,738 | 6,952 | 3,476 | 0 | n/a | n/a | n/a | 0 | 24 | 2 | 0.05 | 0.96875 | frozen at verified group 160 |
| high-combined-dual | v2 | 98 | 379 | 1,516 | 810 | 0 | 124 | 11 | 465 | 148 | 87 | 2 | 0.05 | 0.90625 | continue |
| high-cyclic-memory | v2 | 60 | 218 | 872 | 452 | 0 | 67 | 0 | 137 | 129 | 116 | 2 | 0.05 | 0.90625 | continue |
| high-jones-12 | v2 | 100 | 319 | 1,276 | 652 | 0 | 5 | 9 | 160 | 25 | 207 | 2 | 0.50 | 0.50 | frozen at verified group 100 |

`Attempts` is the sum of attempts stored on introduced challenges. `Evidence`
counts replayable witness files attributed to the scientist in the shared
append-only inventory. `Latest retention` is the latest non-null solve fraction
on the 32-item frozen rehearsal panel at each inherited witness's historical
target. `Max observed lineage depth` includes current heap nodes and recent
event history. Strict successes require a verified solution at the challenge
target; relaxed training successes are useful learning outcomes but cannot
tighten a knot's upper bound.

## Dose calibration

No dose was promoted. Combined-dual remained at SIM32 after 26 paired SIM32/64
observations, cyclic-memory at SIM128 after 8 paired SIM128/256 observations,
and Jones at SIM32 after 7 paired SIM32/64 observations. None of the paired
observations had a success at either dose, so a higher dose could not pass both
the scientific-success and successes-per-CPU promotion criteria. Continuing at
the lower doses is therefore the registered decision, not an arbitrary compute
cut.

## Snapshot evidence and certification

All three v2 scientists consume the immutable evidence snapshot
`best-solutions-pool-20260815T2000Z.json`, SHA256
`1a08b891078438229391b24ee62a6d18ea24083ebdea583b57219dda05be38c7`.
It contains 108 replayable representations mapped to 26 knot identities.
Snapshot distillation had occurred once each for combined-dual and
cyclic-memory and zero times for Jones, after scientist give-up and within the
5% training cap. The live shared inventory contained 302 replayable witness
files mapped to 37 knot identities: 148 combined-dual, 129 cyclic-memory, and
25 Jones.

Certification remains dormant because no genuine improvement exists. The
first future improvement must be replay verified and recorded with its solver
version, simulation parameters, source state, knot identity, starting
representation, action trace, and resulting representation. It may be called
exact only if an independent lower bound reaches the new upper bound.

## Operational provenance

The two stop decisions are implemented by
`scripts/guard_mastery_stop_gate.py` outside the immutable scientist protocol.
The guard waits for an exact group checkpoint, verifies the manifest's
program-state and scientist-state hashes, writes an atomic stop record, and
only then stops the service. A strict success or upper-bound improvement cancels
the stop. It does not alter heaps, training data, curricula, simulation dose,
or checkpoints.

The strand-graph group-240 stop record is
`/srv/braid/artifacts/multi-knot-mastery-v2-20260815/operations/stop-records/simple-strand-graph-group-240.json`.
The verified Jones group-100 stop record is
`/srv/braid/artifacts/multi-knot-mastery-v2-20260815/operations/stop-records/high-jones-12-group-100.json`;
it pins program-state SHA256
`15bbba5d0b143943615b2e383b0e389e4b9f697164bdea32469c0c9aaa114607`
and scientist-state SHA256
`cbb4da7f531beb42bdf590d50933c9ad60d75db2bb3ede6e390bce65c5b50805`.
The verified axial group-160 record is
`/srv/braid/artifacts/multi-knot-mastery-v2-20260815/operations/stop-records/simple-raster-axial-group-160.json`;
it pins program-state SHA256
`cfe439bee589ff72ff716f634668db35bfc1d1b4ed957f0fc6cfc0eb7cb38d69`
and scientist-state SHA256
`25dfc6be76cd38e76e5025e9b03ff83ba33ad9d9bc0127f003b68052e14a6aa1`.
The v1 source tree reported Git commit
`bb886413f21ca549a89a34655d5550c5ea07b75a`; its runtime
tree was dirty, so the immutable source/input/checkpoint hash manifests, rather
than that commit alone, are the executable provenance. The v2 runtime is a
pinned additive migration without Git metadata and is likewise identified by
its source hash manifests, migration records, state hashes, and evidence
snapshot hash.

## Interpretation

The immediate paper-worthy observation is a separation between representation
learning and certified upper-bound progress. V2 can generate many replayable
relaxed witnesses while strict improvements remain zero. Combined-dual and
cyclic-memory justify continued compute because they retain inherited skills
and populate the evidence inventory; Jones does not justify continued compute
past the registered group-100 boundary under its current capacity and schedule.
The completed strand-graph run should be reported as a full 240-challenge null
result rather than silently discarded.
