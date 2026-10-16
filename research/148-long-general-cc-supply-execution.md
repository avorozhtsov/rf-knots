# Long general CC supply: launched and advancing

## Hourly observation,2026-09-24 05:32 UTC

The same generator PID39810 remains running,100%CPU,elapsed1h04m32s and
CPU64m18.85s. Its committed progress advanced from192 to4160/6144 attempts;
mtime05:31:45 UTC. All65 committed shard hashes match. Captured progress bytes
SHA256 `36adc26f0021161f91d148b8d2cf876b9a883d515d1287c639e736c99acb1be1`.
This is a live snapshot, not a completed corpus or a full independent replay.

Selected TRAIN2196 pairs:967 exact0,656 exact1,224 exact2,349 intervals;
1870 query-invariant groups, lengths49..383, strands3..33. DEV731 and TEST312.
Every TRAIN quota is met in the prefix. One TEST quota remains short: the
49..96 band has7 exact2 pairs versus the fixed minimum8. Continue all6144
attempts; do not stop early or relax the quota. Generator stdout contains
normal progress, with no new error observed. Session49791 remains active.

Existing139 controllerPID15420/session78472 and childPID21909 are also alive.
Child100%CPU,elapsed2h16m32s,CPU136m09.20s; the controller remains at12/24
completed jobs while the first search configuration is ongoing. Its first
8/128 conditions committed at04:48:03 UTC:48 rollout episodes,22808 policy
neural evaluations,0 solved conditions. This is a prefix of one configuration,
not a terminal solver comparison. The older controller manifest timestamp is
not evidence of a stall: child CPU time and the search log advanced. No process
was interrupted or relaunched. Next routine combined observation no earlier
than06:32 UTC, retaining the user's hourly cadence.

The initial execution evidence follows unchanged.

2026-09-24,04:32 UTC. Protocol147 data construction is VERIFIED ACTIVE, not
completed and not yet eligible for optimization. No public database campaign
was launched or imported; the user's question about that additional scope
remains separate from the active embedding goal.

## Execution

New output did not exist and no generator process was present before launch.
One process was launched at04:27:55 UTC, PID39810, unified session49791:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/build_long_general_cc_supply.py \
  --out artifacts/long-general-cc-supply-v1
```

At04:30:34 UTC the actual process was running at99.1%CPU, elapsed2m40s,
CPU2m38s. Progress had advanced from0 to64 to128, and the subsequent committed
snapshot reached192 of6144 attempts. All1093 input/source pins matched; the
snapshot's selection, counts and minimum-support gaps were recomputed from its
three immutable committed shards. No error was emitted by the generator.
The optional Plink tkinter warning concerns an unused GUI, not invariant or
certificate failure.

| Committed shard | SHA256 |
| --- | --- |
| attempts-000000.json | `e34bcb4ecaccba78241a4df2e6c41e8f32e77f758913b2727000fc1f66f0f7a2` |
| attempts-000064.json | `eb6a65324331ea8270e9a03358f64c20b75ef019c6f0ebfa19c741eb11ba60d7` |
| attempts-000128.json | `7f5769179b06a2de169312aeae3ac9e9c26155d550174e9d7da91f02defe7298` |

Captured192-attempt progress-payload canonical digest:
`ede1bc1be0cb710fc33dc9193d73c07d2eacfa510d493bf5309b357c81ccffd5`.
This is the captured JSON-value digest, NOT a claim that the live progress file
has stopped changing or has this byte-level SHA.

An initial observation used the strict post-exit shard-directory checker while
the producer was still writing. It rejected an additional shard published after
the observer's manifest snapshot. The corrected observation checks exactly the
hash-bound committed prefix of a fresh manifest. This was an observation race,
not a generator restart, missing evidence, or permission to overwrite a shard.
The strict directory checker remains appropriate for post-exit/resume consumers.

## Interim supply, not the final denominator

| Split | Accepted unique canonical pairs | Exact0 | Exact1 | Exact2 | Intervals | Query groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN |100|37|37|14|12|97|
| DEV |34|17|8|2|7|34|
| TEST |21|9|6|5|1|21|

TRAIN endpoint lengths51..376, strands3..24 in this partial prefix. Exact2
TRAIN counts in the three length bands are8/3/3; exact1 counts13/14/10.
All14 exact2 TRAIN pairs have long endpoints, unlike the zero long exact2 pairs
in the old ordinary-graph source. This is concrete new support but not evidence
of sufficient variety, model improvement, independent knot types, or readiness.
The frozen minimum per-class/per-band gates are still unmet. Continue all6144
attempts and preserve the specified thresholds even if some fail at completion.

## Independent consumer and validation

Added `scripts/audit_long_general_cc_supply.py`, without modifying the running
builder or historical sources. It requires a completed hash-bound corpus,
recomputes every deterministic construction and selection/counts, and separately
replays every selected pair with fresh integer invariants and Burau checks.
It checks all TRAIN path states, fresh DEV/TEST query ownership and exclusions,
cross-split pair uniqueness, size bands and all redundant fields. Rejected
invariants remain checked by the pinned builder, not re-replayed by the consumer.
The report explicitly distinguishes certificate validity from minimum-support
readiness. The corpus itself remains immutable with optimizer_ready=false;
downstream training must require the passed consumer's training_ready flag.

Targeted suite:46 tests passed (26 builder,11 consumer,9 historical138), Ruff
passed. Tests include real certificates, mutation rejection, path ownership,
query exclusions, interval-versus-exact quotas, duplicates and shard coverage.
These tests are not a completed full-data audit. A finite spot-check of three
selected exact2 TRAIN certificates, one per length band, completed with the
separate consumer (session75049, exit0):

| Attempt | Word length | Strands | Signed signatures along the two-CC path |
| --- | ---: | ---: | --- |
|38|62|7|-14,-16,-18|
|23|134|5|-38,-40,-42|
|8|288|3|-80,-82,-84|

Literal sign-flip replay provides upper2, and endpoint signature difference4
provides lower2. Every state's determinant matched the separate integer Burau
calculation. This independently checks real long exact2 supply, not the entire
corpus or the diversity/independence gates; the full post-exit audit remains due.

After generator exit, run once with a new output:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/audit_long_general_cc_supply.py \
  --data artifacts/long-general-cc-supply-v1 \
  --out artifacts/long-general-cc-supply-v1/consumer-audit.json
```

Only after this audit and sufficient support: freeze and execute a matched
new-data training experiment, retain short-u supervision and use fresh search
queries. No new model training has begun here; don't count construction as
optimizer updates or MCTS evaluations as online learning.

Existing139 controller/worker remain unchanged. Their most recent live check
was04:21 UTC in the preceding user-answer turn, not repeated by this generator
work. Next routine combined process inspection should be no earlier than05:32
UTC, respecting hourly monitoring. Existing dirty work and artifacts in all
three repositories were preserved; no commit, push, release or cloud action.
