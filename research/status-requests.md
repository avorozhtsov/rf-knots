# Status-request vocabulary

This document defines the short requests used to inspect the Nebius knot
experiments. The goal is to make repeated status answers compact, comparable,
and reproducible without silently changing metric denominators.

## Run names

| Short name | Canonical experiment |
|---|---|
| `Q-run` | The 12-strand Q4000 curriculum: Q20, Q40-1, Q40-2, Q200-1 through Q200-19, and Q100-tail. |
| `R-run` | The main optimized Semantic-v2 R200 experiment with five arms: static/no-sharing, adaptive/no-sharing, static/sharing, adaptive/sharing, and static-random/no-sharing. |
| `R-L1000-run` | The four-architecture, three-curriculum L1000 ablation: L1000-only, mixed/fixed-total-compute, and mixed/matched-L1000-exposure. |
| `R-Depth-run` | The paired residual invariant-block depth-dose suffix experiment. |
| `R-Invariant-run` | The completed R24 comparison of classical, Alexander-only, Jones-only, combined FiLM, and combined dual-tower invariant scientists. |
| `R-Mutation-run` | The completed paired R11 architecture-mutation experiment. |
| `P-Invariant-run` | The completed five-family invariant foundation pretraining and deterministic checkpoint selection. |
| `DKT72-run` | The frozen five-scientist L1000 evaluation on DKT72-PD-v1. |
| `Recovery-sync` | The MacBook mirror and Google Drive recovery-capsule process. This is infrastructure, not a scientific run. |

Names are case-insensitive. Hyphens, spaces, and harmless spelling errors such
as `adapive` are normalized when the intended run or arm is unambiguous.

## Shared rules

Every live answer begins with a UTC snapshot time. It reads current artifacts
and process state; previously quoted progress is not treated as current.

### Rounds and progress

- A round is counted only after its event is durably committed.
- `up to round N` is **zero-based and inclusive**. Thus `up to round 19`
  means event files `000` through `019`, or the first 20 completed rounds.
- A bare `20/200` means 20 committed rounds out of 200. It does not mean that
  round index 20 has completed.
- A multi-arm compact progress vector is always labelled. For example:
  `{SNS:60, ANS:45, SS:33, AS:63, SRNS:12}/200`. Never print an unlabelled set
  such as `{60,45,33,63,12}/200`.
- If a requested cutoff has not been reached, print `not reached (K/N)` rather
  than extrapolating missing observations.
- For newer transactional runs, report both coordinated rounds and durable
  native scientist-events. For older arms without separate `native-events/`,
  label native results embedded in completed events as `legacy event payload`.

Arm abbreviations are:

| Code | Arm |
|---|---|
| `SNS` | static, no sharing |
| `ANS` | adaptive, no sharing |
| `SS` | static, sharing |
| `AS` | adaptive, sharing |
| `SRNS` | static-random, no sharing |

### Health and resources

`Active` means the expected coordinator and worker processes exist, their CPU
counters advance, and recent logs contain no fatal error. `Stale` means no
committed artifact and no convincing CPU or I/O progress for the run-specific
staleness window. A long round with advancing CPU counters is reported as
`slow`, not `stale`.

Resource columns have precise meanings:

- `cores now`: effective CPU cores consumed by the run's cgroups over a short
  sample, `delta(CPUUsageNSec) / delta(wall time)`. A value of 7.5 means the
  run consumed the equivalent of 7.5 fully busy cores during the sample.
- `average cores`: cumulative CPU seconds divided by active wall seconds. If a
  run crossed service restarts, sum preserved accounting intervals. If only
  the current service activation is available, label the number
  `since current activation`.
- `VM cores now`: the same effective-core calculation for the whole VM when it
  is available. Load average is shown separately as `load`; it is not labelled
  CPU utilization.
- `memory`: current cgroup memory, plus VM available memory for live status.
- `time`: elapsed wall time since the current run or phase began and the age of
  the newest committed artifact.

### ETA

ETA is based on committed-event timestamps, never on process CPU time alone.
Use the median recent seconds per round, preferably over the last ten rounds,
and also inspect a shorter recent window for a throughput change. Report:

1. per-arm ETA;
2. the run's barrier ETA, which is the latest required arm or phase ETA;
3. the sample window and a range, not false precision; and
4. `unknown` when fewer than two comparable rounds have completed or the phase
   has changed materially.

Growing rehearsal banks and adaptive simulation budgets make a simple
whole-run linear projection optimistic. When recent and lifetime rates differ,
the recent rate is primary and the lifetime estimate is shown only as context.

### Scientific metrics

For objective ratio `A`, a failed supported representation has capped cost

`U_A = 20*A + 128`,

so `U_10 = 328` and `U_1000 = 20128` for the current action horizon. A solved
objective is not silently clipped; if a bounded metric is desired, report it
separately as `min(objective, U_A)`.

Every leaderboard defines these columns inline:

- `coverage`: representations with at least one verified solve divided by all
  supported representations in the comparison panel;
- `attempt solve rate`: verified solved attempts divided by all evaluation
  attempts;
- `capped mean`: mean best-per-representation objective after substituting
  `U_A` for a supported failure;
- `portfolio coverage` and `portfolio capped mean`: best verified result across
  the requested scientists, chosen separately for every representation;
- `best cc` and `best moves`: crossing changes and semantic moves for verified
  successes, with conditional means explicitly labelled as conditional;
- `NN evals`: scheduled network evaluations, separated into native training and
  evaluation when both are available;
- `retention`: paired retained solves/attempts and capped cost after rehearsal;
- `wall time`: measured run or scientist wall time, not CPU time.

Skipped Q-run representations and supported-but-unsolved representations remain
in coverage and capped-cost denominators. Capacity-unsupported representations,
as in the old five-strand DKT72 evaluation, are reported separately and never
converted into ordinary search failures.

Unless an objective is named, scientist leaderboards show both L10 and L1000
columns but use L1000 as the primary ordering: coverage descending, capped mean
ascending, NN evaluations ascending, then wall time ascending. L10 gets a
separate rank. No hidden combined L10/L1000 score is used.

### Comparable panels

Within one coordinated arm, scientists are compared on the same committed
rounds. Across arms or curricula, the primary comparison uses the exact
intersection of completed representation identities after applying the
requested cutoff. Always print:

- the number and identity hash of common representations;
- treatment-only and control-only representation IDs, or their counts plus a
  saved machine-readable list when long; and
- exact solved-set intersections and side-only solves.

Do not compare raw prefix averages when adaptive selection or random ordering
produced different task identities. If the common intersection is too small to
support a useful rank, print pairwise tables or `insufficient common panel`
instead of forcing a global ordering.

Sharing comparisons use native results committed before translation or
donation. Donation results may be reported in a separate section but never
substituted for native innovation.

## Status requests

### `Run statuses` or `Runs status`

Print one compact row per named run, active runs first and recently completed
runs second.

Required columns:

| run | phase/progress | workers | cores now | average cores | elapsed | ETA | health/newest artifact |
|---|---|---:|---:|---:|---:|---:|---|

For `R-run`, progress is the labelled five-arm vector. For `R-L1000-run`, show
the three curriculum vectors. For `Q-run`, show the active group and both
coordinated and per-scientist native progress. End with one VM row containing
total effective cores now, total average utilized cores, load, available
memory, and free disk. Do not include scientific leaderboard metrics in this
compact response unless a gate or failure materially changed.

### `<run> status`

Print a detailed operational status for one run.

Required content:

1. snapshot time and overall health;
2. one row per arm, curriculum, group, or scientist process as appropriate;
3. exact committed progress and latest artifact age;
4. worker count, cores now, average cores, memory, elapsed time, and ETA;
5. recent fatal errors or `none`;
6. current phase/barrier and the exact condition for advancing; and
7. a short interpretation of the bottleneck.

`R-run status` prints all five arms and identifies the arm controlling the R200
completion barrier. `Q-run status` additionally prints the group composition,
strand quotas, skip ledgers, capacity exceptions, high-strand native successes,
retention gate, and whether the next group is closed or open.

### `<run> resources`

Print resource accounting without scientific metrics: cgroup/service, expected
and live workers, cores now, average cores, memory, CPU time, elapsed wall time,
newest artifact age, nice level/CPUWeight, and VM totals. Explicitly identify
idle capacity and whether it is safely usable or reserved by priority policy.

### `<run> ETA`

Print recent throughput, lifetime throughput, remaining rounds, per-arm ETA,
barrier ETA, confidence/range, and assumptions. For Q-run, estimate only the
active group until at least one later group provides a defensible scaling
factor; do not project Q4000 from an unfinished first Q20 event.

### `<run> health`

Print only process and artifact health: units, PIDs, worker count, advancing CPU
and I/O counters, artifact freshness, memory headroom, errors, and stale/slow
classification. Do not rank scientists.

## Leaderboard requests

### `<run> leaderboard`

Print the current scientist leaderboard. It contains one row per scientist
instance, not merely per architecture: copies in different arms or curricula
remain distinct.

Required columns:

| rank | scientist instance | arm/curriculum | paired n | L10 coverage | L10 capped mean | L1000 coverage | L1000 capped mean | innovation | native NN evals | evaluation NN evals | retention | wall time |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

The primary table uses the exact common completed representation panel among
the requested scientist instances. A second compact operational table may show
each scientist's own full native panel, but it must not be presented as a
cross-arm rank.

### `<run> leaderboard up to round N`

Apply the inclusive event cutoff `000..N` independently to every requested
arm, then build the exact representation-identity intersection. Print the
requested cutoff, each arm's available event count, common-panel size and hash,
side-only identities, and the standard scientist table.

Example: `R-run leaderboard up to round 19` evaluates the first 20 committed
events of every arm that has reached round 19. An arm that has not reached the
cutoff is marked unavailable and is not silently truncated to a smaller N.

### `<run> leaderboard up to round N, <arm selectors>`

Restrict the report to the selected arm. For example,
`R-run leaderboard up to round 19, adaptive, sharing` means the `AS` arm and
events `000..019`. Because all scientists inside one coordinated arm share the
same tasks, the table uses all 20 tasks without a cross-arm intersection.

Accepted R-run selectors are `static`, `adaptive`, `random`, `sharing`, and
`no sharing`; combinations must resolve to exactly one arm. If they do not,
print the candidate arms and request clarification.

### `<run> arms leaderboard`

Rank arm-level portfolios rather than individual scientists. Required columns:

| arm | paired n | portfolio coverage | L10 portfolio capped mean | L1000 portfolio capped mean | innovation | total NN evals | wall time | retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

For R-run, prefer the registered pairwise contrasts:

1. `SNS` versus `SS`: sharing effect under static order;
2. `ANS` versus `AS`: sharing effect under adaptive selection;
3. `SNS` versus `SRNS`: fixed ACS order versus fixed random order.

Static-versus-adaptive contrasts use exact task intersections and are labelled
selection-policy comparisons. A five-arm global table is allowed only on the
five-way identity intersection and must state its size.

`<run> arms leaderboard up to round N` combines this definition with the
inclusive cutoff rule.

### `<run> innovation leaderboard`

Use only native post-learning evaluations committed before sharing. For
scientist `i`, objective `A`, and task `x`, let `C_A(i,x)` be its capped native
cost and `C_A^(2)(x)` the best cost outside the winning set. Tied winners split
credit and margin. Report the additive score

`I_A(i) = sum_x (C_A^(2)(x) - C_A(i,x))/U_A + sole_solves + record_gain`.

Print L10 and L1000 separately, the paired-panel size, sole solves, tied wins,
normalized quality margin, and record gain. `record_gain` is zero unless a
provenance-bearing incumbent bank was supplied. Scores from panels of different
sizes are not compared directly.

### `<run> top innovators`

Print a compact, evidence-bearing view of the scientists responsible for the
largest native improvements, rather than the full leaderboard. The default is
the top five scientists by L1000 innovation on the exact common completed task
panel. Also print a compact top-three L10 list because L10 and L1000 innovation
can select different architectures. If the request names an objective, such as
`R-run top innovators for L10`, print only that objective.

Required columns are:

| rank | scientist instance | arm/curriculum | paired n | innovation | sole solves | tied wins | normalized margin | coverage | capped mean | native NN evals | retention |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

After the table, print each ranked scientist's strongest contributing events:
representation identity, its native cost, the best competing native cost,
normalized margin, and whether the contribution is a sole solve, quality win,
tie, or provenance-backed record gain. Long contribution lists are truncated
in prose but preserved as a machine-readable list when the report is saved.

The command obeys the same native-evidence and common-panel rules as
`<run> innovation leaderboard`:

- only post-learning native evaluations committed before translation or
  donation are eligible;
- the snapshot time, cutoff, panel size, and representation-identity hash are
  mandatory;
- scientists from different arms or curricula are ranked together only on the
  exact task intersection;
- if the global intersection is too small, print the registered pairwise or
  within-arm top innovators instead of forcing a global rank;
- scores on fewer than ten representations are labelled `exploratory`, and
  scores on fewer than twenty are labelled `provisional`;
- high innovation does not imply high average quality, so coverage, capped
  mean, retention, and NN evaluations remain visible; and
- `record_gain` remains zero unless the run has a versioned incumbent bank and
  a verified witness that strictly improves it.

Accepted forms include:

- `<run> top innovators` — top five L1000 and compact top three L10;
- `<run> top N innovators` — change the displayed count;
- `<run> top innovators up to round N` — apply the inclusive cutoff before
  constructing the common panel;
- `<run> top innovators, <arm selectors>` — restrict to one arm or registered
  contrast; and
- `<run> top innovators for L10` or `for L1000` — restrict the objective.

For example, `R-run top 3 innovators up to round 39, adaptive` compares the ANS
and AS scientist instances using only committed events `000..039` and their
exact shared representation identities. `Q-run Q20 top innovators` keeps skips
and unsolved representations in the denominator and reports whether an
innovation occurred on a 6+ strand representation.

### `<run> efficiency leaderboard`

Use the standard paired panel and report coverage and capped quality alongside
native NN evaluations, evaluation NN evaluations, wall time, solved
representations per million NN evaluations, and quality improvement over the
failure cap per million NN evaluations. Do not rank by speed alone when coverage
or retention regresses.

### `<run> retention leaderboard`

Report the last common paired retention checkpoint: retained solves/attempts,
solve rate, capped mean, configured target, rehearsal dose, regression from the
pre-rehearsal state, and pass/fail. Retention is never inferred from native
solve rate.

## Run-specific requests

### Q-run

- `Q-run <group> status`, for example `Q-run Q20 status`: operational and gate
  status for exactly one group.
- `Q-run <group> leaderboard`: scientist results within that group, with skips
  and unsolved rows retained in denominators.
- `Q-run gate`: exact pass/fail/pending evidence for advancing from the active
  group, including all-scientist completion, retention, capacity exceptions,
  and native success on a 6+ strand representation.
- `Q-run curriculum status`: completed and remaining groups, representation
  totals, strand-quota audit, skip budgets, and estimated next transition.
- `Q-run strand leaderboard`: results stratified by `1..5`, `6..8`, `9..11`,
  and `12` strands; print the denominator in every stratum.

### R-L1000-run

- `R-L1000-run curricula leaderboard`: compare the three curricula on the exact
  common task panel, both pooled across architectures and separately per
  architecture.
- `R-L1000-run architectures leaderboard`: compare the four architectures
  within each curriculum; do not pool curricula into an undocumented score.
- Both forms accept `up to round N` and an optional architecture or curriculum
  selector.

The primary response variable is L1000 coverage and capped mean. Also report
innovation, NN evaluations, retention, and wall time so additional mixed
training compute is visible.

### DKT72-run

- `DKT72-run leaderboard`: the frozen supported-panel scientist and portfolio
  table, with capacity-unsupported rows reported separately.
- `DKT72-run upper-bound status`: supported open knots, incumbent intervals,
  best verified crossing-change counts, strict improvements, and whether any
  result would certify an exact value.
- `DKT72-run witnesses`: witness identity, scientist, stochastic seed,
  crossing changes, semantic moves, replay verification, and stored artifact
  path/hash. Do not print long action sequences unless explicitly requested.

### Completed bounded runs

`R-Invariant-run leaderboard`, `R-Mutation-run leaderboard`, and
`P-Invariant-run status` are immutable final reports. They always print the
artifact/report SHA-256 and label the result completed. `R-Depth-run` remains a
live suffix experiment until all 11 events and its final report are committed.

### Recovery-sync

- `Recovery-sync status`: age and integrity of the local mirror, latest Drive
  delta receipt, latest weekly full capsule, free local space, pending uploads,
  and any filename/size conflicts.
- It never reports experiment quality and never deletes local, Drive, or remote
  artifacts.

## Examples

- `Runs status`
- `R-run status`
- `R-run resources`
- `R-run ETA`
- `R-run leaderboard`
- `R-run leaderboard up to round 19`
- `R-run leaderboard up to round 19, adaptive, sharing`
- `R-run arms leaderboard up to round 39`
- `R-run innovation leaderboard up to round 39`
- `R-run top innovators`
- `R-run top 3 innovators up to round 39, adaptive`
- `R-run top innovators for L10`
- `R-L1000-run curricula leaderboard up to round 19`
- `R-L1000-run architectures leaderboard, L1000-only`
- `Q-run Q20 status`
- `Q-run Q20 leaderboard`
- `Q-run gate`
- `DKT72-run upper-bound status`
- `Recovery-sync status`
