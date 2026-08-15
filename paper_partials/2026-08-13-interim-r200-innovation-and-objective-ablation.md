# Interim R200 evidence, innovation, and the L1000 objective ablation

Date: 2026-08-13. Status: **preliminary snapshot, not a final R200 result**.
Snapshot time: 2026-08-13 22:28:54 UTC.

This partial preserves the first paper-relevant evidence from the concurrent
Semantic-v2 R200 arms and invariant-oracle experiments. The active jobs were
not stopped to make the snapshot. All comparisons below therefore use frozen
common prefixes or exact representation intersections and must not be read as
final treatment effects.

## Experimental objects and provenance

The four R200 arms run from `pgx-mcts-bench` commit
`4be0bac60c7dec35f995f9a4912cedf9907a12f2`, with executable-source SHA-256
`7169d55a938bf5316f2dc1d07998088071f3efc38da87040d23bfe0b7599be2e`.
They use the same R200 bank, seed 20262120, three scientists, L10/L1000,
adaptive compute and rehearsal, EV4, and eight self-play games per native
iteration. Static versus adaptive order and sharing versus no sharing are the
two treatment dimensions.

At the snapshot the completed-event counts were:

| arm | completed R200 rounds | manifest SHA-256 | event-tree SHA-256 |
|---|---:|---|---|
| static, no sharing | 50/200 | `a175738c63780ddf550db41b7d961d237ad2d193b6fbb3c1794ac3b853814ed8` | `aec5286bb86056fed69cf672ffcfc6f75f931c30df458c0598a6e6e76631861a` |
| adaptive, no sharing | 31/200 | `3b282deecea1be161554dbbcdfa2f8b3bc60831d936bd6596b43c322b87a7337` | `398a7f35d8b227f333bfa334488f115f6178886ae1522fffb1b4e23c0f6ccdae` |
| static, sharing | 29/200 | `5542b89c622782af9e4f976c20f93ad3bdfc437e349dd622099d850da43c1a3b` | `ae905b00e21d0c312130171f0fd24a78bd55468d6f340028f734789d91db62a8` |
| adaptive, sharing | 49/200 | `3d26407b4b822da7919eb19a55ab8f037ff6b2521b0372c1a1fad8ada1dc075d` | `260c5ff3475dc370ca030f44bae079df3cada105783480ce550bb1518c8fc91c` |

The tree hash is the SHA-256 of the ordered event filenames and their
individual SHA-256 digests. It identifies this evolving snapshot; it is not a
final-run identifier.

## Metrics

For objective ratio `A`, a failed representation has capped cost

`U_A = 20*A + 128`,

therefore `U_10=328` and `U_1000=20128`. `Portfolio average` is the average,
over the paired representation panel, of the lowest capped cost found by any
of the three scientists in an arm. `Scientist average` first averages capped
cost over representations for each scientist and then averages across the
three scientists.

Innovation is measured only from the native evaluation after task-local
learning and before translation or donation. For scientist `i`, objective `A`,
and task `x`, let `C_A(i,x)` be its best native cost and let `C_A^(2)(x)` be the
best cost outside the winning set. Tied winners split credit and margin. The
current score is

`I_A(i) = sum_x (C_A^(2)(x)-C_A(i,x))/U_A + sole_solves + record_gain`.

The record term is unavailable until a provenance-bearing incumbent bank is
supplied, so it is zero in the tables below. Scores are additive over tasks and
may be compared within a paired panel, not across panels of different size.
Each sharing/no-sharing result below is computed as one pooled six-scientist
contest on the common representations; arm labels keep the two copies of an
architecture distinct.
The reproducible implementation is
`pgx-mcts-bench/scripts/rank_sv2_innovation.py` at commit `f557eee`.

## Sharing versus no sharing: first paired evidence

The static arms have the same first 25 representations. The adaptive arms
selected different tasks; their first-25 comparison is therefore restricted to
the 18 representations selected by both arms. Within every row below, the
portfolio solved sets are exactly equal: there are no sharing-only or
no-sharing-only representation IDs.

| schedule/panel | objective | innovation, no sharing | innovation, sharing | portfolio avg, no sharing | portfolio avg, sharing | scientist avg, no sharing | scientist avg, sharing |
|---|---|---:|---:|---:|---:|---:|---:|
| static, 25 common | L10 | 0.5091 | **0.6006** | 86.20 | 86.20 | **110.35** | 134.21 |
| static, 25 common | L1000 | 0.2622 | **0.4711** | 6033.56 | **5714.52** | **7878.60** | 8491.07 |
| adaptive, 18 common | L10 | **0.9964** | 0.5310 | **48.06** | 56.22 | 95.00 | **94.63** |
| adaptive, 18 common | L1000 | **0.3890** | 0.2005 | **3638.00** | 4301.22 | **5192.61** | 5843.85 |

This does **not** support the unconditional claim that independent scientists
are better innovators but worse on average. In the adaptive comparison,
no-sharing is both more innovative and better as an L10/L1000 portfolio. In the
static comparison, sharing is currently more innovative and improves the L1000
portfolio, although the average individual sharing scientist is worse. A
reasonable paper hypothesis is narrower: adaptive sharing may homogenize
policies, while static sharing may consolidate complementary solutions. R200
must progress much further before this is presented as a result.

## Human-invariant scientists: 23-task interim result

The five invariant scientists used identical static/no-sharing R24 tasks,
SIM64, F10, adaptive rehearsal, EV4, and mixed L10/L1000 training. Their source
was hash-certified but uncommitted, based on `4be0bac`; executable-source
SHA-256 was
`4d4cd886c8a4ce38f6b994bd71001c971a95fd4d20770a4407797b98fcff8524`.
The snapshot contains 23 completed events, event-tree SHA-256
`ba99cd00cd246b1ea58ed93ac4e58c231cc8c2b0d115da30ad491ac674d852c5`.

| objective | scientist | innovation score | coverage | capped average | native network evaluations |
|---|---|---:|---:|---:|---:|
| L10 | Jones-only | **2.4090** | **23/23** | **24.83** | 3,426,410 |
| L10 | combined FiLM | 0.5742 | 22/23 | 37.48 | **3,133,520** |
| L10 | combined dual | 0.0564 | 21/23 | 51.65 | 4,216,225 |
| L1000 | classical | **0.3521** | 22/23 | 2839.87 | 5,676,580 |
| L1000 | combined FiLM | 0.1542 | 22/23 | 2359.74 | **3,133,520** |
| L1000 | combined dual | 0.0878 | **23/23** | **2051.17** | 4,216,225 |

Alexander-only is omitted from this compact table but retained in the source
data; it covered 16/23 at L10 and 19/23 at L1000. The scientifically important
distinction is that combined dual currently gives the strongest L1000 coverage
and capped average, while classical has the highest L1000 innovation score.
Portfolio quality and innovation are therefore different response variables.

The invariant vector is cached over invariant-preserving operations and
recomputed after every crossing change. This supplies information about the
new knot type inside MCTS rather than only describing the initial knot.

## Capacity mutations: preserve, but do not promote yet

The first architecture-mutation suffix completed 11 tasks. A second paired
depth-dose run had completed 9/11 at this snapshot. Its event-tree SHA-256 was
`8d2fcaabca605adc72031cb6c46cffbd7d2ea56ad2ef463b3eff103c8e3cf13f`.
All candidates share the same frozen 13-representation rehearsal prefix and
the same suffix identities.

On the nine-task common panel, `combined-dual + 1 residual invariant block`
had the highest L10 innovation score (0.00531), solved 9/9, and had capped
average 27.22. The unchanged combined FiLM parent had the best observed L1000
capped average among the listed parents/mutations (2008.44, 9/9). These margins
are tiny and the panel is too small for promotion. The important preserved idea
is methodological: deeper networks may start worse on easy tasks and catch up
later, so capacity mutations should be evaluated on a long paired suffix, not
selected from reception-class solve rate.

## Launched causal ablation: does mixed-objective training help L1000?

The next clean experiment should use four promising and structurally diverse
architectures:

- `raster-invariant-combined-dual`;
- `raster-invariant-combined-film`;
- `strand-graph`; and
- `raster-axial`.

Each architecture is cloned from one identical pretrained checkpoint into
three independent, paired curricula:

| curriculum | self-play games per iteration | purpose |
|---|---|---|
| L1000-only | 8 L1000 | direct target-objective learner |
| mixed, fixed total compute | 4 L10 + 4 L1000 | tests transfer under the same total game budget |
| mixed, matched L1000 exposure | 8 L10 + 8 L1000 | separates transfer/interference from reduced L1000 exposure |

All 12 scientists use static ACS order, no sharing, identical
representation-keyed seeds, R24 followed by R200, and L1000-only evaluation.
R24 fixes `F_native=10` and SIM64 while retaining adaptive rehearsal. At the
group boundary, the exported controller is explicitly reset to `F_native=5`
and SIM64; R200 then enables adaptive native compute and rehearsal. The primary
outcomes are L1000 portfolio
capped loss, individual capped loss, exact solved-set differences, innovation,
network evaluations, and wall time.

The experiment launched at 2026-08-13 22:55 UTC as the low-priority systemd
unit `l1000-objective-ablation.service`. Its immutable artifact root is
`/srv/braid/artifacts/l1000-objective-ablation-20260813`. It contains the four
frozen starting checkpoints, selection and experiment manifests, the launcher,
and the exact source patch applied to base commit
`f557eee3a8f31568f8ea0bffeb1ce8bda202045d`. The three R24 manifests declare
training allocations `8 x L1000`, `4 x L10 + 4 x L1000`, and
`8 x L10 + 8 x L1000`, respectively, while all declare evaluation ratios
`[1000]`. The wrapper enforces a complete R24 phase barrier before exporting
states and starting R200. Initial verification found all three coordinators and
all 12 scientist workers alive, with 89 GiB memory available and no error
trace. These launch facts are protocol evidence, not outcome evidence.

Interpretation is predeclared:

- mixed 4+4 beating L1000-only is evidence that L10 transfer helps despite
  halving direct L1000 experience;
- L1000-only beating mixed 4+4 but tying mixed 8+8 indicates allocation, not
  negative interference;
- both mixed variants losing to L1000-only indicates genuine distraction;
- mixed 8+8 winning only with much larger compute is useful transfer but not
  necessarily better compute efficiency.

## Operational compute note

At 22:25 UTC the 32-vCPU host had load average 8.38, used 33 GiB of 125 GiB,
and had 92 GiB available. The four arms are concurrent but their persistent
scientist workers are mostly single-core search/training processes; substantial
CPU capacity is therefore idle. Recent per-round throughput gave a linear
completion estimate of roughly 11--12 additional wall days, with a safer
12--16-day range because rehearsal cost grows with the history bank. The
corresponding planning estimate was 45--60 CPU-core-days. These are operational
estimates, not final resource measurements; the paper should report actual
systemd CPU accounting after R200 completes.

## Durable native-evidence boundary for future runs

Commit `f557eee3a8f31568f8ea0bffeb1ce8bda202045d` introduces a transaction
boundary required for paper-grade innovation claims. Every future round:

1. completes native learning and evaluation for all scientists;
2. atomically writes `native-events/NNN.json`;
3. `fsync`s the file and directory and verifies its SHA-256;
4. only then allows translation;
5. verifies the native hash again before block-level distillation; and
6. writes a completed event referencing the immutable native commit.

Existing R200 workers were not restarted and do not use this durability layer.
Their completed event payloads contain native pre-sharing evaluations, but
future experiments provide the stronger crash-auditable guarantee.

## Preserved artifacts

A compact local snapshot exists at
`/Users/artemvorozhtsov/Documents/rf-knots-experiment-backups/nebius-20260813`
and as the 405 MiB tar archive
`/Users/artemvorozhtsov/Documents/rf-knots-experiment-backups/nebius-20260813.tar`.
Archive SHA-256:
`3a2588fdeafcd27d582bc86c92ab49355597e2d66d7df351ab7bae8a26b294e5`.
A copy was also uploaded to Google Drive. The snapshot deliberately keeps
current/final states, selected checkpoints, manifests, events, reports, banks,
and executable source while excluding redundant model histories and Python
environments. Its internal `SHA256SUMS` verifies every preserved file.

The machine-readable statistics accompanying this partial are in
[`data/2026-08-13-interim-r200-innovation.json`](data/2026-08-13-interim-r200-innovation.json).

The exact lightweight manifests, reports, and event JSON used for this partial
are separately frozen in
`/Users/artemvorozhtsov/Documents/rf-knots-experiment-backups/paper-evidence-20260813T222854Z.tar`
(26 MiB), SHA-256
`1bdbc2bde20bb6873430f27a7c7efba7fbc6c8a456512e02a1c59b4b07a4856d`.
This second archive is the exact evidence behind the 22:28:54 UTC tables and
does not contain model binaries.
