# Raster-axial capacity-family preregistration

On 2026-08-11 three new scientists were defined from the admitted local
`raster-axial` architecture. The purpose is to use otherwise idle CPU while
testing three distinct limitations of the 64-channel, four-block baseline.

| Scientist | Width | Blocks | Residual | Persistent memory | Hypothesis |
|---|---:|---:|---|---|---|
| `raster-axial-v2` | 96 | 6 | standard masked axial | none | moderate capacity is sufficient |
| `raster-axial-v3` | 96 | 8 | masked axial + LayerScale | none | stable depth improves geometric reasoning |
| `raster-axial-v4` | 96 | 8 | masked axial + LayerScale | writable 8-symbol tape | scanning needs persistent global memory |

The existing raster already has mask-aware GroupNorm in every residual block and
an output normalization. LayerScale is not an extra generic normalization layer:
it is a learned per-channel residual multiplier initialized to 0.1, keeping the
eight-block network close to an identity refinement at initialization.

Foundation pretraining starts independently from seeds 71, 72, and 73. It uses
the frozen six-stage protocol: equal L10/L1000 sampling, adaptive
`F_native=5,8,12,16`, simulations `64,128,256,512`, adaptive rehearsal from 1 to
8, acquisition floor 70%, promotion/retention target 80%, balanced replay, and
success-only policy/cost training. The deterministic selection rule is the
lowest-numbered fully promoted seed per architecture; if none fully promotes,
the architecture fails admission rather than receiving a favorable hand-picked
checkpoint.

Here “fully promoted” means that the run contains an explicit objective-quality
promotion at every foundation stage. Merely reaching the final stage index is
insufficient: the adaptive runner may advance past an unpromoted stage after
exhausting its declared dose. The selection script checks the complete set of
per-stage promotion records, not only `highest_stage`.

The selected v2/v3/v4 checkpoints will run together as a three-scientist
static-order, no-sharing R24 arm with the same 64 simulations, `F_native=10`, four
evaluation attempts, and adaptive rehearsal as the completed control. Individual
scientist metrics remain primary so the three-model portfolio cannot hide a weak
variant. R200 begins only after R24 and uses the registered static ACS order and
adaptive compute. Reports must include solve attempts, portfolio coverage, L10
and L1000 sums, crossing changes, semantic moves, network evaluations, wall time,
and retention.

The Nebius pretraining job runs at Unix nice level 15, with nine single-threaded
workers restricted to CPUs 0-29. It uses a separate source checkout so active
R24/R200 workers cannot import changed modules when spawning children. The
machine has 125 GiB RAM, of which 109 GiB was available before launch.

The first Linux smoke found that `ProcessPoolExecutor` inherited the platform
`fork` default after JAX had initialized threads. All workers waited on the same
futex before their first MCTS step. Foundation workers now use an explicit
`spawn` context; a repeated three-architecture MCTS/train/checkpoint smoke passed.
The complete local suite then passed 313 tests.

The full job started at `2026-08-11T21:44:12Z` as the transient systemd unit
`raster-axial-capacity-pretrain.service`. Its artifact root is
`/srv/braid/artifacts/raster-axial-capacity-pretrain-20260811`. Nine spawned
workers were active, and the first post-launch CPU sample was 93.8% busy with
6.2% idle. Selection, per-scientist state export, R24, and R200 have dedicated
launchers. R24 is not started until all three architectures have a fully promoted
seed; R200 is not started until the R24 retention and objective report is
inspected.

At `2026-08-11T22:23:48Z`, two separated CPU samples remained near 62% busy
despite all nine primary workers running at one full core each. A supplementary
robustness run was therefore launched for seeds 74 and 75 across all three
architectures. It is explicitly excluded from the primary selection rule. The
transient unit `raster-axial-capacity-supplementary.service` runs six workers at
nice 19 and `CPUWeight=10`; its artifact root is
`/srv/braid/artifacts/raster-axial-capacity-supplementary-seeds74-75-20260811`.
The immediate post-launch sample rose to 81.8% busy, all primary processes
remained alive, and 96 GiB RAM remained available.

At `2026-08-12T02:23Z`, primary `raster-axial-v4` seed 71 completed the six-stage
foundation curriculum with an explicit objective-quality promotion at every
stage. Its final stage required adaptive escalation to `F_native=16` at 128
simulations and then promoted at 100% solve rate. This makes v4 the first member
of the preregistered primary capacity family with a selectable checkpoint.

Supplementary `raster-axial-v3` seed 75 independently completed all six stages,
using the maximum 512-simulation search level on its final promotion. This is
evidence that the architecture can learn the curriculum, but it remains excluded
from primary checkpoint selection because seed 75 was declared supplementary.

At `2026-08-12T04:23Z`, primary `raster-axial-v2` seed 72 recovered from its
initial zero-solve attempts and promoted stages 0, 1, and 2 consecutively. The
stage-2 promotion required `F_native=16` and the maximum 512 simulations. V2 is
therefore still eligible, but its foundation compute is already materially above
v4 seed 71, whose final promotion used 128 simulations.

At `2026-08-12T05:23Z`, the preregistered three-scientist capacity arm became
inadmissible. Primary v3 seed 71 exhausted stage 0 at `F_native=16` and 512
simulations with zero solves and no objective promotion; seeds 72 and 73 had
already advanced without a stage-0 promotion. Thus none of primary seeds 71--73
can satisfy the strict all-stage selection rule for v3. The two successful
supplementary v3 seeds demonstrate architectural learnability but are not
substituted into the primary arm.

At the same checkpoint, primary v2 seed 72 had validly promoted stages 0--4 and
was attempting its final stage. V4 seed 71 remained fully valid. These individual
scientists may be evaluated later under a newly declared comparison, but the
original v2/v3/v4 R24 launch is closed rather than retrospectively repaired.

At `2026-08-12T06:23Z`, primary `raster-axial-v2` seed 72 completed all six
stages with an objective-quality promotion at every stage. Its final promotion
used `F_native=16` and 512 simulations, versus 128 simulations for primary v4
seed 71. The primary family therefore produced two selectable scientists, v2
seed 72 and v4 seed 71, with materially different foundation-search cost.
