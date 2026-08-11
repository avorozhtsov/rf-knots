# Semantic-v2 sequential curriculum preregistration

## Motivation

The semantic-v1 R24 gate measured frozen zero-shot transfer and was not a valid
proxy for curriculum learning. Under the same 64-simulation, two-attempt
protocol, archived rung-18 `s-window-128` and `s-tape4` checkpoints covered only
8/24 and 7/24 representations, despite retaining 36/36 performance on their
historical rung. Semantic-v2 therefore defines one fixed representation as one
rung and permits a declared learning dose before evaluation.

## First arm

`SV2-3S-R24-SIM64-F10-AR-EV4-NO-SHARING`

- scientists: `strand-graph` seed 71, `raster-axial` seed 71, and
  `cyclic-memory` seed 73;
- order: global static ACS order;
- objective mixture: L10 and L1000;
- simulations per move: 64;
- native dose: 10 iterations per representation;
- one native iteration: eight self-play games followed by 96 optimizer steps;
- evaluation: four attempts per representation and objective;
- rehearsal: `F_old` adapts through `1,2,4,8` at ten-rung boundaries;
- sharing: none;
- action horizon: 128 semantic-controller steps.

The 24 representations have provenance-phase sizes `6+6+3+3+3+3` and scramble
depths `0,2,1,2,3,4`. These phases do not constrain order; all 24 are globally
sorted using

`ACS = 10 * strands + 5 * exact_u + len(braid_word)`.

Here `len(braid_word)` is the number of crossings in the presented braid
diagram, not the knot type's minimal crossing number.

The exact unknotting-number distribution is `u=0: 2`, `u=1: 10`, `u=2: 9`,
and `u=3: 3`. Thus the prefix contains nine (u=2) representations rather than
being an all-(u=1) curriculum.

## Four-arm comparison

1. static order, no sharing;
2. adaptive evidence-backed order, no sharing;
3. adaptive order with strictly-better verified donations;
4. static order with strictly-better verified donations.

All arms share the same initial checkpoint per scientist, fixed group identity,
and representation/scientist-keyed random seeds. All use the fixed R24 native
compute. Adaptive native/search compute begins in the following 200-rung group.

## Donation dose

Donation dose `D` is the exact number of controlled optimizer exposures per
still-eligible donated witness and takes values `1,2,3`. A witness is eligible
only when verifier replay succeeds and its semantic objective is strictly below
the receiver's best native objective for the same task and ratio. Eligibility is
rechecked before every exposure.

Start at `D=1`. Require at least ten distinct eligible donations and two
consecutive portfolio-noninferior donation blocks before increasing the dose.
A donation-only paired regression rolls back only the donation phase, lowers
`D` immediately, and resets the healthy-block count.

## Arm 1 completed result

The static/no-sharing R24 arm completed from the three frozen post-pretraining
checkpoints. Each scientist received 1,920 native self-play attempts and 192
four-attempt evaluation cells across L10 and L1000.

| Scientist | Native self-play | Rung evaluation | Final paired retention | Final `F_old` |
|---|---:|---:|---:|---:|
| `cyclic-memory` | 1,753/1,920 | 188/192 | 48/48 | 4 |
| `raster-axial` | 1,874/1,920 | 192/192 | 48/48 | 4 |
| `strand-graph` | 1,735/1,920 | 184/192 | 46/48 | 2 |

The post-hoc no-sharing portfolio solved all 24 representations under both
objectives. Its final best-per-representation sums were L10 = 563 and
L1000 = 39,171. The selected L10 routes used 39 crossing changes and 173
semantic moves; the selected L1000 routes used 39 crossing changes and 171
semantic moves. The sum of exact published unknotting numbers in the prefix is
37. Thus the final portfolio used two crossing changes above the aggregate
known optimum in this one-attempt retention snapshot.

## Coordinated-runner mechanism smokes

On 2026-08-11, the other three arm modes were developed and smoke-tested
sequentially with the exact same architecture-specific checkpoint hashes as Arm
1. The smoke allocation was deliberately small: four simulations per move,
`F_native=1`, two self-play games, two optimizer steps, one evaluation attempt,
and a 64-action horizon. It tests mechanisms and has no scientific treatment
interpretation.

| Arm | Rungs | Native solves | Evaluation solves | Translated | Admitted | Exact donation exposures |
|---|---:|---:|---:|---:|---:|---:|
| adaptive/no-sharing | 2 | 12/12 | 12/12 | 0 | 0 | 0 |
| static/sharing | 2 | 11/12 | 11/12 | 8 | 3 | 3 |
| adaptive/sharing | 3 | 15/18 | 15/18 | 12 | 4 | 4 |

Adaptive/no-sharing produced different proposals from the three scientists and
selected the minimum verifier-backed L10 qualification result. The combined
adaptive/sharing smoke selected two unknots and then `T(2,5)+2`; its third rung
admitted four strictly better donations and applied exactly one optimizer
exposure to each. All donation guards preserved both objective-specific
portfolio coverage and capped cost. A completed adaptive/sharing state also
passed manifest-bound resume.

The implementation command is `pgx-mcts-bench braid-sv2-coordinated`. Its
manifest records the bank and checkpoint hashes, representation-keyed seeds,
qualification allocation, native/rehearsal doses, donation rule, and donation
transaction semantics. Adaptive qualification solutions remain native evidence
owned by the proposing scientist. Equal or worse translated solutions receive
zero optimizer updates.

The final smoke artifacts are under
`pgx-mcts-bench/artifacts/current/semantic-moves-v2/smoke-coordinated-v3-20260811`.
All three manifests record executable-source SHA-256 prefix `06536ea868c3` and
match the checkpoint hashes frozen by Arm 1. The adaptive/sharing smoke was
loaded again with `--resume` after completion without changing its three
committed events.

The coordinated runner now keeps one persistent process per scientist, while
the scheduler and donation transaction remain centralized. An exact
counterfactual ran the same three-rung adaptive/sharing smoke sequentially and
with process actors. The processed order, every event/controller field, and all
478 final model tensors were identical. The reference artifacts are under
`artifacts/current/semantic-moves-v2/process-equivalence-20260811`.

Full R24 runs publish compact block certificates after rungs 10, 20, and 24.
Each certificate records the selected representations; native and evaluation
solve counts and scheduled network evaluations per scientist; rehearsal dose
and post-rehearsal retention; translated/admitted donations; and the complete
donation-only guard. The CPU-32 launcher is
`scripts/run_sv2_r24_three_arms_cpu32.sh`. It assigns disjoint CPU sets to the
three remaining arms and freezes both repository commits plus SHA-256 hashes of
the bank and three starting checkpoints in `launch.env` before executing.
