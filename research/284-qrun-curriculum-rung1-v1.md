# Q-run-like policy curriculum rung 1, fixed before training

2026-10-05. The independent v3 matched DEV search finished 192/192 tasks
with 64 simulations each; all 192 shard SHA and 93 found solutions passed
independent replay. On 16 short roots, the unchanged F144 parent solved
11/16 for each L10 and L1000; short-only256 solved 14/16 and 12/16;
mixed256 solved 13/16 and 12/16. On 16 long roots, parent solved 4/16
and 6/16; short-only256 fell to 1/16 and 0/16; mixed256 solved 4/16
and 5/16. Among jointly solved short roots mixed256 was never dearer
than parent (2 cheaper L10; 4 cheaper L1000); this small one-seed panel
does not establish a winner. Short-only training has a clear long-retention
failure. Thus choose mixed256 as a pragmatic start for a new exploratory
curriculum rung, not as an established superior model. Cost sums over
different solved subsets are not compared as if paired.

Rung 1 starts from the frozen mixed256 checkpoint, not F144, and uses all
1,000 replay-verified distinct TRAIN roster roots. Keep the existing raster
policy architecture and frozen metric objective. Train only the policy for
1,024 native-MPS AdamW steps, batch four distinct roots, LR 5e-5, decay
.01, gradient clip 1, seed 20261005, checkpoints every 32 steps with
optimizer and all RNG states. Each root is chosen by deterministic
stage-wise round robin; lambda and one verified witness position are chosen
from saved RNG. The policy loss is the negative log probability of that
verified next action in the four-proposal candidate pool plus target.
Unchosen actions are not labelled bad, and no bounded search failure is a
negative. This is imitation/rehearsal, not F-dominant R2/R4 or AlphaZero.

Stages correspond to verified witness difficulty strata in the roster.
Over four consecutive 256-step phases, the per-eight-batch (32-example)
stage quotas are respectively [16,8,4,4], [8,8,8,8], [4,8,8,12],
and [4,4,8,16]. Short-root rehearsal remains in all phases, while harder
roots receive increasing weight. All 1,000 distinct roots must have been
used by step 512; audit the counters rather than infer coverage from steps.
No DEV-based checkpoint selection during this rung.

After the fixed endpoint, run a separately frozen, new root-disjoint DEV
panel with matched search budgets against mixed256 and F144, using full
replay, CC/moves, solve fraction, CPU and neural evaluations by length and
L10/L1000. The already viewed v3 DEV panel is not used to tune this rung.
TEST150 remains excluded. This rung does not start the separately planned
24-hour R2/R4 timer or establish superiority over non-neural algorithms.
