# v28b: reviewed yield failure and separate connected continuation

User authorization: 2026-10-24, diagnose yield_stop, repair in a separate version,
check disk and resume full state without changing A or the counters to B.

## Closed v28a evidence

`random-TRAIN-stream-v28a/batch-000001` generated raw lengths 84,89,94,99,
reference witnesses 48,51,54,57 elementary moves, strands 9..12. This was a
deterministic size jump from batch0's raw lengths20,32,35,38 caused by coupling
index to depth/strands. No curriculum kept the task scale matched to20CPU.
All16conditions hit the CPU ceiling (321.071357 total searchCPU), mostly3..5
completed simulations;45episodes were independently replayed. Roots0 and3 had
formal successful paths, but root3 repeated its starting braid after2moves.
Its two61-move witnesses were correctly excluded as cyclic, leaving one
admissible solved view. The unchanged gate needs>=2 and<=300CPU/newview.
Reference proofs are feasible upper bounds, not MCTS labels or minimum-CC proofs.
Nothing in this namespace is retried, erased or relabeled.

## Frozen prospective correction

Separate v28b collector keeps seed20261023460 and advances to batch2/index512;
it excludes every earlier roster including failed and unsolved rows. Before
any new scores it fixes strands3..6 and constructive depth4..8 (raw<=41,
known proof<=25elementary moves). Four distinct TRAIN groups/normalized roots,
ownership/heldout/v3-panel exclusions and legacy quarantine all remain.
Search/evaluator/budgets/seeds/yield/cycle gates are unchanged. This is a shallow
synthetic curriculum, not evidence of broad knot-family coverage or utility.
Failure stops this component, with no seed retry or automatic budget escalation.

## Orchestration fixes

Old code advertised a16semantic-path refresh trigger but implemented only
>=4random normalized views. It also allowed only one4-batch MAIN intake per
round, reset the intake cursor on resume, and could report VERIFIED_ACTIVE
with supply_stopped. The separate supervisor implements both refresh triggers,
sequential4-FULL-batch intake with32pending-path backpressure, reuses audited
READY after restart, refuses blind resume of a failed dataflow, and reports
DEGRADED_DATAFLOW for a failed source. Full-state handoffs remain>=3600wall
seconds, not tiny checkpoints or live bank changes. Telemetry NEW IDs remain
relative to original A-era banks across rounds; counts are local actual native
updates and the global sampler phase is never reset.

## State and experiment preservation

Latest clean parent:1635952 SHA
f8462581baadee91f6c2714d4d2be126e30e7088d42d6c9e4f43882d09e07854.
Validated full finite model/objective/head/71AdamW and Python/Torch/MPS RNG,
all parent contract pins. Durable active since A3317.898728448199seconds.
A remains1628687 SHA
a2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c;
v3 origin206004.2251009539, conservative MAINcursor39168 unchanged. New versions
link this exact parent checkpoint/manifest so original v3 training_delta sums
the whole lineage, excluding downtime. No new panel or evaluation is authorized.

Startup requires>=16GiB free; unchanged runtime guard12GiB. Observed disk is
volatile, so this is not a guarantee of48hours uninterrupted execution. No
cleanup or deletion is authorized. Native MPS is only accessible outside the
sandbox; explicit tool-managed elevated launch is required, no fallback/service.

Acceptance is not READY or PID: require native progress beyond commissioning,
SOURCE50:50, exact draw/update counts, consumed bank hashes, actual NEW-random
labels in AdamW and no failed component. Monitor v28b, not historical v28a.
