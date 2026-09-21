# Clean-start multisize distance/policy comparison

This experiment follows the audit correction in note 23. Its purpose is to
measure whether ordinary CC-distance supervision improves the policy/search of
the scalable strand-graph architecture. No historical checkpoint is a parent.
It is an intermediate experiment toward joint RL on long words and many strands.

## Frozen comparison, before training or new panel scoring

Two arms use the same initialization seed 20261006, replay data, policy batches,
optimizer, architecture and final-step selection:

- `embedding-multisize-clean-joint-v1`: metric weight 1.
- `embedding-multisize-clean-policy-v1`: metric weight 0.

Architecture: width 48, three necklace blocks, embedding dimension 64,
two strand-graph rounds, automatic multiscale context, window size 8 with four
shared-weight window levels, normalized global knot view. No generator-ID
vocabulary or maximum-length output layer is used. The policy context includes
all five resource features (remaining moves, CC, length slack, strand slack and
CC cost weight). This run uses the existing
site action head. The factorized proposal head has no trained role in these
arms; evaluation uses the same lazy catalogue and progressive widening.

Training: 4096 updates, metric batch 64, eight policy samples per update,
AdamW learning rate 0.0003, one CPU thread per arm, uniform witness weighting,
balanced root-u sampling. All parameters are initialized from scratch. The
policy loss uses 64 evenly spaced lazy-catalogue candidates plus every
demonstrated positive, materializing no successor words. It is a conditional
shortlist loss, not an unbiased estimator of the full-action softmax. Both arms
use identical shortlists. This removes the O(L*N) policy-data preparation cost.
The metric arm also trains the explicit mirror-trefoil distance-two sentinel;
the policy control processes the same batch with zero metric coefficient.
There is no checkpoint selection by validation loss and no early stopping.

Each arm presents 262144 metric examples and 32768 policy samples, sampled
with replacement. These are not counts of distinct knots or fresh self-play
episodes. No value outcome targets are invented: this experiment trains
distance/policy, with constant leaf value at search evaluation.

## Data

Source: `embedding-short-u-augmented-corpus-v2/corpus.json` (4096 train records).
The builder `build_multisize_replay_corpus.py` makes a new immutable corpus at
`embedding-multisize-clean-corpus-v1`. Every third train record is unchanged;
the remaining thirds receive exactly replayed zero-CC lifts to at least (L=60,
N=9) and (L=128,N=17). All pair-prefix lengths are extended consistently and
every record is reverified, including signed-signature bounds and train-path
ownership. These are correlated additional representations. The lifts include
easy inverse moves; they do not stand in for intrinsically difficult long knots.

Validation views remain as in the source. In particular the field `size_shift`
is now a historical view name and need not exceed the new training dimensions.
The training hashes, parameter configuration and source snapshots are recorded
in each run manifest. There are no warm-start ancestors to audit.

Separate evaluation-only generation: `embedding-multisize-clean-test-v1`, seed
20261007, target 32 records, at most eight per fingerprint group, maximum 40000
generation attempts, partition `test`. Selection uses no model scores.
Do not reduce the target, change the seed, or relax the cap after inspecting
outcomes. If generation is partial, retain it as partial and do not report the
predeclared panel as completed. Its records never enter training or checkpoint
selection. Test groups may have appeared in earlier research cohorts; report
training-disjoint evidence and disclose this development-history limitation.

## Evaluation

Require final-step checkpoint hashes and matching initial states/configuration
except metric weight/output path. Compare ordinary signed CC interval errors,
exact-u groups, constants 0/1/2, and the untrained model. Metric improvement must
be checked alongside actual returned witness costs.

Use all completed test representations for an anytime cost search, 128
expansions with identical witness-feasible path limits, separately for L10 and
L1000. Compare learned embedding against the same untrained architecture and
length heuristic; report neural evaluations and elapsed time as well as
expansions. This is equal expansion count, not equal neural compute.

Also use one model-blind shortest witness per fingerprint group, four seeds,
512 sparse MCTS simulations, widening batch four, constant leaf value 0.5,
root Dirichlet fraction 0.25, separately at CC weights 10 and 1000. Compare both
trained arms and untrained. Replay every returned witness; failures are
censored and first-found costs do not imply optimality. Report new/lost solves
and paired costs on shared solves, not only averages over successful cases.

The full objective remains useful short distances and smaller L10/L1000 on
variable sizes. This comparison alone is not a million-node RL result or proof
of reliable extrapolation to arbitrary sizes. It determines the next rollout
collection and larger training step using the scalable architecture.

## Construction and launch evidence

The multisize corpus completed: 4096 reverified training records, lengths
6..128 and strand counts 2..17. Its SHA-256 is
`04ffeca1bebccc5e8af6a7f7553b2faa2895e098041320e73bf4a2cd8093600e`;
source corpus SHA is
`4a7def7fe814725eabbef42812c77c4a07c1691c28f35137d2db8bf2b1bc6bec`.

Test generation exhausted the fixed 40000 attempts with 31/32 records in six
fingerprint groups. The panel remains explicitly partial and is not scored.
Intervals are 18 [0,2], nine [2,2], three [0,1], one [1,2]. No learned scores
were consulted. This is an unresolved independent-evaluation requirement;
development results cannot replace it or complete the overall objective.

Both training runners were launched after these data checks and 20 focused
tests passed, including an assertion that sparse witness candidates retain
every demonstrated positive without enumerating successor words. Initial
processes: joint PID 42593, policy PID 42601. These are historical launch IDs,
not a later liveness claim. Each runner records its own lock, source snapshots,
initial tensors and training checkpoints. Check fresh process CPU and manifest
advancement when continuing; never start a duplicate based on an old PID alone.

### Verified active snapshot, 2026-09-20

Both runners advanced through saved steps 16, 32, 48 and subsequently 176/4096.
CPU time independently advanced beyond six minutes in both processes. All
model/objective tensors in the step-176 snapshots were finite; optimizer and
all four RNG states were present. Recorded training source hashes still
matched the files on disk. These are active runs, not completed results.

Initial model and objective tensors were checked pairwise for exact equality.
Both initial checkpoint files have SHA-256
`e0e0b4acf6518c64e60c2bbd8a57a9040bda13662ff5bd431f52f5e60d12737b`.
The only configuration differences are output directory and metric weight;
the recorded source dictionaries also match. Each arm prepared 244412 policy
samples along correlated witness paths, not 244412 distinct knots.

Step-176 checkpoint byte snapshots (files continue changing):

- Joint: `f52b85ad8894e00b8d48888bad6221a81248bbe14992a86f10cbbaf88f4d99c8`.
- Policy: `f5ffdc5d3444053a96e9aa6d140ec5b9463511c0f5a3896b7703cc8af8c9cae9`.

The panel evaluator now loads SHA-pinned saved initial tensors when recorded,
and labels legacy seed-only reconstruction explicitly. Five focused evaluator
and sparse-policy tests passed. No running trainer dependency was modified.
Next: finish both arms, verify final hashes, and resolve the incomplete test
panel through a separately declared evaluation protocol before claiming
independent distance or L10/L1000 improvement. Do not silently score the 31/32
partial panel or choose a checkpoint from development results.

### Independent-evaluation preparation continuation

A separate prospective fixed-effort protocol is recorded in
[note 28](28-fresh-group-fixed-budget-test.md); it does not modify the partial
31/32 panel. Its builder is active in `embedding-fresh-group-test-v1`, historical
launch PID 46188. At 7000/40000 attempts it had one accepted representation in
one new fingerprint group. This does not yet satisfy its eight-group/sixteen-
representation adequacy gate. The complete exclusion inventory on disk has five
prior panels. Construction source hashes and exclusion file hashes were checked
unchanged while it ran.

At the same snapshot the joint/control trainers advanced to 1216/1184 of 4096
steps, respectively; both process CPU times advanced beyond eleven minutes,
and all recorded trainer source hashes still matched. No held-out model scores
were used for these decisions.

Saved initial-state loading now also applies to the anytime-search and sparse
MCTS evaluators via `embedding_checkpoint_io.py`, including SHA and architecture
validation. Evaluator-only changes do not affect the running training sources.
Fourteen focused cost-search, evaluation and fresh-panel tests passed, followed
by four initial-state/bounds tests after the single-read checkpoint hardening.

### General short-pair cohort constructed

[Note29](29-general-short-gordian-test.md) prospectively adds distance evaluation
between nontrivial knots. Its fixed4000-attempt construction completed at
`embedding-general-gordian-test-v1/panel.json`, SHA-256
`27ccaa601f3ff5cc63028b33b1f2cdefe40142a055b96b0bcee2d3a148341c3d`.
There are84 original pairs in18 query fingerprint groups:29 exact-distance0,
20 exact-distance1,9 exact-distance2 and26 intervals, plus84 replayed
zero-CC left-endpoint lifts to L160,N25. The predeclared adequacy gate passed;
no network has scored this cohort yet. This is not a short-u/search replacement.

The separate evaluation validator allows a test reference endpoint (necessary
for same-type zero-distance pairs) without changing training ownership rules.
Tests cover exact0/1/2 prefixes, lifted prefixes, corrupt bounds/fingerprints and
continued rejection by the training validator. Eighteen focused evaluator and
cost-search tests passed; the subsequent thirteen-test fresh-panel/sparse-policy
regression also passed. The metric evaluator now retains pair indices and
fingerprints and reports per-group errors and constants, not only row means.

Full lineage audit is in progress, historical PID49160 / exec session8703.
It already replayed all168 test records, cross-checked18 group determinants
using the independent implementation, and advanced through1024/4096 training
records with intermediate path fingerprints. Do not report the complete
independence audit as passed until its immutable output exists:
`embedding-general-gordian-test-v1/independence-audit.json`.

Both trainers remain live. At the earlier simultaneous snapshot they had
2304/2240 steps; the separate short-u generator had28500/40000 attempts,
only four accepted representations in three groups and was still inadequate.
Neither that partial generation nor the general metric cohort establishes
improved L10/L1000. Continue the existing sessions, do not duplicate them.

### Full audit passed; final evaluation waiting on training

The general-pair audit completed and passed: all4096 training records and168
test-view records replayed,123 training-path groups,18 test query groups,
empty query/train-path and query/prior-panel intersections. Eighteen independent
determinant checks agree. Audit SHA-256:
`004a6d9a90e587b170d33af9e45dca09af4854920c60165dada3087230e825f6`.
This is query-side independence for the audited lineage and listed panels;
known reference endpoints and incomplete legacy-exposure coverage remain explicit.

The fresh short-u generator exhausted40000 attempts with four representations
in three groups. Construction completed but the adequacy gate failed;
`panel.complete` remains false and this cohort is not scored. Its quota, seed,
exclusions and acceptance rules were not changed. Independent short-u/search
evidence is still missing; the general-pair metric cohort does not substitute
for it.

A single gated evaluation runner is now waiting at
`embedding-multisize-general-evaluation-v1` (historical PID51342, exec54600).
It checks live intended training command lines, then requires both completed
step4096 checkpoints, identical initialization/configurations apart from metric
weight, matching source/data hashes and the completed audit before scoring
joint, policy and initial controls. No intermediate checkpoint is scored.
Its own source/input hashes are frozen; do not edit its dependencies while it
waits or evaluates. On a gate failure inspect the error; do not bypass it or
launch a duplicate. The last verified training steps were3088 and3008.
