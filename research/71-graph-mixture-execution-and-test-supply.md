# Graph-mixture execution and independent test supply

2026-09-21. Follow-up to the frozen [training69](69-graph-mixture-training-protocol.md)
and [evaluation70](70-graph-mixture-prospective-evaluation.md). This is an execution
record, not a protocol amendment or a scientific promotion.

## Scope

The target remains a network without fixed word-length or strand-count lookup
ceilings, with useful ordinary short-distance embeddings and a full policy that
finds lower replayed L10/L1000 solutions. Shape compatibility, useful learned
invariants, and larger-knot solution quality are separate claims. No new residual
head is introduced. This experiment trains the encoder and policy jointly; the
unused scalar value head stays frozen.

## Actual training

The serial pair controller is running at
`artifacts/graph-mixture-training-pair-v1`. It started the graph arm and will
start the old-data control only after the graph arm completes. Both train from
the same joint57 checkpoint with the same 4,096-step contract and draw schedule.

At 06:14 UTC, graph-arm progress was independently read at 400/4,096 steps.
Repeated process samples showed advancing CPU time, and saved steps advanced
80 → 192 → 336 → 400. The manifest, progress tensor step, and contract matched;
the saved model tensors were finite. Checkpoint SHA at this *nonfinal snapshot*:
`e0357db92f294f2be7575bcb4a391ea41fe21f442e350c4bc95c43a474a4ad6e`.
The PID is not a future liveness guarantee. No final metric result exists yet.

The final-only `audit_graph_metric_training_pair.py` checks both branches as
full policy-training arms (unlike the earlier metric-only control), identical
initial weights/inputs/architecture/RNG schedules, finite optimizer tensors,
changed encoder/policy tensors, unchanged value tensors, and sample accounting.

## Ordinary graph TEST supply: complete, inadequate

The exact planned source was exhausted without model scoring:

- 1,563 source-local one-CC edges from source `validation`/`test`, with endpoint
  source splits matching each edge; no source-TRAIN rows;
- 1,635 total single-edge/two-edge paths, deterministic permutation seed20261033;
- every physical signed Artin path reconstructed; signed signatures,
  independent determinants, and any signed cyclic-coordinate equality checked;
- **zero eligible fresh TEST query pairs** after the frozen exclusions.

There are 108 unique endpoint fingerprint groups in these paths. Nine belong to
the invariant TEST partition, and **all nine have appeared as historical test
queries**. The other groups are 76 TRAIN, 22 validation, and one quarantine.
Of the TRAIN groups, 73 are in the audited optimizer lineage and three are not;
the latter still cannot be relabeled TEST. Source sidecar splits and invariant
ownership are distinct checks. All 1,635 paths received `no_fresh_query`.

Immutable artifacts:

- `artifacts/graph-mixture-fresh-gordian-test-v1/panel.json`, SHA
  `6bb53163c63ffddaa15bdb602aade0e38b646f40cd90cc4abdb75f597510d0e2`;
- `artifacts/graph-mixture-fresh-gordian-test-v1/independence-audit.json`, SHA
  `fa375ae541ffc125b5cae1add905227180490be9615984ac04b80b78f616928e`.

The audit status is `passed` for integrity, but `evaluation_ready=false` and the
panel has `complete=false`. Do not score this empty panel, top it up, reuse old
groups as independent evidence, or silently change the seed/selection rule.
General Gordian-distance promotion evidence is unavailable under this source
contract. A different source or construction requires a separately declared
prospective experiment; the frozen short-u/search comparison can still proceed.

## Fresh short-u panel and comparison preparation

The fixed 10,000-attempt constructive builder remains active at
`artifacts/graph-mixture-fresh-short-u-test-v1`. At the same snapshot it had
completed 1,900 attempts and selected 57 groups/57 roots (32 witness-CC2 and25
witness-CC1). These are partial generation counts, not a completed panel, exact
unknotting numbers, model scores, or distinct certified knot identities. No
early stopping or additional attempts are introduced.

`graph_evaluation_partition.py` reuses hash-verified full replay certificates and
collects the full current optimizer lineage: old metric paths, old rollout paths,
MCTS paths, all eligible ordinary-graph paths, audited metric suffixes, explicit
unknot and the chiral sentinel. Their union is 1,210 invariant groups. The exact
15 historical panels contain 424 query groups. No moving filesystem glob extends
or changes the frozen exclusions.

After construction, first run the full-witness `audit_joint_policy_metric_panel.py`
to `base-lineage-audit.json`, then `audit_graph_mixture_short_u_panel.py` to the
new panel's `independence-audit.json`. The extension verifies overlap with the
additional graph optimizer groups and reports raw size strata against both
metric TRAIN L254/N21 and policy TRAIN L208/N17.

`evaluate_graph_mixture_cost_mcts.py` and `audit_graph_mixture_cost_mcts.py` are
prepared for the initial, graph, control and uniform arms. For each root:

```
q0 = (raw_word_length + raw_strands - 1) / 2
move_cap = max(128, 4*q0 + 32, reference_moves)
CC_cap = reference_CC
growth_length >= 8; growth_strands >= 4; reference must fit
horizon = move_cap
```

The existing batched continuation accepts a shared horizon, so its batch horizon
is the maximum of the instance caps. Each instance terminates at its own cap
first; tests replay mixed-cap batches with each individual horizon and check that
no artificial horizon termination appears. This does not allow extra moves for
shorter roots. Lambda10/1000, seeds, simulations, proposals and accounting remain
as in70. No witness actions are supplied to the model, no value returns are used,
and unsuccessful budget-censored trajectories are not assigned negative labels.

The new tests cover source-split filtering, endpoint orientation, global group
caps, signed/unordered pair deduplication, cyclic-zero certificates, corruption
of CC steps/signatures/bounds, enlarged raw resource caps, both learned/uniform
mixed-cap replay, RNG identity, and TRAIN/historical-query collision rejection.

## Next transitions

1. Keep the existing serial training writer and constructive builder alive;
   verify CPU and durable checkpoint/file advancement, not stale PIDs alone.
2. Audit the completed short-u panel and the completed training pair. Stop the
   evaluation transition if either gate fails; do not alter their source data.
3. Score short-u distances for all three checkpoints and constants0/1/2, reporting
   exact2 separately. Run and independently replay all four cost-MCTS arms.
4. Compare coverage, common-success L10/L1000 and reference cost gaps, with actual
   work counts and explicit missing general-distance evidence. Do not infer
   unlimited-size solving or a million-game RL result from this finite experiment.

No shared model source was edited during the live run. No paid GPU was started,
no worker was interrupted, and no dirty/generated work was removed or committed.

## Continuation: evaluation transition is now active

`run_graph_mixture_evaluation.py` was actually launched, not merely prepared, at
`artifacts/graph-mixture-evaluation-v1` (controller PID84595 at the live snapshot).
It currently waits on the exact live constructive-builder command and the exact
serial training launcher/worker commands. Its manifest lists14 serial jobs:
two short-u audits, one final training-pair audit, three short-u distance reports,
and four MCTS runs each followed by an independent replay audit. It never starts
or restarts a training/data worker. The queued control is distinguished from a
missing live training launcher; duplicates, changed contracts, inadequate panels,
or missing partial evaluation workers stop the transition for inspection.

The unavailable general-distance test is explicitly recorded as unavailable in
the controller manifest; there is no metric job for its empty panel. A completed
job matrix will therefore mean the short-u/search matrix completed, not that
general Gordian distance or unlimited-size solution quality was established.

The controller has exclusive ownership of its output, pinned inputs/code plus
source snapshots, final artifact verification, and committed-prefix recovery.
Six new sequence tests cover the serial queue, dead launcher, duplicate workers,
step/config checks, live builder requirement, inadequate data, exact commands,
and the fixed14-job matrix. A read-only real gate check found both expected
workers and the queued control before launch. Repeated controller manifests
then showed attempts and optimizer checkpoints continuing to advance.

A separate score-blind [neighborhood probe72](72-ordinary-neighborhood-fresh-query-probe.md)
completed afterward. It does not change this controller or either frozen
training/evaluation protocol; its supply result is recorded separately in73.

## Fresh short-u construction completed

The original builder subsequently finished all10,000 prescribed attempts and
exited successfully. No seed extension or early stopping was used. It retained
64 original queries in64 conservative fingerprint groups, plus64 separately
labelled zero-CC lifted views. The producer counts were8,701 other-partition,
567 prior-group exclusions,272 determinant-one exclusions,64 accepted and396
over the fixed witness-CC class quotas.

Final panel SHA256:
`3432122621600c892070970d469f52a55c9ca59d0a0d42fa6f539786f37c4f84`.

Raw original query lengths are64..407 and strand counts3..33. Relative to the
metric TRAIN envelope L254/N21,38 originals are within,13 exceed length only,
6 exceed strands only and7 exceed both. The64 lifted views all exceed both
envelopes (L256..420,N33), but remain presentations of the same original knots,
not64 additional independent knot types or harder topologies.

The panel's determinant-strengthened interval counts are32 exact1,8 exact2
and24 intervals[1,2]. Witness CC=2 is therefore **not** synonymous with u=2.
These are panel supply statistics, not model scores; independent replay and
full optimizer-lineage exclusion still have to pass.

The already running core controller automatically launched the base full-replay
audit (PID3689, child of84595 at this snapshot). Its exact command and advancing
CPU time were checked. The second lineage-extension audit remains next in that
same queue; no duplicate auditor or training worker was started manually.

Both audits then completed successfully and were committed as jobs1/14 and2/14
by the controller. The base certificate SHA is
`3d39e225fb2b0566b7e2034e29245f7943faa748fd2dbe284740ad11aaef369d`;
the full optimizer-lineage extension SHA is
`b4f7943667ef6f390591e2bf5926aa04079dc103411c44a82635737069dab2c4`.
All1,033 base and1,061 extension pinned inputs were re-hashed after completion
and matched. Query intersections with the audited TRAIN and historical query
groups are empty. The extension's metric/policy size strata supersede the
base audit's older-training-envelope strata; they describe different envelopes,
not contradictory query sizes.

The queue now waits only for the two final training arms before the training-pair
audit and actual model evaluations. At the post-audit snapshot the live graph
worker had saved2,576/4,096 steps; control was still queued. Data readiness is
now established, but no new distance or search performance claim follows yet.

## Graph arm completed; serial control started

The graph arm subsequently completed all4,096 optimizer steps. The launcher
committed its final manifest SHA
`e2238acbd6406b37c24014c7fb0f6687cd4eaf872a1a8ba216d591c9ec0c25e4`
and checkpoint SHA
`dd3c8d8396da69b9f0ff76747b1b57019e5ea23a33b9bfbb1a0eb8668748da53`.

A read-only single-arm implementation check then ran the actual
`audit_graph_metric_training_pair.inspect_arm` routine. The final checkpoint
and progress bytes matched the manifest; initial.pt matched the declared
joint57 initial SHA. All964 input pins and60 source pins matched. The check
found325 finite tensors,65 changed encoder tensors,14 changed policy-head
tensors,zero changed value-head tensors, and80 optimizer parameter states,
all at step4,096. Counts were524,288 metric draws,131,072 policy-state draws,
and262,144 added graph draws. These are repeated draws, not unique knots or
new MCTS trajectories.

This is a **single-arm implementation/completion check**, not the final paired
audit and not a quality result. No intermediate or final TEST scores have been
used to change the protocol.

The same serial launcher automatically started the control process (PID29933,
parent77947 at the snapshot). Its actual command selected mode=control and the
matching output directory. Advancing CPU and the preparation manifest confirmed
live input preparation with the same initial checkpoint/config. The core
evaluation controller recognized the graph completion and now waits only on
that control worker. No duplicate worker or manual replacement was launched.
