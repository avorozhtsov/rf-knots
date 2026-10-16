# Unselected TRAIN representation collection for a budget-conditioned value head

2026-10-02. Corrective protocol after the diversity smoke and first head fit.
This document is prospective; no broad collection or retraining is claimed.

## What went wrong

`raster-budgeted-value-teacher-diverse-v1` used three TRAIN representations
(base indices 35, 122, 80), selected because earlier searches had produced
multiple admissible successful paths. Its 60/60 solved searches therefore
measure a deliberately success-enriched subset, not general search success.
The 32-simulation budget also differs from the earlier 16-simulation smoke.
The first head fit then trained on only two of those representations and used
the third as a held-out TRAIN diagnostic. That fit is not a general-purpose
value head and must not be inserted into MCTS or used for a utility claim.

## Population and split, fixed before looking at new search outcomes

Use every one of the 128 distinct TRAIN `base_root_index` entries from
`artifacts/diverse-cost-mcts-training-v1/roots.json`, each paired at L10 and
L1000. Verify the completed source manifest, root-file SHA, checkpoint SHA,
TRAIN ownership and pair identity. Do not filter by past or newly observed
search success. Keep the 128 fixed in the denominator when reporting root
success; a success-only subset may be described only as such.

Split by `base_root_index`, not episode or state. Freeze the split with a
deterministic seed before collection. Stratify using only pre-search fields:
word length (<=30, 31-60, 61-120, >120) and reference-witness CC (1, 2,
>=3). The reference witness is a known upper bound, not a hardness label.
Report each stratum and both L10/L1000; do not let a high episode count from
one easy representation dominate training or aggregate metrics. At least one
whole-root group must remain an untouched TRAIN-origin diagnostic group;
actual DEV and new independent TEST stay outside training and model selection.

## Collection and labels

Freeze one immutable actor/checkpoint, legal actions, caps, seeds and search
version. Run S=10 independent searches per representation and lambda; S=20
only as a predeclared extension, not selected by individual outcome. Record
the full distribution at separate search budgets (e.g. 16, 32, 128 and 512
tree simulations) without treating one budget as another. A small, evenly
stratified CPU timing pilot may run while 248 is healthy; the full collection
must not compete with healthy 248. Use resumable, hash-checked per-attempt
shards and full replay/ownership checks. This teacher collection is not an
evaluation against heuristics; those comparisons use canonical
`pgx-mcts-bench`.

For the bounded-success output, every TRAIN-owned completed search attempt
contributes one binary outcome for its **actual** node-local simulation
budget. A failure means only no verified solution was found within that
budget. No failed attempt supplies a negative action label or a CC/move target.
For CC/move outputs, use only fully replayed, TRAIN-owned successful paths,
plus separately identified known-solution distillation targets. Preserve the
source type and resource context for each target. Audit quarantine/held-out
ancestry on each continuation; never silently convert excluded episodes into
negative examples. Weight roots, not raw episode multiplicity.

Train and score the multi-output head on several simulation budgets. The
earlier `local_simulations=1` head cannot be interpreted at 32 or 512. Report
success calibration/Brier and conditional CC/move errors per root, stratum,
lambda and budget, including simple constant baselines on the same panel.
Only after satisfactory held-out DEV search at matched compute may its
outputs guide MCTS. The eight-arm tournament sweep remains a separate stage.

## Operational gate

Before any new worker: inspect the current process command and CPU, not an
old PID; inspect source/checkpoint hashes and existing output namespace. If
process inspection is unavailable, do not infer that a worker stopped and do
not launch a duplicate. Preserve all previous failed/smoke artifacts. Do not
change frozen v1/v2 sources, the running 248 source, or its contract.

## First execution note, 2026-10-02

The 248 pair completed at 36,864 optimizer steps per arm, 265,765.95 summed
active training seconds; both final checkpoint SHA256 values matched their arm
manifests. No new teacher weights were needed: the teacher actor remains the
already pinned 32,768-step checkpoint.

`scripts/bootstrap_cost_value_teacher_v3.py` was launched as a bounded
eight-attempt CPU timing pilot under
`artifacts/raster-budgeted-value-teacher-unselected-v1`. Its contract lists
all 128 source TRAIN representations, both weights, ten independent seeds and
simulation budgets 16/32/128/512; **only the first eight attempts have run**.
They cover four representations with word lengths 23, 56, 83, 124, each at
both lambda weights, and a reference-witness CC range 1-8. Seven of eight raw
searches solved; six attempts contained at least one TRAIN-admissible solved
continuation. The eight attempt shards passed SHA checks. Total measured
per-attempt wall time was 39.28 seconds. This small first slice is a runtime
and integrity smoke, not a search-success estimate for all 128 or a trained
value head. The broad collection remains incomplete.

At the next three-hour check, the same saved contract was resumed for 24 more
attempts after process inspection confirmed no matching worker. It now has 32
attempts over 16 distinct representations, each at both lambda weights and
16 simulations. Raw solves: 30/32 (15/16 at each lambda). Attempts containing
at least one TRAIN-admissible solved continuation: 27/32 (L10 14/16, L1000
13/16). Total recorded per-attempt wall time: 169.68 seconds. All 32 shard
SHA256 values matched the manifest. Broad Python source-tree digests were
unchanged before/after the resumed batch: RF scripts/src
`e3b303e8a882fe785d6303322382ad14b6ee5afef7ff6d1928364dc7474a58c6`,
canonical pgx source `66bf1a13e36f1d4dc4a36c154959a5048a870a5723d8318dd5ca45fbd915dbf8`.
These broad digests are an extra audit, not part of the v3 per-task contract.
No general success estimate is claimed from this first, cheap-budget slice.

The first 256 committed attempts now cover all 128 source representations at
both lambda weights for one independent seed and 16 simulations. A separate
replay/ownership audit (`scripts/audit_tournament_teacher_v1.py --through 256`)
rechecked every committed shard SHA, every action sequence and TRAIN admission:
2,915 episodes, 2,091 admitted episodes, 574 admitted solved episodes.
Raw solved attempts: L10 100/128 and L1000 96/128. Attempts with an admitted
solution: L10 89/128 and L1000 84/128. After exact-path deduplication, 101
root/lambda groups had different verified costs and 132 exact
state-and-resource divergence contexts. This exceeds the *data-adequacy
minima* of protocol254 but is only one of the predeclared ten independent
searches, at the smallest budget; it is neither an eight-variant model
comparison nor an out-of-sample utility result. The remaining independent
16-simulation attempts have been launched separately under the original
unchanged v3 contract.
