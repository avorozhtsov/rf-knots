# Prospective matched long-general data experiment

2026-09-24. PREPARATION, NOT LAUNCHED. Protocol147 supply is still building;
139 MCTS remains untouched. This contract defines the data intervention and
training comparison before seeing scores on147 DEV/TEST. Its input hashes and
new short-u search test must be bound before any optimizer run. Engineering
tests or loader readiness are not successful training or search evidence.

## Hypothesis, comparator and scope

Test whether adding general long-to-long short-CC supervision repairs the
underestimated positive distances observed in144/146 while preserving short-u
and improving, or at least not degrading, actual L10/L1000 search. Both arms
start from the SAME Q254 local-raster/pretrained-raster-graph initialization,
with its new upper graph/global/policy modules, not from either135 endpoint.
The architecture still tiles arbitrary-length words and supports variable
strands; a finite experiment does not prove arbitrary-size generalization.

Use the original raw resource feature map in BOTH arms, matching the unmodified
architecture's baseline. Do not compare raw versus units here or assert that139
has selected a winner. Keep width64/global dimension64/two graph rounds,
whole-word embeddings, head-relative positional features and all trainable
parameters unchanged. No residual/value adapter or new oracle input.

Arms `control` and `long`,2048 AdamW updates each, MPSfloat32/fallback disabled,
lr1e-4,weight_decay0.01,grad_clip1,CPUthreads1. The loss implementation and
positive/relative weights remain135/97: positive0.1, full relative group every16
updates with weight0.16, exact existing metric geometry and witness weighting.
Seed20261076 for the historical draw stream and Torch/MPS; seed20261077 for
the additional long-pair stream. Both streams are consumed in both arms,
including discarded shadow samples. Full RNG states and draw digests belong
in resumable checkpoints; compare data/initialization identities independently
after both arms complete. No automatic early stop or DEV checkpoint selection.

## Exactly what changes

Control batch is unchanged in composition:7 original +8 short-u suffix +16
ordinary-graph +1 mirror sentinel =32 metric pairs, plus8 positive policy states.
The long arm replaces the LAST8 ordinary-graph slots with8 new147 TRAIN pairs.
First8 graph slots, all original/suffix/sentinel rows, positive states and chosen
relative group match exactly, update by update. Consume all16 old graph draws
and all8 additional long draws for BOTH arms before selecting actual rows.
Thus a different source's random consumption cannot shift policy targets.

Long pairs sample uniformly by length band, then uniformly among that band's
nonempty certified classes(exact0,exact1,exact2,interval), then uniformly by
unordered endpoint invariant-group pair, then row. Randomly swap endpoints
with probability1/2. Always use five random() calls per record. Every band's
three exact classes must exist, as well as147's stronger predeclared support
gate. Report actual class/group/pair coverage, not draws as unique knots.

The147 record's legacy `witness_semantic_length` field here is only the literal
CC-edge count(0..2), including0 for a cyclic certificate. It is NOT a replayed
RF-MCTS semantic trajectory length. Use it only as the metric source's
unit-weight convention, matching existing ordinary-graph rows. All exchanged
graph/long rows have weight1; per-batch total metric weight, retained rows'
effective weights and policy-loss coefficients therefore remain equal between
arms. Do not convert these records into policy actions, L10/L1000 costs or
complete unknotting witnesses.

The intervention changes size/family/class mixture(including general exact0),
not only length. A win would support this specific replacement, not identify
which component caused it. Both get65,536 metric draws and16,384 positive-state
draws; long uses16,384 new-source draws, not16,384 unique knots. Equal counts
do not imply equal hardware work: report word/strand/tile sizes and elapsed
time, and retain actual search counters.2048 is a first matched experiment,
not proof that million-scale training is unnecessary or sufficient.

## Fail-closed data and evaluation boundaries

Require completed147 corpus and a passed post-exit consumer with
training_ready=true, matching corpus SHA, complete attempt/selected coverage,
all source/artifact hashes, minimum support and exclusions. Corpus metadata
alone must never authorize optimizer use. Re-materialize TRAIN only; verify
every recorded path state's TRAIN ownership and historical-query exclusion.
Fresh147 DEV is separate from optimizer rows; TEST rows are never sampled.
Q254 pretraining independence remains incompletely audited.

Before training, freeze a new independent short-u panel with complete witnesses,
query groups absent from historical optimizer/DEV/TEST queries and147 held-out
queries. TRAIN/query separation must also hold against all147 optimizer path
groups, not just its roots. Do not silently reuse136/138 as fresh TEST. The new
short-u panel must pass its own replay/exclusion consumer and fixed adequacy
gate; missing evidence prevents launch. This panel and147 TEST must be bound
in the new evaluation contract and unchanged throughout training.

Score both initial/final models on147 general TEST and the new short-u original
and size-shift views; report interval violation versus constant1, exact0/1/2,
positive-only ranking, group-weighted aggregates and length/strand strata.
All rankings over pairs of records must disclose their dependency, rather than
claiming pairwise comparisons are independent observations. Freeze evaluation
before scoring; seen136/138 remain exploratory diagnostics only.

Search must compare both arms on the SAME new short-u roots at BOTH L10/L1000,
with an initial checkpoint reference and a distance-prior removal control.
Use the established native state/action/resource and full-replay contracts;
freeze the precise matrix, budgets, seeds and counters before training begins.
No guarantee of equal wall-time from equal simulations, no budget failures as
certified negatives, and no mean costs across different solved subsets.
Report new/lost successes and costs on common successes separately. Removing
the tree's distance prior alone is not an embedding-channel ablation.

Next: finish147/audit, prepare and audit the fresh short-u panel, bind the final
training/evaluation inputs, validate real-weight execution/resume, then actually
run both arms and the independent distance/search evaluations. Do not treat
preparation as completion of the user's embedding goal or a publicDB release.

## Implemented preparation, September24

`scripts/long_general_training_data.py` now implements the required post-audit
loader, separate TRAIN sampler, fresh-DEV materialization and the paired draw
function. No TEST rows are returned to the optimizer. Both arms consume the
same historical and additional RNG streams; shared and selected batch digests
are distinct fields. Tests check unchanged23-prefix/sentinel slots, identical
positive/relative targets, exact five-draw long sampling, both RNG states on
resume, equal metric normalization, ownership/quarantine/exposure, duplicate
and class/size rejection, and refusal of absent/unready/mismatched audits.

`scripts/build_long_mixture_short_u_test.py` and protocol150 prepare the fresh
short-u acquisition/consumer. They cannot create an apparent acquisition run
before147's completed consumer passes. Tests exercise a real newly seeded
TEST-owned witness, its full385/34 lift/replay, exclusion and tamper rejection,
and a held generator lock preventing duplicate launch. The full150 panel has
NOT been generated; a one-record test is not a substitute for its64-root gate.

The combined targeted suite passed81 tests, including the unchanged135/138
checks. Ruff passed. These results validate preparation only. Subsequent runner
and fixed search-matrix preparation is recorded in151; the real-weight MPS
resume gate and complete data audits remain required before launch. Existing
135/139 sources must not be monkeypatched or edited to run the new experiment.
No new production optimizer or MCTS job was launched in these preparation
turns.147/139 were not polled between hourly checks; the next routine combined
process inspection remains no earlier than05:32 UTC.
