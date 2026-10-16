# What protocol97 actually trained on

2026-09-21. Protocol121 reproduced the complete data stream of BOTH completed
arms. No model was run or updated, no TEST scores were calculated and the active
MCTS protocol was unchanged. This is a data-coverage audit, not a new learning
experiment or a conclusion about whether embeddings are useful.

## Reproduction and checks

Root: `artifacts/raster-train-draw-coverage-v1/`.
Report SHA256:
`1541d03cff334a5f0377ea132746e07e986b98f7fc5987a8f735c78ccf86752d`.
Input report SHA256:
`011f34946580c8321a92739e5eacb0b8322cec6a892cf1c2db09e49547b02960`.
Pool-support SHA256:
`73e6f967f96da83d9c8389a1fca2b4d63d7eef9248cf31f41f935cd7c7b2d905`.
All2048 draw hashes SHA256:
`cca987e5225b7dea98e0e50d67c023c17b64bc75efb70d8b056bc945eda0f255`.

The final chain is
`ef8ee40a53aa71d9c3361330fd483077d3922c6ebfec6eb6b1e2a2a2c2c69e41`,
exactly the value committed by BOTH trained arms. Scheduled counts, full-group
sizes and visits also match both manifests. All1153 pinned files were rehashed.
The existing loaders replay/validate the frozen inputs before sampling; only
TRAIN rows enter the measurements. Other corpus partitions may be integrity
checked by that loader, but no held-out statistics or scores are used here.

Separately, the2048-element hash chain was reconstructed with direct hashlib/JSON,
source-count and interval-count sums were checked, normalized weight fractions
sum to1, and the original pool's interval strengthening and raw endpoint sets
were recomputed directly from its JSON records. Three targeted tests passed;
Ruff passed; the audit process exited0. The distinct-draw statistics are produced
by the published audit implementation, not an independent second sampler run.

## Presentations versus diversity

Per arm:65536 metric draws, but10344 different UNORDERED RAW endpoint pairs and
13586 different raw `(strand count, word)` representations. These are not counts
of distinct knots, isotopy classes or independent statistical examples. Both
arms saw the same stream, so summing their counts does not double diversity.

| Source | Pool records | Actual draws | Distinct drawn endpoint pairs | Mean normalized example-weight share |
| --- | ---: | ---: | ---: | ---: |
| Original | 4802 | 14336 | 2416 | 31.618% |
| Successful suffix to unknot | 6301 | 16384 | 3545 | 44.828% |
| Ordinary graph | 50615 | 32768 | 4516 | 22.169% |
| Chiral distance2 sentinel | 1 | 2048 | 1 | 1.386% |

The combined distinct-pair count is lower than the sum because sources overlap.
Weights above mean the average across batches of each source's sum of witness
priority weights divided by the batch total. They are NOT fractions of the
observed numerical loss, parameter gradients, or optimization benefit. No neural
forward was performed to estimate those quantities in this audit.

The graph supplies50% of rows but only22.17% of normalized example weight. Its
audited edges are literal crossing changes and paths have length1 or2. Therefore
their witness-semantic-length priority is1 even when their words are long. This
is consistent with the source builder/replay contract, not a missing long-witness
metadata bug. The priority concerns the witness path, not endpoint word length.

Graph sampling visited4516/50615 available endpoint pairs (8.922%) and7156/58237
available raw endpoint representations (12.288%). It samples uniformly by bound
class, then invariant endpoint group, then record. This deliberately differs from
uniform record sampling; large groups do not automatically receive their share
of records, and small groups can be revisited often. Low unique coverage alone
does not establish that this balancing choice is wrong.

## Zero-CC supervision really was present

Actual exact-zero draws:7665/65536 (11.70%). ALL7665 have different raw endpoints
and different word lengths;7305 also change strand count. No exact-zero draw is
merely a pair of identical representations. Their mean normalized weight share
is32.011%, distinct from both draw fraction and actual loss contribution.

Of these,1795 compare two non-anchor representations (2.739% of all metric draws)
and5870 include the empty-one-strand unknot anchor. The1795 all come from the
original corpus. Its full general/exact0 pool contains512 records distributed
over19 conservative `(determinant, absolute signature)` buckets. Those19 buckets
are NOT19 proved knot types. The original pool has1024 exact-zero records in all,
including512 anchor pairs; all have different raw endpoints.

The graph sampler excludes its three exact-zero canonical-equality pairs, but
this does not remove zero-CC supervision from the mixed training stream. The
claim "the embedding was never trained to identify equivalent representations"
is contradicted by the actual data. Conversely, observing these pairs does not
prove that the trained model learned robust invariance or extrapolates to longer
words. Fixed endpoint pairs and arbitrary new zero-CC views are different tests.

Actual metric draw intervals:

| Certified interval | Draws |
| --- | ---: |
| [0,0] | 7665 |
| [0,1] | 811 |
| [0,2] | 1104 |
| [1,1] | 19035 |
| [1,2] | 16342 |
| [2,2] | 20579 |

The original sampler uses protocol97's existing determinant-based lower-bound
strengthening. Consequently its reported sampled/pool intervals need not equal
the unstrengthened interval histogram in the raw corpus JSON. This audit did not
change those bounds. All draw bounds remain between0 and2.

## Policy data and scale limits

Positive supervision:16384 visits,10302 distinct full resource/action-context
hashes and the same number of context-plus-positive-target hashes,256 distinct
instance hashes,6386 distinct raw representations. L10 accounts for8286 visits
and L1000 for8098. Metric and policy representation sets overlap; do not add their
distinct counts to infer total distinct training knots or representations.

Relative supervision:128 complete-group updates with counts33/33/34/28 for groups
of334/52/74/54 rows;16766 full-state visits. EVERY such row has L1000 context.
The four groups are not millions of fresh self-play trajectories. Training97
reuses frozen metric/positive/relative inputs and does not regenerate MCTS games
inside its optimizer loop. The separately running evaluation is not online RL.

Actual metric inputs reached word length254 and21 strands. Actual positive
policy inputs reached length208 and18 strands. The network's ability to accept
larger inputs, demonstrated in engineering checks, must not be mistaken for
having trained on those larger sizes or for solving them successfully.

## Implications for the next substantial experiment

Do not discard embeddings from this limited static-data experiment. Neither
"no invariance supervision" nor "millions of independent examples" describes
what ran. A larger count of optimizer draws alone would not meet the intended
million-representation/self-play objective.

For a NEW contract, track unique raw representations and endpoint pairs alongside
draws and replayed games. Assess within-group coverage/repetition before changing
group balancing; possible proposals include bounded without-replacement cycles
inside groups and replay-certified zero-CC views of sampled endpoints. Those are
proposals, not changes made here. Preserve leakage boundaries, exact witnesses,
set-valued positive actions, full shared-prefix cancellation and censored failure
semantics. Mirror images must not be silently treated as zero-CC-equivalent knots.

Combine this coverage evidence with119's conditioning diagnosis and the still
pending FULL paired MCTS/replay comparison before selecting a larger training
run. The present audit does not select hyperparameters, release a checkpoint,
import database discoveries, or prove improvement in short distances/L10/L1000.
