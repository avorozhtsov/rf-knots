# Public-knot search: input readiness, not a launched campaign

2026-09-23. Read-only inventory following the user's proposal to connect
embedding/policy training to meaningful UnknotDB expansion and upper-bound
improvements. Protocol97 continues unchanged. This document does not authorize
or register a new search, training run, DKT promotion, database import or release.

## Measured public-input coverage

Update: the completed source refresh in133 finds important differences in
current official DKT bounds and a corrected v4 paper. Read133 before using the
historical counts below as contemporary public targets.

Counts below were recomputed from the local JSON files, not inferred from old
processes or checkpoint names. Public intervals refer to the **2026-08-14**
KnotInfo snapshot; they are not verified September values.

| Frozen input | Named rows | Stored starts | Fits initial L48/N12 input |
| --- | ---: | ---: | ---: |
| DKT72 authors-workbook PD-to-braid panel | 72 | 72 | 70 |
| KnotInfo open-gap catalogue | 482 | 51 | 51 |

The two DKT exceptions are `13a_579` and `13a_656`: each has word length53
and10 strands. Across DKT72, maximum word length is53 and maximum strands10.
Do not truncate either input or omit these failures from a full-panel denominator.
Input fit alone does not prove that all intermediate states of a solution fit.

The51 available gap-catalogue starts comprise37 historical intervals[1,3]
and14 intervals[2,4]. Their words have length12..35 and at most8 strands;
their stored names have no overlap with DKT72 names. This is not a proof of
independence from training or an independently revalidated knot identification.
A route with at most2 or3 crossing changes, respectively,
would improve that historical snapshot, subject to exact identity and replay.
Whether it improves today's public bound requires a fresh source comparison.
The other431 entries have no stored braid in this manifest:426 are outside
the local crossing table and five were excluded by a historical strand cap.
The latter five are12a_1037,12a_1097,12a_1113,12a_156,12a_392. They require
recovering and validating actual representations, not merely lifting a flag.

The executable DKT starts are standard workbook PD conversions, not the
authors' inflated search diagrams or their witness paths. The gap catalogue's
scalar intervals likewise supply no replayable external witness.

## Existing model evidence and restrictions

`artifacts/qrun-baseline-input-check-v2/report.json` records exact native weight
loading and finite forwards for frozen mid-Q254 raster and Q304 strand graph,
both configured for L48/N12. The checkpoint file hashes were rechecked today:

- Q254 raster: `36d1122f5494d502e556994083a1a69adf1643d5be95cd9f80ffc13b68e68d63`.
- Q304 proof-distilled strand graph:
  `0c7a3f3d5b2a08b4bc9c307958264378f8a17148438f0eae7e3d5e93b1739dfa`.

These are available verified candidates, **not a fresh selection of the best
Q-Run solver**. The Q304 checkpoint is specifically the proof-distilled parent,
not the RL-control branch. The historical August31 exact-common report favored
the strand-graph RL control on its ten-task prefix; it was not a terminal
all-model ranking. Do not substitute that label for a checkpoint or compare
its score directly with protocol97's different tasks and resources.

The Q305/Q314 descendant promotion is BLOCKED; a larger generation number is
not evidence of better quality. Existing DKT72-PD-v2 registration is also
blocked with zero selected candidates. Its capacity, retention, exposure and
selection requirements must not be bypassed by renaming a run. The prepared
four-task shadow proxy uses Q204, not Q254 or Q304, and is not a completed test.

Registered-root overlap checks in102 do not establish full Q-pretraining
independence. For public witness discovery, prior exposure does not invalidate
a correctly replayed solution. For claims of held-out model generalization,
exposure matters and must be disclosed/audited separately. A database-building
training cohort must not silently become the held-out benchmark.

## Next executable comparison: unresolved decisions before launch

1. Refresh and pin the public catalogue snapshot. Keep the August14 comparison
   as a separately labelled historical baseline. Verify each target's starting
   representation and named identity, without using weak invariant agreement
   as an identity proof.
2. Select actual Q baseline checkpoints using a current matched comparison,
   or explicitly label the already verified Q254/Q304 pair as limited baseline
   candidates. Preserve native controller position and legal action semantics.
3. Freeze candidate identities, splits, checkpoints, objective budgets, seeds
   and replay rules before scoring. The51 stored starts are an immediate input
   pool, not a predeclared leakage-free test set or an automatic launch command.
4. Compare public-bound reproduction and improvement separately from L10/L1000
   route-cost improvement. Count neural evaluations, controller plies, semantic
   moves and elapsed time separately; equal MCTS simulation counts need not
   imply equal work. Preserve unsupported and budget-censored denominators.
5. If graph-tail lookup is added, compare policies under the same frozen graph
   and lookup contract. Replay the prefix, exact attachment identity and full
   tail. CC cost is prefixCC+tailCC; L10/L1000 include all tail moves. This is
   a separate search configuration, not an unannounced protocol97 optimization.
6. Import only independently replayed results into a new candidate database.
   Search failures are not lower bounds or automatic negative action labels.
   Separate coverage gains, improvements over our database, and improvements
   over contemporary public bounds.

## Release boundary

The September21 release preflight was inspected and both proof-file hashes
rechecked today: published101213 representations versus candidate106310,
5097 added keys and674 improved finite upper bounds on common representations.
The latter are historical graph gains, not674 distinct knot types or new public
theorems. No old keys are missing and no common upper bounds regressed in that
hash-bound audit. This inventory did not rerun the full edge replay.

The candidate does not yet meet the user's20% growth condition. Representation
growth alone requires at least121456 total keys,15146 more than the candidate.
Distinct verified knot-type growth needs its own measured denominator; it is
not established by vertex counts or raw identification labels. The proposal
to require20% in both categories remains explicit rather than conflating them.
All graph-dependent sidecars must be coherently rebuilt, the disputed identity
resolved or quarantined, and release/query/replay gates passed before upload.
See `../unknotdb/docs/release-preparation-2026-09-21.md` in the sibling project.

## Input hashes measured for this inventory

| File, relative to rf-knots | SHA256 |
| --- | --- |
| benchmarks/dkt2026-table1-authors-pd-braids-v1.json | `83c591d4b8ea002e5082ede424766fe7eb05c87958defe70413f679495a289fb` |
| benchmarks/dkt2026-table1-knotinfo-20260814.json | `6fa1a9bd03287bc20463fd2de469b8b5eac6087c83eb848be39de673d7697156` |
| benchmarks/knotinfo-unknotting-gap-candidates-20260814.json | `f8706d67d74adc332bb3cd94728c9d76f1c5b98387d31c60c734d14d45b61d10` |
| artifacts/qrun-baseline-input-check-v2/report.json | `f4641b47ef49f344aa4b4574027ebcfb8476a376dec87927be009c24962fab6d` |

No current97 process inspection was performed between scheduled hourly checks.
The next inspection remains no earlier than11:18 UTC on September23.

## September23 follow-up: direct proof-graph coverage measured

Evidence: `artifacts/public-gap-unknotdb-coverage-v1/report.json`, SHA256
`e285014ba90a4608d40e6e5604c3283041d2fe8b2da74a1f34312d17f12bea32`.
All51 input names were joined to133's refreshed intervals. The existing Rust
runtime supplied mirror/origin-normalized keys; SHA256 of its canonical encoding
was checked and the decoded database representation matched exactly for every
hit. Both databases were opened read-only. Source/report/database/runtime and
decoder hashes were unchanged across the audit. No policy inference or MCTS.

| Exact-start result | Published v0.10.1 | Existing local candidate |
| --- | ---: | ---: |
| Direct normalized-key miss | 1 | 1 |
| Stored U above current public upper | 49 | 45 |
| Stored U equals current public upper | 1 | 5 |
| Stored U below current public upper | 0 | 0 |

The missing direct start is12n_307 (length28,7 strands). This is not proof that
the knot type is absent: initial reduction/policy preprocessing and alternate
representations were deliberately not searched. Likewise, the table concerns
the exact starting representation's stored route, not the best claim across
every identified representation of its knot type.

Five candidate starts match their public upper:12a_815 at4CC and12n_167,
12n_304,12n_721,12n_522 at3CC. The latter four improved relative to the release,
from11,9,9,14CC respectively. These are prior database gains, not new search
results generated by this inventory. No full witness replay was repeated here;
the earlier release preflight binds these exact graph files.

Examples with large recorded gaps:12n_118 has18CC versus public upper3,
12n_818 has17 versus3, and12n_101/12n_443 have16 versus3. Before assigning
these to expensive learned search, compare existing preprocessing/graph-tail
access and deterministic constructive baselines. Large stored U is not intrinsic
knot complexity, and a public number is not an available executable tail.

The first audit attempt incorrectly compared packed database storage bytes with
the runtime's unpacked canonical bytes. It stopped before writing a report.
The corrected audit uses the existing UnknotDB storage decoder and compares
decoded strands/flags/word, while retaining the canonical-key SHA check. This
was a harness encoding mistake, not evidence of corrupt proof data.

These45 starts give a concrete database-route improvement cohort distinct from
the51-target public-bound improvement task. They cannot establish20% knot-type
growth:50 starts already hit the graph, and the remaining type is not shown
absent. Their training exposure also remains unaudited; do not call them an
unseen benchmark merely because this inventory is new.

## September23 follow-up: constructive CC certificates generated and replayed

`artifacts/public-gap-descending-baseline-v1/report.json`, SHA256
`b77f05666e18c883fa369fecd431bd3b04ebd66cc4701d79134c7512f5b367d7`,
contains51 standalone certificates with exact input words, crossing positions,
intermediate words and encoded checkpointed proof programs. A private probe
links the existing UnknotDB Rust library without modifying that library or its
production CLI. The report pins the probe, Rust source, library, caller and
input inventory, and records compiler version. Required local compile flags
include `--edition=2021 -C lto=thin -C opt-level=3`; omitting Rust-side LTO
failed against the existing LLVM bitcode, before any certificate was generated.

The algorithm is the existing descending-v0 traversal from top strand0.
It chooses the smaller of the descending flip set and its complement (ascending),
with ties selecting descending. Each resulting program applies the selected
crossing changes and the existing `DescendingCollapse` terminal checker.
Encoding/decoding round-trip and exact replay toB1/empty succeeded for all51.
There are37 descending and14 ascending-complement certificates. Positive and
negative trefoil controls each required1CC; a two-component closure and a zero
generator were rejected. All reported certificates respect the refreshed
public lower bound. These controls do not independently prove named identity.

Twelve certificates improve the **recorded route for the exact start** in the
unchanged candidate graph:

| Representation name | Stored CC upper | Constructive certificate CC |
| --- | ---: | ---: |
|12n_873|8|6|
|12a_1115|11|7|
|12a_147|11|7|
|12a_148|11|7|
|12a_692|11|7|
|12a_816|11|7|
|12n_75|11|7|
|12n_216|14|12|
|12n_101|16|12|
|12n_443|16|10|
|12n_818|17|11|
|12n_118|18|12|

The directly missing12n_307 start also receives a12CC certificate. None of the
51 certificates matches or improves its current public upper bound. Generation
and Rust replay across51 subprocesses took about0.17 seconds after compilation;
this excludes compilation, input audits, graph import and any semantic expansion.
It is not a measured neural-search speedup.

Critical boundary: `DescendingCollapse` certifies a zero-CC topological tail,
but does not expand that tail into the RF MCTS semantic move alphabet. Therefore
L10/L1000 and semantic move counts are explicitly null; policy-training eligibility
is false. These are valid certificates under the existing UnknotDB checker,
not51 policy rollouts or short-distance exact-u labels. They are not imported.
An eventual graph import must split one-CC edges, replay preprocessing/attachment
and update routes under the actual database contract. Candidate proof SHA remains
de54cc3d2f36758c05915781559af8f00cc993f2770a914f66b965916d30b726.

This establishes a cheap constructive baseline the next learned search should
beat, while leaving all45 below-public-quality starts as public-bound targets.
It does not justify changing the frozen97 experiment, calling the embedding
useful, claiming new public mathematics, or publishing a database release.

## September23 follow-up: bounded Q254 root-exposure comparison

`artifacts/public-gap-q254-root-exposure-v1/report.json`, SHA256
`8614c6dd4233ba56886de210d426edc9699e8ba81e563398e1ffbfdc5c9aa742`,
compares51 public starts with102's254 registered Q roots and its227-root
prior-plus-completed-prefix subset. It rechecked61 relevant input/source pins,
including native banks/stage JSONs and frozen checkpoint/metadata, before and
after comparison. It did not repeat the unrelated synthetic TEST audits.

Neither subset has a matching registered name, cyclic word, or cyclic/mirror
word. However,45/51 public starts have determinant/absolute-signature collisions
with registered roots, including roots in the completed-prefix subset. For
example12a_815 has(95,abs(signature)=2), shared with registered11a_82 and10_116.
These weak invariants do not identify a knot: collisions are not evidence of
training leakage, and their absence would not prove complete independence.

`independence_proven=false` remains explicit. This covers only registered Q254
roots, not all inherited pretraining or MCTS/replay descendants, Q304 exposure,
or RF97 graph-supervision exposure. Do not discard the45 collisions to manufacture
a six-case "clean" result. Retain51 as a clearly labelled diagnostic/discovery
pool unless a separate full exposure and split audit supports a held-out claim.
No weights, split ownership, running jobs or production data were changed.
