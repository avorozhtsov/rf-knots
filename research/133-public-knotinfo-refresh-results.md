# Public-bound refresh: historical DKT72 labels are not current authority

2026-09-23. Source-only audit. No search, training, proof database or frozen
benchmark was modified. This supplements132 and changes how the next public
evaluation must be labelled; it is not an upper-bound improvement by our solver.

## Acquisition and raw-data checks

The historical URL `https://knotinfo.org/knotinfo_data_complete.xls` returned404.
The [official download page](https://knotinfo.org/homelinks/database_download.php)
now links to the `knotinfo-org/assets` repository's zipped XLS. The downloaded
main-branch ZIP and a second retrieval at commit
`1ed2e4606072529372e19288e12cd3ce7aa1f056` had identical SHA256.
That asset commit is dated2026-09-16T14:38:33Z. Retrieval date is September23.

Local evidence directory: `artifacts/public-knotinfo-refresh-20260923-v1/`.
The source XLS was read without alteration using bundled Python and an isolated
xlrd2.0.2 reader dependency. The only worksheet is `sample_dat`,12968 rows and
248 columns. `name` is columnA, `unknotting_number` AG, references AH. Duplicate
names and unsupported required interval syntax fail closed; all482+72 requested
rows were located. These cohorts are not an exhaustive fresh catalogue.

| Evidence | SHA256 |
| --- | --- |
| Downloaded ZIP | `1aa3337b4767a2d655a7305999d62ef861920dc3d615d6b03f966d2f17f04f8a` |
| Extracted XLS | `f4de747845b6432636c49e2dfa84b1c8d3bab681ee12b27402898cb3e4883be0` |
| comparison.json | `2de05e3ea2738ea436441a421c91a203239d9d4cdd96da1e33883f1b63ca03be` |
| historical-audit.json | `2478b0cb24ff739295746f27327852817564f1ae6594c29c21cf14bfafaf3de3` |
| dkt-v4.html | `c2c63ad27763c6ed8c0e5116c70b0f79289b21640ab04b80eb2cb0f76fa6059a` |
| knotinfo-unknotting-description.html | `5259f7cbab78e8218391d21ac819f45111b66236aa16b014cebc6883395f8eb0` |

The historical XLS in `tmp/knotinfo_data_complete-2026-08-14.xls` still matches
its pinned SHA1829b7056eefc653f77a42acc9e471df4020e64fe93a8b6a211ca3d49fe86e7b.
A separate read of its cells matched all554 previous interval records exactly.
This rules out an overlay extraction mismatch for those records; it does not
prove what the remote site served historically or explain all source changes.
Both reports retain exact row locations and their script/input hashes.

## Measured changes from the locally frozen August14 cohorts

| Cohort | Rows | Interval changed | Smaller upper | Larger upper | Currently exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Available gap starts | 51 | 1 | 0 | 0 | 0 |
| All historical gap candidates | 482 | 33 | 0 | 3 | 0 |
| Historical DKT72 | 72 | 70 | 0 | 70 | 2 |

The51 available starts remain usable as an input pool. Current intervals are
37x[1,3],13x[2,4],1x[3,4]. The sole changed entry is11a_354: [2,4] to[3,4],
at `sample_dat!A606,AG606`. A verified three-CC route would meet the newly
listed lower bound, but that lower-bound attribution still needs its own source
verification before an exact-u claim. This audit generated no such route.

For example, current cellsAG266/AG335/AG642 give[2,3] for11a_14/11a_83/11n_23,
where the historical local source gave2. This is a public-source discrepancy,
not evidence that we can silently overwrite replayed bounds in UnknotDB.

## Primary-source explanation and revised interpretation

[KnotInfo's current description](https://knotinfo.org/descriptions/unknotting_number.html)
explicitly says the DKT work is pending and its results are not installed.
It therefore does not support treating the August DKT overlay as current
official KnotInfo truth.

[DKT v4, September9, Section6D](https://arxiv.org/html/2603.07955v4#S6.SS4)
now reports15 checked witnesses from a corrected run, rather than our frozen
v3 Table1's72 rows. Its identification checks are numerical, not interval
certificates. It reports that all15 were found before invoking the RL reducer.
The version also warns that arbitrary Jones monomial shifts cannot establish
normalized-polynomial equivalence. These are the authors' statements, not
independently replayed imports into our database.

## Consequence for next work

- Keep DKT72 as a labelled **historical target set**, not the current list of
  accepted public improvements. Do not edit pinned source/labels mid-experiment.
- Record separately: current official catalogue bound, historical/paper claim,
  and our complete replay-verified bound. A win against one is not a win against
  all three. A looser new catalogue value does not automatically revoke a
  tighter independently valid witness or establish scientific novelty.
- Before ranking a new public campaign, audit v4's15 target-specific witness
  materials and exact identity links. Do not adopt the old workbook's recursive
  bounds or an invariant-only endpoint match as a verified proof tail.
- The51-input pool avoids the DKT72 source-label issue at the name level, but
  still needs train/test exposure disclosure, native action/resource checks and
  a search protocol. Availability is not authorization or a launched job.

This refresh contributes source provenance, not new training draws or search
successes. No current97 state was polled between hourly inspections. The next
scheduled inspection remains11:18 UTC; ongoing computation was not altered.
