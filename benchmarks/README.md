# Frozen benchmarks

`rungs-v1.json` removes ladder repetition and assigns each distinct braid-word
instance to a stable train, validation or test split by its SHA-256 identity.
Regenerate it only when deliberately versioning the benchmark:

```bash
uv run python scripts/freeze_rung_benchmark.py
uv run rf-knots benchmark-check benchmarks/rungs-v1.json
```

The external hard-unknot corpus is catalogued but not vendored. Downloading a
mutable archive is not enough for a result: record the archive SHA-256, freeze
every parsed instance into a manifest, and publish that manifest with the run.
Never tune on the `test` partition.

The rung manifest is a calibration set, not a claim of external generalisation.
The external test sets and any adversarially generated transfer set must remain
separate.

## DKT 2026 Table 1 upper-bound targets

[`dkt2026-table1-upper-bounds-v1.json`](dkt2026-table1-upper-bounds-v1.json)
freezes all 72 rows printed in Table 1 of Dranowski--Kabkov--Tubbenhauer,
*RL unknotter, hard unknots and unknotting number*. It preserves the table's
historical KnotInfo and paper-workbook intervals. It is a target catalogue, not
an executable representation set and not a collection of the authors'
unknotting witnesses.

[`dkt2026-table1-knotinfo-20260814.json`](dkt2026-table1-knotinfo-20260814.json)
overlays a pinned current KnotInfo snapshot. Of the 72 rows, 48 are now exact and
24 remain open. The exact rows are a calibration panel; only the 24 open rows
can currently yield a strict official upper-bound improvement.

[`dkt2026-table1-authors-pd-braids-v1.json`](dkt2026-table1-authors-pd-braids-v1.json)
is the executable 72-instance test manifest. Each braid was deterministically
converted by Spherogram from the named knot's PD presentation in the authors'
pinned workbook. These standard workbook presentations are not the inflated
diagrams on which the paper found its improvements, so representation source
must be reported with every result.

The short benchmark name is **DKT72-PD-v1**: DKT identifies Dranowski,
Kabkov, and Tubbenhauer; 72 is the complete published Table 1; and PD records
that the executable braids come from the authors-workbook PD presentations.

Regenerate all three files from pinned source snapshots with:

```bash
uv run --with snappy --with openpyxl --with pandas --with xlrd --with-editable . \
  python scripts/build_dkt2026_table1_dataset.py \
  --paper-pdf /path/to/2603.07955v3.pdf \
  --authors-workbook /path/to/upperbounds/data/unknotting.xlsx \
  --knotinfo-xls /path/to/knotinfo_data_complete.xls
uv run rf-knots benchmark-check \
  benchmarks/dkt2026-table1-authors-pd-braids-v1.json
```

## KnotInfo open-gap campaign

[`knotinfo-unknotting-gap-candidates-20260814.json`](knotinfo-unknotting-gap-candidates-20260814.json)
freezes all 482 knots whose pinned KnotInfo interval is `[1,3]`, `[1,4]`, or
`[2,4]`. It overlays the exact stored local braid when available: 51 candidates
are immediately searchable, five were excluded only by the historical local
strand cap, and 426 lie outside the at-most-12-crossing local table.

The stored braid is a starting representation, not evidence for a bound and not
an enumeration of the knot type's crossing-change neighbourhood. Reproduce the
catalogue with:

```bash
uv run --with pandas --with xlrd --with-editable . \
  python scripts/build_unknotting_gap_candidates.py \
  --knotinfo-xls /path/to/knotinfo_data_complete-2026-08-14.xls
```

## Unknotting evidence index

[`unknotting-evidence-index-20260815.json`](unknotting-evidence-index-20260815.json)
is the replay-verified cross-run witness database used by the single-knot
mastery curriculum. It maps knot identities to each scientist's best native
crossing-change witness and retains event/bank paths and SHA-256 hashes. It also
contains the two ten-slot curriculum blocks described in
[`research/20-single-knot-mastery.md`](../research/20-single-knot-mastery.md).

Regenerate it from a current recovery mirror with:

```bash
uv run python scripts/build_unknotting_evidence_index.py \
  --results ../pgx-mcts-bench/artifacts/nebius-semantic-v2-live-backup/mirror/results \
  --output benchmarks/unknotting-evidence-index-20260815.json
```

The pseudo-scientist `knotinfo-shortest-evidence` is fail-closed: a KnotInfo
scalar upper bound or bibliography link is not assigned an L10 score. It becomes
rankable only when the source supplies a complete semantic action path that
passes the same replay verifier as native evidence.
