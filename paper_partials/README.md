# Paper partials

This directory collects paper-ready evidence and reusable fragments before they
are assembled into the manuscript. Suitable contents include:

- verified facts and concise conclusions;
- draft paragraphs and definitions;
- figures and their source data;
- tables and metric definitions;
- experimental protocols, commands, and environment manifests; and
- negative results or caveats that materially affect interpretation.

Every experimental partial should identify:

1. the date and experiment/run identifier;
2. the exact repository commit and configuration;
3. the source artifact or command used to derive it;
4. the evaluated representation set and seed(s);
5. the definition and denominator of every reported metric; and
6. whether the result is preliminary, verified, or superseded.

Prefer descriptive filenames such as
`2026-08-09-foundation-pretrain-admission-table.md`. Keep generated figures
beside their source data or link both from the partial. Do not copy an aggregate
number here without preserving enough provenance to reproduce it.

## Current experiment evidence

- [`2026-08-16-single-knot-mastery-v1-v2-interim.md`](2026-08-16-single-knot-mastery-v1-v2-interim.md): five-scientist mastery snapshot separating strict improvements from relaxed training evidence, with registered exhaustion stops and dose-calibration decisions.
- [`2026-08-16-q20-six-strand-capacity-cliff.md`](2026-08-16-q20-six-strand-capacity-cliff.md): live Q20 gate snapshot showing perfect lower-strand acquisition followed by universal failure on the first six-strand presentation.
- [`2026-08-16-r200-paired-snapshot.md`](2026-08-16-r200-paired-snapshot.md): exact-identity five-arm R200 snapshot, including solved-set and completed-set differences for static, adaptive, and frozen random-order comparisons.
- [`2026-08-14-r200-static-random-no-sharing.md`](2026-08-14-r200-static-random-no-sharing.md): registered fifth R200 arm using a frozen random order, canonical post-R24 states, no sharing, and ten-rung checkpoints.
- [`2026-08-13-interim-r200-innovation-and-objective-ablation.md`](2026-08-13-interim-r200-innovation-and-objective-ablation.md): paired interim R200 evidence, invariant-scientist results, innovation metric, capacity mutations, durable native logging, and the planned L1000 objective-mixture ablation.
- [`2026-08-12-optimized-four-arm-r200-and-large-banks.md`](2026-08-12-optimized-four-arm-r200-and-large-banks.md): immutable four-arm launch and 2,700-representation continuation banks.
- [`2026-08-12-human-invariant-oracle-family.md`](2026-08-12-human-invariant-oracle-family.md): invariant feature definitions and the five-architecture admission protocol.
- [`2026-08-12-search-evaluation-optimization.md`](2026-08-12-search-evaluation-optimization.md): search/evaluation performance engineering and its measured speedups.

## Internal diagnostics excluded from paper claims

- [`DKT72-PD-v1 five-strand capacity diagnostic`](../research/diagnostics/2026-08-14-dkt72-pd-v1-five-strand-capacity-diagnostic.md): 47/72 presentations were unsupported and no strict upper-bound improvement was found. Retain for engineering provenance; retry only with genuinely trained 12-strand scientists.
