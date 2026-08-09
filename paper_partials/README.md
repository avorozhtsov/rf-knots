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
