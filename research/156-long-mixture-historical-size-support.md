# Historical149 TRAIN size support: complete pool audit

2026-09-24. Finite audit completed with exit0 (PID67167,session15614), without
neural forwards or optimizer steps. New report:
`artifacts/long-mixture-historical-size-support-v1/report.json`, SHA256
`1e0b3e58c0456ae0153a03d308350548949e9f914e30516b6ef53c3214146b82`.

`scripts/audit_long_mixture_size_support.py` loads the same frozen samplers and
policy-target pools as149, verifies their source/input hashes, and recomputes
the full historical metric support against the completed121 pool audit. It
inventories every available policy input, including full relative groups, not
only the states sampled during one2048-step run. It performs no new TEST score
and does not inspect or alter live139/147 outputs.

| Historical source | Representation appearances | Distinct raw representations | Maximum word length | Maximum strands |
| --- | ---: | ---: | ---: | ---: |
| Original metric endpoints |9604|5848|208|17|
| Short-u suffix endpoints |12602|6302|208|17|
| Ordinary-graph metric endpoints |101230|58237|254|21|
| Positive-policy states |19189|8768|208|18|
| Relative-policy entries |514|451|128|17|
| Mirror sentinel endpoints |2|2|3|2|
| Combined, union for distinct count |143141|72556|254|21|

Appearances include duplicates and, for metric rows, both endpoints. Distinct
counts use exact raw `(strands,word)` identity, not invariant groups, canonical
knot identity or independent training games. Do not sum per-source distinct
counts. The policy columns concern input word/strand sizes, not every budget,
legal-action or positional feature. All representation-size histograms and
combined appearance arithmetic were separately checked after publication.

## Consequence for prospective size evaluation

Every historical source here is below385 letters and34 strands. Protocol147
separately limits its new TRAIN endpoints to at most384 letters and33 strands;
its completed consumer must enforce those limits before optimization. Thus,
subject to that required consumer,150's original-plus-lift design supplies
lifted input views at least385/34 beyond the maximum word length AND strand
count of all supplied149 optimizer pools, not merely beyond sampled maxima.

This is a size-support statement, not a prediction of useful extrapolation.
The lifted views are correlated with their originals, not new independent knot
types. Full earlier Q254 pretraining/exposure is not audited by this inventory;
do not generalize the conclusion to all historical pretraining. Original150
queries may themselves cross some size limits; classify actual rows rather
than assuming every original is within-size. Model quality and L10/L1000 search
effects still require the complete prospective evaluation.

The audit also fills a precision gap in121: its reported positive-policy maxima
came from16384 sampled visits. Here the full positive pool is checked; its size
maxima agree, but the full pool contains8768 distinct raw representations versus
6386 observed in those older sampled visits. Neither count is a knot count.

## Reproduction and implementation verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  scripts/audit_long_mixture_size_support.py \
  --out artifacts/long-mixture-historical-size-support-v2/report.json
```

Use a genuinely new path; the completedv1 report and source are preserved.
Six size-accounting tests and the related controller/training-audit tests passed
(33 total). Ruff and formatting passed. Separate preparation fixes ensure new
distance/cache/search-audit/training-audit outputs can create missing parent
directories only after validation; their mathematical contracts are unchanged.
None of these newly prepared consumers was used by the live139/147 jobs.
