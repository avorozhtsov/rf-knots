# Tiny paired-search parity passed; no short-word speedup established

2026-09-21. Protocol114 completed in a separate process/output directory. The
active uncached protocol97 evaluation was neither modified nor polled outside
the hourly schedule. No training/checkpoint updates or scientific promotion.

## Execution and verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/probe_raster_cache_search.py \
  --pgx-root /Users/artemvorozhtsov/projects/pgx-mcts-bench \
  --out artifacts/raster-cache-paired-search-gate-v1
```

The command exited0. Output directories must be new. Report:
`artifacts/raster-cache-paired-search-gate-v1/report.json`.
SHA256: `31b0bc8530fe03b355283d5ae3167a49bb8c39137b7a181ff8e44027ced077c1`.

After completion, all139 source/checkpoint/gate pins and all12 saved run hashes
were independently rechecked. Saved trajectories were replayed again, counters
recomputed, and each comparison reconstructed from the saved run files. Three
pairs were cached-first and three uncached-first, as declared. Fourteen focused
tests passed in3.79s without skips; targeted Ruff passed.

## Correctness result

Both final trained arms, three fixed seeds per arm, four tiny synthetic
conditions per run, two inference modes: six paired comparisons and12 runs.
All six had exactly equal recorded discrete transcripts, tree statistics,
incumbents, witnesses, L10/L1000 and logical search counters. Chosen-action
probabilities also matched exactly in these runs (maximum difference0).
All model/objective tensors stayed unchanged. Actual CNN-input counts agreed
with cache misses; the cache did not replace global graph/policy computation.

Saved replay totals:144 continuation episodes,414 continuation moves and1908
candidate action scores. All48 condition evaluations found witnesses, but these
are repeated runs of only four conditions on two trivial/trefoil words. They are
**not48 unique knots, not the64-root scientific panel, and not new evidence of
general solver quality**. Successful and censored continuation episodes were both
audited even when the overall condition was eventually solved.

## Timing and computation

Totals over three seeds per model/mode. Time includes search, cache administration
and replay inside the search call; excludes loading/copying, common warmup, JSON
serialization and the subsequent saved-file replay.

| Final model / mode | Actual CNN tiles | Search+replay seconds | Continuation moves |
| --- | --- | --- | --- |
| Q254 / uncached | 568 | 0.439408 | 103 |
| Q254 / cached | 165 | 0.492746 | 103 |
| Random / uncached | 515 | 0.430592 | 104 |
| Random / cached | 94 | 0.501813 | 104 |

Median paired uncached/cached time ratios were0.88849 for Q254 and0.84644 for
random (values below1 mean the cached call was slower). Five of six pairs were
slower with the cache; one was faster. Subsecond timings under concurrent host
load and only three seeds are noisy and do not establish a universal slowdown.
They do establish that reduced CNN tile counts alone cannot justify claiming an
end-to-end speedup. Signature hashing, key generation and bookkeeping can offset
savings on tiny inputs; no timing breakdown isolates their individual costs here.

## Decision and limits

Do not integrate this prototype into the active scientific run. Local numerical
parity, full-network parity and tiny-search transcript parity now have separate
evidence, but a useful runtime benefit on representative longer states has not
been shown. Any further performance gate should predeclare non-TEST workloads
and include the complete inference cost and memory bound, not only count tiles.
Keep the conservative invalidation guarantees; do not trade stale weights for
speed. Training-gradient caching remains unsupported.

This finite gate does not prove equality of all future stochastic trajectories,
especially on long words where112 showed nonzero probability differences. Nor
does it answer the ongoing scientific questions about learned short distances,
Q-transfer benefit, embedding ablation, or lower L10/L1000 on held-out problems.
