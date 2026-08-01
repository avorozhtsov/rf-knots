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
