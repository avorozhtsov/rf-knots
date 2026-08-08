# Representation experiments

Supporting code for [18 — the braid diagram as a picture](../18-raster-representation.md).
These are **probes, not training runs**: they ask whether an encoder can compute
the things the agent needs, which takes minutes, rather than whether an agent wins
more, which takes days. Read the verdicts in note 18 with that scope in mind.

Everything imports `rf_knots` and `torch`. The repository does not depend on
torch, so install it into the project environment before running:

```bash
uv pip install torch
```

## The two experiments

| script | question |
|---|---|
| `probe_run.py` | Which member of the raster family is best, at matched parameters? |
| `window_probe.py` | Does swapping `s-window-128`'s input for a raster help, and where? |

`probe_data.py` builds the exact labels (`destab`, `isknot`, `determinant`,
`distance`); `probe_models.py` holds the arms and the width solver that matches
their parameter counts.

## Running them

```bash
.venv/bin/python research/experiments/probe_run.py --output artifacts/raster-probe --workers 5
```

```bash
.venv/bin/python research/experiments/window_probe.py --output artifacts/window-probe --workers 5
```

Both write `run.log`, `rows.json` and `report.md` into `--output`, which is under
the untracked `artifacts/`.

## Three things that will bite

**Pin the thread count.** Each worker sets `torch.set_num_threads(1)`, but torch
still opens its pools from the environment. Export the limits before Python
starts, or eight workers become three hundred threads and the wall-clock numbers
become fiction:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

**The label cache is keyed by `(probe, split, seed)`, not by the flags.** Labels
land in `<output>/datasets/` and are reused if present, which is what makes a
re-run cheap. Change `--train`, `--test` or `--max-len` and you must delete that
directory, or you will silently rerun against the old data.

**Equal parameters are not equal compute.** The raster canvas is `max_strands`
times taller than the one-hot strip, so at 100k parameters it does roughly eight
times the arithmetic. Both sweeps carry an explicit `-8x` arm that spends that
arithmetic on the one-hot encoding instead; quoting a raster win without it is
quoting a bigger network.
