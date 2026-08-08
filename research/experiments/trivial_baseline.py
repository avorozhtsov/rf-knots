r"""What two scalars already answer, before any network is trained.

Every probe in `probe_data` can be attacked without looking at the diagram at all,
using only the **word length** and the **strand count** — both of which the
environment already broadcasts as observation scalars, and both of which every arm
therefore gets for free. A network that scores below this line has learned
nothing; a network that scores just above it has learned almost nothing.

This is not a hypothetical worry. Two of the four probes turn out to be largely
decided by these two numbers:

* `isknot` is a **parity** question in disguise. The closure's permutation is a
  product of `L` transpositions on `k` letters, and an `k`-cycle is even exactly
  when `k` is odd, so being a knot forces `L = k - 1 (mod 2)`. That single rule is
  most of the label.
* `distance` at the instance sizes the exact oracle can reach is close to a
  function of how long the word is and how many strands it has, because the
  optimal solution is mostly "destabilise `k-1` times and undo the walk".

Reporting probe scores without this line would credit the representation for
arithmetic on two broadcast scalars. Run it against the same cached datasets the
sweep used, so the comparison is on identical instances:

    .venv/bin/python research/experiments/trivial_baseline.py \
        --cache artifacts/raster-probe-20260808/datasets
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_data import PROBE_KIND  # noqa: E402


def _load(cache: Path, probe: str, split: str, seed: int):
    path = cache / f"{probe}-{split}-seed{seed}.pkl"
    return pickle.loads(path.read_bytes()) if path.exists() else None


def baseline(cache: Path, probe: str, seed: int) -> dict[str, float]:
    """Lookup on `(len(word), strands)`, fitted on train, backing off to the
    global majority (or mean), which is the best a constant predictor can do."""
    kind, _ = PROBE_KIND[probe]
    train = _load(cache, probe, "train", seed)
    if train is None:
        return {}
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for word, strands, label in train:
        buckets[(len(word), strands)].append(label)
    if kind == "regress":
        table = {key: float(np.mean(v)) for key, v in buckets.items()}
        fallback = float(np.mean([label for _, _, label in train]))
    else:
        table = {key: Counter(v).most_common(1)[0][0] for key, v in buckets.items()}
        fallback = Counter(label for _, _, label in train).most_common(1)[0][0]

    scores: dict[str, float] = {}
    for split in ("in", "wide"):
        test = _load(cache, probe, split, seed)
        if test is None:
            continue
        predicted = np.array(
            [table.get((len(w), k), fallback) for w, k, _ in test], dtype=float
        )
        truth = np.array([label for _, _, label in test], dtype=float)
        if kind == "regress":
            residual = float(((predicted - truth) ** 2).mean())
            variance = float(((truth - truth.mean()) ** 2).mean())
            scores[split] = 1.0 - residual / max(variance, 1e-9)
        else:
            scores[split] = float((predicted == truth).mean())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    lines = [
        "# The two-scalar baseline",
        "",
        "Word length and strand count only, as a lookup table fitted on the same",
        "training split each arm saw. Both scalars are already broadcast into every",
        "observation, so this is the floor a representation has to clear to have",
        "contributed anything. `mean ± half-range` over seeds.",
        "",
        "| probe | in | wide |",
        "|---|---:|---:|",
    ]
    for probe in PROBE_KIND:
        per_seed = [baseline(args.cache, probe, seed) for seed in range(args.seeds)]
        per_seed = [s for s in per_seed if s]
        if not per_seed:
            continue
        cells = []
        for split in ("in", "wide"):
            values = [s[split] for s in per_seed if split in s]
            cells.append(
                f"{np.mean(values):.3f} ± {(max(values) - min(values)) / 2:.3f}"
                if values
                else "--"
            )
        rows.append({"probe": probe, "in": cells[0], "wide": cells[1]})
        lines.append(f"| `{probe}` | " + " | ".join(cells) + " |")
        print(f"{probe:12s} in={cells[0]}  wide={cells[1]}", flush=True)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "baseline.json").write_text(json.dumps(rows, indent=2))
        (args.output / "baseline.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
