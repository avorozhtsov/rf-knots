"""Run the representation probe sweep.

Every arm sees the same data, the same optimizer, the same number of gradient
steps, and a width chosen so the parameter counts match. What differs is the
encoding and the symmetries claimed for it, which is the only way the result can
be read as a statement about representation.

Two test sets are reported for every run:

* **in** -- held-out instances from the strand counts trained on;
* **wide** -- instances on strand counts never seen. This is the claim the whole
  idea rests on, and it is also where a one-hot letter alphabet has nothing to
  say: the channels an eight-strand braid activates were dead throughout its
  training.

Usage:

    uv run python research/experiments/probe_run.py --output artifacts/raster-probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_data  # noqa: E402
import probe_models  # noqa: E402

TARGET_PARAMETERS = 102_439  # `s-window-128`, so the budget is the project's own
PROBES = ("destab", "isknot", "determinant", "distance")


def _encode(arm: probe_models.Arm, samples, rows: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([arm.encoding(word, strands, rows) for word, strands, _ in samples])
    y = np.asarray([label for _, _, label in samples], dtype=np.float32)
    return x, y


def _metric(probe: str, logits, y) -> float:
    """Accuracy for the classifiers, `R^2` for the regression -- both "higher is
    better" and both scale-free, so one table can hold every probe."""
    kind, _ = probe_data.PROBE_KIND[probe]
    if kind == "binary":
        return float((((logits[:, 0] > 0).float()) == (y > 0.5)).float().mean())
    if kind == "class":
        return float((logits.argmax(dim=1) == y.long()).float().mean())
    residual = float(((logits[:, 0] - y) ** 2).mean())
    variance = float(((y - y.mean()) ** 2).mean())
    return 1.0 - residual / max(variance, 1e-9)


def dataset_path(cache: Path, probe: str, split: str, seed: int) -> Path:
    return cache / f"{probe}-{split}-seed{seed}.pkl"


def build_dataset(job: dict) -> dict:
    """Labels are expensive (`distance` runs a search, `determinant` runs Burau)
    and do not depend on the arm, so they are built once and shared by all of
    them. Sharing is also the stronger experiment: every arm is then scored on
    literally the same instances, not on independently drawn ones."""
    import pickle

    path = dataset_path(Path(job["cache"]), job["probe"], job["split"], job["seed"])
    if path.exists():
        return {"path": str(path), "count": -1, "seconds": 0.0}
    started = time.perf_counter()
    samples = probe_data.build(
        job["probe"],
        strand_counts=job["strand_counts"],
        count=job["count"],
        seed=job["label_seed"],
        max_len=job["max_len"],
    )
    path.write_bytes(pickle.dumps(samples))
    return {
        "path": str(path),
        "count": len(samples),
        "seconds": time.perf_counter() - started,
        "probe": job["probe"],
        "split": job["split"],
        "seed": job["seed"],
    }


def run_one(job: dict) -> dict:
    import pickle

    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])

    probe, seed = job["probe"], job["seed"]
    kind, outputs = probe_data.PROBE_KIND[probe]
    arm = next(a for a in probe_models.arms() if a.name == job["arm"])
    arm, parameters = probe_models.match_parameters(arm, TARGET_PARAMETERS, outputs)

    cache = Path(job["cache"])

    def load(split: str):
        path = dataset_path(cache, probe, split, seed)
        return pickle.loads(path.read_bytes()) if path.exists() else None

    train = load("train")
    test_in = load("in")
    test_wide = load("wide")

    net = arm.build(outputs)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = (
        nn.BCEWithLogitsLoss() if kind == "binary"
        else nn.CrossEntropyLoss() if kind == "class"
        else nn.MSELoss()
    )

    rows = job["rows"]
    x_train, y_train = (torch.from_numpy(a) for a in _encode(arm, train, rows))
    batch = job["batch"]
    started = time.perf_counter()
    net.train()
    for _step in range(job["steps"]):
        index = torch.randint(0, len(x_train), (batch,))
        logits = net(x_train[index])
        target = y_train[index]
        if kind == "class":
            loss = loss_fn(logits, target.long())
        elif kind == "binary":
            loss = loss_fn(logits[:, 0], target)
        else:
            loss = loss_fn(logits[:, 0], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    row = {
        "arm": arm.name,
        "rationale": arm.rationale,
        "probe": probe,
        "seed": seed,
        "width": arm.width,
        "parameters": parameters,
        "seconds": time.perf_counter() - started,
        "ms_per_step": (time.perf_counter() - started) / job["steps"] * 1000,
        "train_loss": float(loss.detach()),
    }
    with torch.no_grad():
        for split, samples in (("in", test_in), ("wide", test_wide)):
            if samples is None:
                continue
            x, y = (torch.from_numpy(a) for a in _encode(arm, samples, rows))
            row[split] = _metric(probe, net(x), y)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--test", type=int, default=800)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--probes", default=",".join(PROBES))
    parser.add_argument("--arms", default="")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"

    def log(message: str) -> None:
        # Flushed per line and never buffered through a pipe: a silent background
        # run is indistinguishable from a hung one.
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    selected = [a for a in probe_models.arms()
                if not args.arms or a.name in args.arms.split(",")]
    probes = args.probes.split(",")
    cache = args.output / "datasets"
    cache.mkdir(parents=True, exist_ok=True)

    # `distance` has no wide split. Its label is an exact search depth, and the
    # shortest solution for an eight-strand instance already exceeds the depth the
    # oracle can reach, so a wide set would be selected for being unusually easy
    # rather than being representative. Said out loud because a silently absent
    # column reads as a result.
    def splits(probe: str):
        # A search label costs ~60 ms and a Burau determinant ~20 ms, so those two
        # probes get smaller sets. A smaller set is a noisier number, which is why
        # the size is reported rather than smoothed over.
        divisor = 2 if probe in ("determinant", "distance") else 1
        out = [
            ("train", probe_data.NARROW, args.train // divisor, 1000),
            ("in", probe_data.NARROW, args.test // divisor, 5000),
        ]
        if probe != "distance":
            out.append(("wide", probe_data.WIDE, args.test // divisor, 9000))
        return out

    label_jobs = [
        {
            "cache": str(cache), "probe": probe, "split": split, "seed": seed,
            "strand_counts": counts, "count": count,
            "label_seed": offset + seed, "max_len": args.max_len,
        }
        for probe in probes
        for seed in range(args.seeds)
        for split, counts, count, offset in splits(probe)
    ]
    log(f"labelling {len(label_jobs)} datasets ({args.workers} workers)")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for made in pool.map(build_dataset, label_jobs):
            if made["count"] >= 0:
                log(f"  {made['probe']:12s} {made['split']:5s} seed {made['seed']} "
                    f"{made['count']} labels in {made['seconds']:.0f}s")

    jobs = [
        {
            "arm": arm.name, "probe": probe, "seed": seed, "cache": str(cache),
            "steps": args.steps, "batch": args.batch, "rows": args.max_len,
        }
        for probe in probes
        for arm in selected
        for seed in range(args.seeds)
    ]
    log(f"{len(jobs)} runs: {len(selected)} arms x {len(probes)} probes x "
        f"{args.seeds} seeds, {args.steps} steps each, {args.workers} workers")

    rows: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs):
            rows.append(row)
            scores = " ".join(f"{k}={row[k]:.3f}" for k in ("in", "wide") if k in row)
            log(f"  {row['probe']:12s} {row['arm']:26s} seed {row['seed']} "
                f"{scores}  ({row['parameters']:,}p, {row['seconds']:.0f}s) "
                f"[{len(rows)}/{len(jobs)}]")
            (args.output / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"done in {time.perf_counter() - started:.0f}s")
    (args.output / "report.md").write_text(report(rows, probes, selected))


def report(rows: list[dict], probes, arms) -> str:
    order = [a.name for a in arms]
    lines = [
        "# Representation probe",
        "",
        "Accuracy for `destab`, `isknot` and `distance`; `R^2` for `determinant`.",
        "Higher is better everywhere. `mean +- half-range` over seeds.",
        "",
    ]
    for probe in probes:
        subset = [r for r in rows if r["probe"] == probe]
        if not subset:
            continue
        splits = ["in"] + (["wide"] if any("wide" in r for r in subset) else [])
        lines += [f"## `{probe}`", "",
                  "| arm | params | ms/step | " + " | ".join(splits) + " | what it adds |",
                  "|---|---:|---:|" + "---:|" * len(splits) + "---|"]
        for name in order:
            runs = [r for r in subset if r["arm"] == name]
            if not runs:
                continue
            cells = []
            for split in splits:
                values = [r[split] for r in runs if split in r]
                if not values:
                    cells.append("--")
                    continue
                spread = (max(values) - min(values)) / 2
                cells.append(f"{np.mean(values):.3f} ± {spread:.3f}")
            cost = np.mean([r["ms_per_step"] for r in runs])
            lines.append(
                f"| `{name}` | {runs[0]['parameters']:,} | {cost:.0f} | "
                + " | ".join(cells) + f" | {runs[0]['rationale']} |"
            )
        lines.append("")
    lines += [
        "`ms/step` is wall clock under contention, not a clean benchmark; read it",
        "as an order of magnitude. It is reported because equal parameters are not",
        "equal compute here -- the raster's canvas is eight strands tall where the",
        "one-hot strip is one -- and `word-onehot-cyclic-8x` is the arm that spends",
        "the raster's arithmetic on the word encoding instead.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
