r"""Testing the torus properly: wrapping at the live strand count, not the canvas.

## Why this file exists

The `raster-torus` arm in `probe_run.py` glues strand `0` to strand `7` with
`F.pad(mode="circular")` on a fixed eight-row canvas. That is **not** the torus
anyone means. A braid on `n` strands lives on a circle of circumference `n`, and
`n` changes during play — stabilisation and destabilisation are two of the moves.
Wrapping at the canvas edge therefore glues strand `0` to a row that is *inactive*
whenever `n < 8`, which is most of the time on the narrow split. It asserts a
symmetry that is false even in the affine braid group.

So the first sweep's verdict on the torus is a verdict on a bug. This module runs
the arm that was meant:

| arm | strand-axis padding |
|---|---|
| `raster-cylinder-edge` | bounded, both boundaries marked — the reference |
| `raster-torus-canvas` | circular at the canvas width, reproducing the flawed arm |
| `raster-torus-n` | circular **at the live strand count** — the real proposal |

## The gather

For a sample with `n` active strands, the padded tensor of height `H + 2` must be

```
row 0        -> strand n-1        (so strand 0 sees strand n-1 above it)
row 1..n     -> strands 0..n-1
row n+1      -> strand 0          (so strand n-1 sees strand 0 below it)
row > n+1    -> zero
```

A `3x3` valid convolution over that gives output row `i` a view of strands
`i-1 mod n`, `i`, `i+1 mod n` for every `i < n`, which is what a circle of
circumference `n` means. Rows at or beyond `n` are masked out downstream. A single
static `circular` pad cannot express this, because the wrap point differs per
sample within a batch.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_data  # noqa: E402
from probe_models import Encoding, _groups, count_parameters  # noqa: E402


def wrap_strands_at(x: Tensor, strands: Tensor) -> Tensor:
    """Pad the strand axis circularly at `strands[b]`, per sample."""
    batch, channels, height, width = x.shape
    source = torch.cat([x, x.new_zeros(batch, channels, 1, width)], dim=2)
    rows = torch.arange(height + 2, device=x.device)[None, :]
    n = strands.clamp(min=1, max=height)[:, None]
    index = torch.where(
        rows == 0,
        n - 1,
        torch.where(
            rows <= n,
            rows - 1,
            torch.where(rows == n + 1, torch.zeros_like(rows), torch.full_like(rows, height)),
        ),
    ).clamp(0, height)
    gather = index[:, None, :, None].expand(batch, channels, height + 2, width)
    return source.gather(2, gather)


class TorusBlock(nn.Module):
    """A residual block whose strand-axis wrap point is a per-sample input."""

    def __init__(self, width: int, *, mode: str):
        super().__init__()
        self.mode = mode  # "bounded" | "canvas" | "live"
        self.conv1 = nn.Conv2d(width, width, 3, padding=0, bias=False)
        self.norm1 = nn.GroupNorm(_groups(width), width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=0, bias=False)
        self.norm2 = nn.GroupNorm(_groups(width), width)

    def _pad(self, x: Tensor, strands: Tensor) -> Tensor:
        # The position axis is a circle in every arm: that is conjugation, and it
        # is not what is under test here.
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        if self.mode == "live":
            return wrap_strands_at(x, strands)
        return F.pad(x, (0, 0, 1, 1), mode="circular" if self.mode == "canvas" else "constant")

    def forward(self, x: Tensor, strands: Tensor) -> Tensor:
        hidden = torch.relu(self.norm1(self.conv1(self._pad(x, strands))))
        return torch.relu(x + self.norm2(self.conv2(self._pad(hidden, strands))))


class TorusProbeNet(nn.Module):
    def __init__(self, encoding: Encoding, *, width: int, blocks: int, outputs: int, mode: str):
        super().__init__()
        self.encoding = encoding
        self.input = nn.Conv2d(encoding.channels, width, 1, bias=False)
        self.blocks = nn.ModuleList(TorusBlock(width, mode=mode) for _ in range(blocks))
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, outputs))

    def forward(self, x: Tensor) -> Tensor:
        mask = x[:, 3:4]
        # The live strand count, read off the occupancy channel rather than passed
        # in: the network is given no more than the picture already contains.
        strands = mask[:, 0, :, 0].sum(dim=1).round().long()
        hidden = torch.relu(self.input(x))
        for block in self.blocks:
            hidden = block(hidden, strands)
        weight = mask.sum(dim=(2, 3)).clamp(min=1.0)
        mean = (hidden * mask).sum(dim=(2, 3)) / weight
        peak = (hidden + (mask - 1.0) * 1e4).amax(dim=(2, 3))
        return self.head(torch.cat([mean, peak], dim=1))


ARMS = {
    "raster-cylinder-edge": ("bounded", True,
                             "bounded strand axis, both boundaries marked"),
    "raster-torus-canvas": ("canvas", False,
                            "wrapped at the canvas width -- the flawed arm, reproduced"),
    "raster-torus-n": ("live", False,
                       "wrapped at the live strand count -- the proposal, tested properly"),
    "raster-torus-n-edge": ("live", True,
                            "wrapped at the live strand count, boundaries also marked"),
}


def _match(mode: str, edges: bool, target: int, blocks: int, outputs: int):
    encoding = Encoding("raster", edges=edges, pad_mode="identity")
    best, gap = 8, None
    for width in range(8, 257, 2):
        total = count_parameters(
            TorusProbeNet(encoding, width=width, blocks=blocks, outputs=outputs, mode=mode)
        )
        if gap is None or abs(total - target) < gap:
            best, gap = width, abs(total - target)
        if total > target * 1.6:
            break
    net = TorusProbeNet(encoding, width=best, blocks=blocks, outputs=outputs, mode=mode)
    return encoding, best, count_parameters(net)


def run_one(job: dict) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    probe, seed = job["probe"], job["seed"]
    kind, outputs = probe_data.PROBE_KIND[probe]
    mode, edges, rationale = ARMS[job["arm"]]
    encoding, width, parameters = _match(mode, edges, job["target"], job["blocks"], outputs)
    net = TorusProbeNet(encoding, width=width, blocks=job["blocks"], outputs=outputs, mode=mode)

    cache = Path(job["cache"])

    def load(split):
        path = cache / f"{probe}-{split}-seed{seed}.pkl"
        return pickle.loads(path.read_bytes()) if path.exists() else None

    def encode(samples):
        x = np.stack([encoding(w, k, job["rows"]) for w, k, _ in samples])
        y = np.asarray([v for _, _, v in samples], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    x_train, y_train = encode(load("train"))
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if kind == "binary" else nn.CrossEntropyLoss()

    started = time.perf_counter()
    net.train()
    for _ in range(job["steps"]):
        index = torch.randint(0, len(x_train), (job["batch"],))
        logits = net(x_train[index])
        target = y_train[index]
        loss = loss_fn(logits[:, 0], target) if kind == "binary" else loss_fn(logits, target.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    row = {"arm": job["arm"], "rationale": rationale, "probe": probe, "seed": seed,
           "width": width, "parameters": parameters,
           "seconds": time.perf_counter() - started}
    with torch.no_grad():
        for split in ("in", "wide"):
            samples = load(split)
            if samples is None:
                continue
            x, y = encode(samples)
            logits = net(x)
            row[split] = float(
                ((logits[:, 0] > 0).float() == (y > 0.5)).float().mean()
                if kind == "binary"
                else (logits.argmax(dim=1) == y.long()).float().mean()
            )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True,
                        help="the datasets/ directory of a previous probe_run")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--target", type=int, default=102_439)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--probes", default="destab,isknot")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    jobs = [
        {"arm": arm, "probe": probe, "seed": seed, "cache": str(args.cache),
         "steps": args.steps, "batch": args.batch, "blocks": args.blocks,
         "rows": args.rows, "target": args.target}
        for probe in args.probes.split(",")
        for arm in ARMS
        for seed in range(args.seeds)
    ]
    log(f"{len(jobs)} runs, cache {args.cache}")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs):
            rows.append(row)
            scores = " ".join(f"{k}={row[k]:.3f}" for k in ("in", "wide") if k in row)
            log(f"  {row['probe']:8s} {row['arm']:22s} seed {row['seed']} {scores} "
                f"({row['parameters']:,}p, {row['seconds']:.0f}s) [{len(rows)}/{len(jobs)}]")
            (args.output / "rows.json").write_text(json.dumps(rows, indent=2))
    (args.output / "report.md").write_text(report(rows, args.probes.split(",")))
    log("done")


def report(rows, probes) -> str:
    lines = ["# The torus, wrapped at the right place", "",
             "`raster-torus-canvas` is the arm from the first sweep, which wrapped at",
             "the eight-row canvas rather than at the live strand count.",
             "`raster-torus-n` is the same idea done correctly. `mean ± half-range`.", ""]
    for probe in probes:
        subset = [r for r in rows if r["probe"] == probe]
        if not subset:
            continue
        lines += [f"## `{probe}`", "", "| arm | params | in | wide | what it is |",
                  "|---|---:|---:|---:|---|"]
        for arm in ARMS:
            runs = [r for r in subset if r["arm"] == arm]
            if not runs:
                continue
            cells = []
            for split in ("in", "wide"):
                values = [r[split] for r in runs if split in r]
                cells.append(
                    f"{np.mean(values):.3f} ± {(max(values)-min(values))/2:.3f}"
                    if values else "--"
                )
            lines.append(f"| `{arm}` | {runs[0]['parameters']:,} | " + " | ".join(cells)
                         + f" | {runs[0]['rationale']} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
