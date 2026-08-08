r"""Swapping the input representation of `s-window-128`, and measuring it.

`s-window-128` is the serial controller from
[research/12](../12-serial-formulation.md): a head that sees seven cells of the
word and may act at any of them. Its observation is the environment's per-letter
one-hot -- one channel per `sigma_g` and one per `sigma_g^-1`. This module keeps
the geometry, the head, and the parameter budget, and changes only what fills
those seven cells: the same seven letters drawn as a `7 x k` picture.

## Why local legality is the right thing to measure

The controller's whole job is to know which rewrite is available where, and the
three local rewrites are decided by *arithmetic on generator indices*:

| move | legal when |
|---|---|
| `REDUCE(p)` | `word[p] = -word[p+1]` |
| `COMMUTE(p)` | `abs(abs(word[p]) - abs(word[p+1])) >= 2` |
| `BRAID(p)` | `abs(abs(word[p]) - abs(word[p+1])) = 1`, same sign, `word[p+2] = word[p]` |

Read those in the one-hot encoding and each is a relation over pairs of symbols
from a `2(n-1)`-letter alphabet: something to be memorised cell by cell, and
memorised again for every generator index the training data happened to contain.
Read them in the raster and they are statements about whether two crossings
*touch*: `COMMUTE` is "their two-column footprints are disjoint", `BRAID` is "they
share exactly one column". Those are local geometry, and they do not care which
generator index it is.

So the prediction this module tests is specific: the raster should not merely
match the one-hot encoding, it should **transfer to strand counts it never saw**,
because the geometry is the same picture wherever it sits on the strand axis.
The one-hot encoding has no route to that -- the channels an eight-strand braid
lights up were dead for its whole training run.

This is a supervised probe, not a ladder run. It answers "can the encoder see the
thing" in minutes rather than "does the agent win more" in days; the second
question is worth asking afterwards and on the strength of this answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_models import MAX_STRANDS, Block, Encoding, _groups, count_parameters  # noqa: E402

WINDOW = 7  # `s-window-128`'s seven cells
NARROW = (2, 3, 4)
WIDE = (6, 7, 8)

# Offsets at which each move's whole pattern is inside the window. A pairwise
# move needs its partner and a braid needs two of them, so the last offsets carry
# no label rather than an unanswerable one.
SLOTS = (("reduce", WINDOW - 1), ("commute", WINDOW - 1), ("braid", WINDOW - 2))
LABELS = sum(width for _, width in SLOTS)


def window_sample(rng: np.random.Generator, strands: int, max_len: int):
    """A cyclic seven-cell window of a random word, and the legality inside it.

    The window is taken cyclically because the closure makes the word a necklace,
    which is what the serial adapter already does.
    """
    length = int(rng.integers(WINDOW, max_len + 1))
    word = [
        int(rng.choice((-1, 1))) * int(rng.integers(1, strands)) for _ in range(length)
    ]
    head = int(rng.integers(0, length))
    cells = [word[(head + offset) % length] for offset in range(WINDOW)]

    labels = np.zeros(LABELS, dtype=np.float32)
    index = 0
    for offset in range(WINDOW - 1):
        a, b = cells[offset], cells[offset + 1]
        labels[index + offset] = float(a == -b)
    index += WINDOW - 1
    for offset in range(WINDOW - 1):
        a, b = cells[offset], cells[offset + 1]
        labels[index + offset] = float(abs(abs(a) - abs(b)) >= 2)
    index += WINDOW - 1
    for offset in range(WINDOW - 2):
        a, b, c = cells[offset], cells[offset + 1], cells[offset + 2]
        labels[index + offset] = float(
            c == a and (a > 0) == (b > 0) and abs(abs(a) - abs(b)) == 1
        )
    return tuple(cells), strands, labels


def build(strand_counts, count: int, seed: int, max_len: int = 20):
    rng = np.random.default_rng(seed)
    return [window_sample(rng, int(rng.choice(strand_counts)), max_len)
            for _ in range(count)]


class WindowNet(nn.Module):
    """Trunk plus the positional readout `s-window-128` actually uses.

    Its policy head reads a `1x1` convolution over the window cells rather than a
    pooled summary, because "which rewrite is available *here*" is a question
    about a cell and pooling erases the answer. The raster arm keeps that head
    exactly and only adds a masked pool down the strand axis to get back to one
    feature per column -- so the two arms differ in their input and nowhere else.
    """

    def __init__(self, encoding: Encoding, *, width: int, blocks: int,
                 wrap_position: bool, wrap_strands: bool):
        super().__init__()
        self.encoding = encoding
        self.input = nn.Conv2d(encoding.channels, width, 1, bias=False)
        self.blocks = nn.Sequential(
            *[Block(width, wrap_position=wrap_position, wrap_strands=wrap_strands)
              for _ in range(blocks)]
        )
        # Built only for the raster arm. An unused module would still be counted
        # by `_match`, which would quietly hand the one-hot arm fewer *live*
        # parameters than its matched budget claims.
        self.row_project = (
            nn.Conv1d(2 * width, width, 1) if encoding.kind == "raster" else None
        )
        self.row_norm = (
            nn.GroupNorm(_groups(width), width) if encoding.kind == "raster" else None
        )
        self.readout = nn.Conv1d(width, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.blocks(torch.relu(self.input(x)))
        if self.encoding.kind == "raster":
            mask = x[:, 3:4]
            weight = mask.sum(dim=2).clamp(min=1.0)
            mean = (hidden * mask).sum(dim=2) / weight
            peak = (hidden + (mask - 1.0) * 1e4).amax(dim=2)
            columns = torch.relu(self.row_norm(self.row_project(
                torch.cat([mean, peak], dim=1))))
        else:
            columns = hidden[:, :, 0, :]
        cells = self.readout(columns)  # (B, 3, WINDOW)
        return torch.cat(
            [cells[:, kind, :span] for kind, (_, span) in enumerate(SLOTS)], dim=1
        )


ARMS = {
    "window-onehot": dict(
        encoding=Encoding("word"), wrap_position=False, wrap_strands=False, scale=1.0,
        rationale="what `s-window-128` reads today",
    ),
    "window-onehot-8x": dict(
        encoding=Encoding("word"), wrap_position=False, wrap_strands=False,
        scale=float(MAX_STRANDS),
        rationale="the control: the raster's compute, spent on one-hot letters",
    ),
    "window-raster": dict(
        encoding=Encoding("raster", edges=True, pad_mode="identity"),
        wrap_position=False, wrap_strands=False, scale=1.0,
        rationale="the same seven letters, drawn",
    ),
    "window-raster-noedge": dict(
        encoding=Encoding("raster", pad_mode="identity"),
        wrap_position=False, wrap_strands=False, scale=1.0,
        rationale="drawn, with the strand boundaries unmarked",
    ),
}


def _encode(encoding: Encoding, samples) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([encoding(cells, strands, WINDOW) for cells, strands, _ in samples])
    y = np.stack([labels for _, _, labels in samples])
    return x, y


def _match(spec: dict, target: int, blocks: int) -> tuple[int, int]:
    goal = target * spec["scale"]
    best, gap = 8, None
    for width in range(8, 513, 2):
        net = WindowNet(spec["encoding"], width=width, blocks=blocks,
                        wrap_position=spec["wrap_position"],
                        wrap_strands=spec["wrap_strands"])
        total = count_parameters(net)
        if gap is None or abs(total - goal) < gap:
            best, gap = width, abs(total - goal)
        if total > goal * 1.6:
            break
    net = WindowNet(spec["encoding"], width=best, blocks=blocks,
                    wrap_position=spec["wrap_position"],
                    wrap_strands=spec["wrap_strands"])
    return best, count_parameters(net)


def run_one(job: dict) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    spec = ARMS[job["arm"]]
    width, parameters = _match(spec, job["target"], job["blocks"])
    net = WindowNet(spec["encoding"], width=width, blocks=job["blocks"],
                    wrap_position=spec["wrap_position"],
                    wrap_strands=spec["wrap_strands"])

    train = build(NARROW, job["train"], 1000 + job["seed"], job["max_len"])
    test_in = build(NARROW, job["test"], 5000 + job["seed"], job["max_len"])
    test_wide = build(WIDE, job["test"], 9000 + job["seed"], job["max_len"])

    x, y = (torch.from_numpy(a) for a in _encode(spec["encoding"], train))
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    started = time.perf_counter()
    net.train()
    for _ in range(job["steps"]):
        index = torch.randint(0, len(x), (job["batch"],))
        loss = loss_fn(net(x[index]), y[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    row = {"arm": job["arm"], "rationale": spec["rationale"], "seed": job["seed"],
           "width": width, "parameters": parameters,
           "seconds": time.perf_counter() - started}
    with torch.no_grad():
        for split, samples in (("in", test_in), ("wide", test_wide)):
            xs, ys = (torch.from_numpy(a) for a in _encode(spec["encoding"], samples))
            logits = net(xs)
            correct = ((logits > 0).float() == ys).float()
            row[split] = float(correct.mean())
            # Every window is legal at all of nothing most of the time, so a
            # network that answers "no" everywhere already scores well on the
            # mean. The per-move rate on the *positive* slots is the number that
            # cannot be reached by refusing to predict.
            row[f"{split}_recall"] = float(
                (correct * ys).sum() / ys.sum().clamp(min=1.0)
            )
            for kind, (name, span) in enumerate(SLOTS):
                start = sum(w for _, w in SLOTS[:kind])
                block = correct[:, start : start + span]
                row[f"{split}_{name}"] = float(block.mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--test", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--target", type=int, default=102_439)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    jobs = [
        {"arm": arm, "seed": seed, "steps": args.steps, "train": args.train,
         "test": args.test, "batch": args.batch, "blocks": args.blocks,
         "max_len": args.max_len, "target": args.target}
        for arm in ARMS
        for seed in range(args.seeds)
    ]
    log(f"{len(jobs)} runs: {len(ARMS)} arms x {args.seeds} seeds, "
        f"{args.steps} steps each")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs):
            rows.append(row)
            log(f"  {row['arm']:22s} seed {row['seed']} "
                f"in={row['in']:.3f}/{row['in_recall']:.3f} "
                f"wide={row['wide']:.3f}/{row['wide_recall']:.3f} "
                f"({row['parameters']:,}p, {row['seconds']:.0f}s) "
                f"[{len(rows)}/{len(jobs)}]")
            (args.output / "rows.json").write_text(json.dumps(rows, indent=2))
    (args.output / "report.md").write_text(report(rows))
    log("done")


def report(rows: list[dict]) -> str:
    lines = [
        "# Swapping `s-window-128`'s input",
        "",
        "Legality of the three local rewrites at every offset of the seven-cell",
        "window. `acc` is over all slots; `recall` is over the legal ones only,",
        "which is the number a network cannot reach by answering \"illegal\"",
        "everywhere. `in` is 2-4 strands, as trained; `wide` is 6-8 strands, never",
        "seen. `mean ± half-range` over seeds.",
        "",
        "| arm | params | in acc | in recall | wide acc | wide recall | what it reads |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        runs = [r for r in rows if r["arm"] == arm]
        if not runs:
            continue
        cells = []
        for key in ("in", "in_recall", "wide", "wide_recall"):
            values = [r[key] for r in runs]
            cells.append(f"{np.mean(values):.3f} ± {(max(values)-min(values))/2:.3f}")
        lines.append(
            f"| `{arm}` | {runs[0]['parameters']:,} | " + " | ".join(cells)
            + f" | {runs[0]['rationale']} |"
        )
    lines += ["", "## Per move, on the wide split", "",
              "| arm | reduce | commute | braid |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        runs = [r for r in rows if r["arm"] == arm]
        if not runs:
            continue
        cells = [f"{np.mean([r[f'wide_{name}'] for r in runs]):.3f}"
                 for name, _ in SLOTS]
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
