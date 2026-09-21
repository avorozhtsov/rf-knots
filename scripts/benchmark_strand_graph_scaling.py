"""Pinned-checkpoint size smoke on connected, positive torus braid words.

This checks finite computation with unchanged weights, NOT knot-solving quality.
Sparse CC candidate scoring deliberately does not enumerate O(L*N) insertions.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path

import torch

from rf_knots.actions import CROSSING_CHANGE
from rf_knots.mcts_loop import SitePolicyValueNet
from rf_knots.reference import num_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--hierarchy-levels", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite benchmark")
    manifest = json.loads((args.run / "manifest.json").read_text())
    sha = hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest()
    if manifest["status"] != "completed" or sha != manifest["checkpoint_sha256"]:
        raise ValueError("require a completed hash-verified checkpoint")
    checkpoint = torch.load(args.run / "checkpoint.pt", weights_only=False, map_location="cpu")
    torch.set_num_threads(1)
    architecture = dict(checkpoint["architecture"])
    if bool(args.window_size) != bool(args.hierarchy_levels):
        raise SystemExit("--window-size and --hierarchy-levels must be enabled together")
    if args.window_size:
        architecture.update(
            window_size=args.window_size, hierarchy_levels=args.hierarchy_levels
        )
    net = SitePolicyValueNet(**architecture).eval()
    loaded = net.load_state_dict(checkpoint["model"], strict=not bool(args.window_size))
    benchmark_budget_dim = net.budget_dim
    shapes = {k: list(v.shape) for k, v in net.state_dict().items()}
    report = {
        "status": "running",
        "pid": os.getpid(),
        "checkpoint_sha256": sha,
        "architecture": architecture,
        "checkpoint_architecture": checkpoint["architecture"],
        "random_initialized_keys": list(loaded.missing_keys),
        "parameter_shapes": shapes,
        "parameters": sum(p.numel() for p in net.parameters()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": 1,
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("src/rf_knots/strand_graph_encoder.py"),
                Path("src/rf_knots/embedding_normalization.py"),
                Path("src/rf_knots/mcts_loop.py"),
                Path("src/rf_knots/site_head.py"),
            )
        },
        "rows": [],
        "caveat": "Forward-only CPU engineering smoke, not solving or generalisation evidence. "
        "Three measured forwards per row after one warmup. Sparse CC candidates only; "
        "full search still has an O(L*N) INSERT enumeration bottleneck. "
        "RSS is the process cumulative high-water mark, including model loading.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.output.with_suffix(".partial")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output)

    save()
    with torch.inference_mode():
        for strands, repeats in [
            (2, 65),
            (2, 257),
            (2, 1025),
            (2, 4097),
            (9, 2),
            (65, 2),
            (257, 2),
            (2049, 2),
            (8193, 2),
        ]:
            assert math.gcd(strands, repeats) == 1
            word = tuple(range(1, strands)) * repeats
            assert num_components(word, strands) == 1
            words = torch.tensor([word])
            lengths, ns = torch.tensor([len(word)]), torch.tensor([strands])
            positions = sorted({0, len(word) // 2, len(word) - 1})
            sites = torch.tensor([[[CROSSING_CHANGE, p, 1, 1] for p in positions]])
            valid = torch.ones((1, len(positions)), dtype=torch.bool)
            budget_values = [1000.0, 2.0]
            if benchmark_budget_dim == 5:
                # Use generous residual size budgets so the smoke exercises the
                # full resource-conditioned interface without imposing a
                # particular search problem on the synthetic words.
                budget_values.extend([1_000_000.0 - len(word), 1_000_000.0 - strands, 1000.0])
            budget = torch.tensor([budget_values])
            net(words, lengths, ns, sites, valid, budget)
            times = []
            for _ in range(3):
                started = time.perf_counter()
                logits, value, effective = net(words, lengths, ns, sites, valid, budget)
                times.append(time.perf_counter() - started)
                assert torch.isfinite(logits).all() and torch.isfinite(value).all()
                assert effective.all()
            encoded = net.encoder.encode_sites(words, lengths, ns)
            assert encoded["tokens"].shape[1] == len(word)
            assert all(
                torch.isfinite(encoded[k]).all() for k in ("tokens", "knot", "layout", "isotopy")
            )
            assert shapes == {k: list(v.shape) for k, v in net.state_dict().items()}
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() != "Darwin":
                rss *= 1024
            row = {
                "strands": strands,
                "length": len(word),
                "repeats": repeats,
                "components": 1,
                "finite": True,
                "sparse_candidates": len(positions),
                "seconds": times,
                "median_seconds": sorted(times)[1],
                "process_peak_rss_bytes": rss,
            }
            report["rows"].append(row)
            save()
            print(json.dumps(row), flush=True)
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
