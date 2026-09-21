"""Isolated exact-adjacency optimization experiment; does not alter live search.

The candidate constructs the same CPU integer adjacency through a NumPy buffer
and transfers it once. Only this benchmark process temporarily swaps the helper.
It measures adjacency and the full encoder, NOT end-to-end search or GPU speed.
"""

import argparse
import platform
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from audit_joint_policy_metric_panel import check_pins
from build_mcts_training_supervision import atomic, sha
from run_joint_policy_metric_evaluation import read

from rf_knots import strand_graph_encoder as graph
from rf_knots.mcts_loop import SitePolicyValueNet

ROOTS = Path("artifacts/mcts-policy-metric-evaluation-schema-v2/mcts/parent/roots.json")
RUN = Path("artifacts/embedding-diverse-metric-joint-v1")


def bulk_crossing_adjacency(word, lengths):
    batch, padded = word.shape
    rows = word.detach().cpu().tolist()
    sizes = lengths.detach().cpu().tolist()
    neighbours = np.broadcast_to(
        np.arange(padded, dtype=np.int64)[None, :, None], (batch, padded, 4)
    ).copy()
    for b, (row, length) in enumerate(zip(rows, sizes, strict=True)):
        lanes = {}
        for j, signed_generator in enumerate(row[:length]):
            g = abs(signed_generator)
            for side in (0, 1):
                lanes.setdefault(g + side, []).append((j, side))
        for crossings in lanes.values():
            for i, (j, side) in enumerate(crossings):
                neighbours[b, j, side] = crossings[i - 1][0]
                neighbours[b, j, 2 + side] = crossings[(i + 1) % len(crossings)][0]
    return torch.from_numpy(neighbours).to(device=word.device)


@contextmanager
def use_adjacency(helper):
    original = graph.crossing_adjacency
    graph.crossing_adjacency = helper
    try:
        yield
    finally:
        graph.crossing_adjacency = original


def tensors(instances):
    padded = max(1, max(len(i["word"]) for i in instances))
    return (
        torch.tensor([i["word"] + [0] * (padded - len(i["word"])) for i in instances]),
        torch.tensor([len(i["word"]) for i in instances]),
        torch.tensor([i["strands"] for i in instances]),
    )


@torch.no_grad()
def measure_case(net, args, repeats):
    baseline = graph.crossing_adjacency
    torch.testing.assert_close(
        baseline(*args[:2]), bulk_crossing_adjacency(*args[:2]), rtol=0, atol=0
    )
    expected = net.encoder.encode_sites(*args)
    with use_adjacency(bulk_crossing_adjacency):
        actual = net.encoder.encode_sites(*args)
    if expected.keys() != actual.keys():
        raise AssertionError("encoder output structure differs")
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    times = {f"{stage}/{arm}": [] for stage in ("adjacency", "encoder") for arm in ("old", "bulk")}
    for repeat in range(repeats):
        # Reverse timing order on alternating repetitions to reduce simple warmup/order bias.
        arms = [("old", baseline), ("bulk", bulk_crossing_adjacency)]
        for arm, helper in arms if repeat % 2 == 0 else reversed(arms):
            started = time.perf_counter()
            helper(*args[:2])
            times[f"adjacency/{arm}"].append(time.perf_counter() - started)
            with use_adjacency(helper):
                started = time.perf_counter()
                net.encoder.encode_sites(*args)
                times[f"encoder/{arm}"].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in times.items()}
    return {
        "batch": len(args[1]),
        "lengths": args[1].tolist(),
        "strands": args[2].tolist(),
        "adjacency_equal": True,
        "all_encoder_outputs_bit_equal": True,
        "seconds": times,
        "median_seconds": medians,
        "speedup": {
            stage: medians[f"{stage}/old"] / medians[f"{stage}/bulk"]
            for stage in ("adjacency", "encoder")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        ROOTS,
        RUN / "checkpoint.pt",
        RUN / "manifest.json",
        Path(__file__).resolve(),
        *Path("src/rf_knots").glob("*.py"),
    ]
    pins = {str(p): sha(p) for p in paths}
    manifest = read(RUN / "manifest.json")
    if manifest["status"] != "completed" or manifest["checkpoint_sha256"] != sha(
        RUN / "checkpoint.pt"
    ):
        raise ValueError("require completed unchanged parent model")
    torch.set_num_threads(1)
    torch.manual_seed(20261027)
    checkpoint = torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=False)
    net = SitePolicyValueNet(**checkpoint["architecture"]).eval()
    net.load_state_dict(checkpoint["model"], strict=True)
    roots = read(ROOTS)
    # All actual root batches, not selected by search success/failure.
    cases = {
        f"root-batch-{begin:03d}": tensors([r["instance"] for r in roots[begin : begin + 8]])
        for begin in range(0, len(roots), 8)
    }
    # Connected T(N,2) braid closures, N odd: forward-size probes, not searches.
    for n in (129, 513, 2049):
        cases[f"torus-N{n}"] = tensors([{"word": list(range(1, n)) * 2, "strands": n}])
    results = {name: measure_case(net, batch, 7) for name, batch in cases.items()}
    check_pins(pins)
    report = {
        "status": "completed",
        "inputs": pins,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "threads": 1,
            "device": "cpu",
        },
        "results": results,
        "scope": (
            "Isolated preprocessing/encoder benchmark, concurrent with unchanged frozen search. "
            "No end-to-end, training, GPU or solution-quality speedup claim. No live source patch."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    atomic(args.out / "report.json", report)
    print({name: row["speedup"] for name, row in results.items()}, flush=True)


if __name__ == "__main__":
    main()
