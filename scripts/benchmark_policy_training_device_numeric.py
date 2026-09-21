"""Versioned numeric MPS probe after independently demonstrated backward variability.

Keeps the failed bit-exact probe unchanged. CPU remains bit-exact. MPS must
match integer adjacency exactly and pass the pre-existing CPU/MPS numerical
tolerance (rtol=1e-3, atol=1e-4). No optimizer or saved weights.
"""

import argparse
import random
from pathlib import Path

import benchmark_policy_training_device as probe
import torch

REPEATABILITY = Path("artifacts/mcts-policy-training-mps-repeatability-v1/report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not torch.backends.mps.is_available():
        raise ValueError("require MPS and a new output")
    evidence = probe.read(REPEATABILITY)
    if evidence["status"] != "completed" or not evidence["adjacency_bit_equal_on_mps"]:
        raise ValueError("missing repeatability evidence")
    pins = {
        **evidence["inputs"],
        str(REPEATABILITY): probe.sha(REPEATABILITY),
        str(Path(__file__).resolve()): probe.sha(Path(__file__).resolve()),
    }
    probe.check_pins(pins)
    torch.set_num_threads(1)
    torch.manual_seed(probe.CONFIG["seed"])
    groups, _ = probe.load_policy_targets(probe.DATA)
    rng = random.Random(probe.CONFIG["seed"])
    selection = [
        rng.choice(groups[c])
        for c in rng.sample(range(len(groups)), max(probe.CONFIG["batch_sizes"]))
    ]
    checkpoint = torch.load(probe.RUN / "checkpoint.pt", map_location="cpu", weights_only=False)
    original_helper = probe.graph.crossing_adjacency
    results, error = {}, None
    try:
        for size in probe.CONFIG["batch_sizes"]:
            selected = selection[:size]
            cpu = None
            for device in ("cpu", "mps"):
                batch = probe.M.build_batch(
                    [r[0] for r in selected],
                    device,
                    budget_dim=checkpoint["architecture"]["budget_dim"],
                )
                if not torch.equal(
                    original_helper(batch["word"], batch["lengths"]),
                    probe.bulk_crossing_adjacency(batch["word"], batch["lengths"]),
                ):
                    raise ValueError("integer adjacency differs")
                original = None
                for name, helper in (
                    ("original", original_helper),
                    ("bulk", probe.bulk_crossing_adjacency),
                ):
                    result, values = probe.benchmark_arm(checkpoint, selected, device, helper)
                    if cpu is None:
                        cpu = values
                    if original is None:
                        original = values
                    result["cpu_numerical_gate"] = probe.difference(
                        cpu, values, exact=device == "cpu"
                    )
                    result["same_device_gate"] = probe.difference(
                        original, values, exact=device == "cpu"
                    )
                    result["same_device_bit_equal_observed"] = all(
                        torch.equal(original[k], values[k]) for k in original
                    )
                    result["integer_adjacency_bit_equal"] = True
                    results[f"batch{size}/{device}/{name}"] = result
                    print(
                        {
                            "completed": f"batch{size}/{device}/{name}",
                            "seconds": result["median_seconds"],
                        },
                        flush=True,
                    )
    except (RuntimeError, ValueError, AssertionError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    probe.check_pins(pins)
    report = {
        "status": "completed" if error is None else "failed_engineering_gate",
        "inputs": pins,
        "config": probe.CONFIG,
        "error": error,
        "results": results,
        "runtime": {
            "torch": str(torch.__version__),
            "mps_fallback_env": probe.os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "selection": [
            {
                "instance": r[0][0].uid,
                "length": len(r[0][1].word),
                "strands": r[0][1].strands,
                "actions": len(r[0][2]),
            }
            for r in selection
        ],
        "amendment": (
            "CPU exactness unchanged. MPS numerical gate follows independently observed "
            "original-helper backward variability; prior failed exact probe retained."
        ),
        "scope": (
            "TRAIN-only policy forward/backward without optimizer updates or saved weights; "
            "not complete training/MCTS/GPU-cloud throughput."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    probe.atomic(args.out / "report.json", report)
    print(report["status"], error, flush=True)


if __name__ == "__main__":
    main()
