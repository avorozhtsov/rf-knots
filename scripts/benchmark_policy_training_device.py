"""TRAIN-only forward/backward device probe; no optimizer step or saved weights.

Compares original/bulk graph preprocessing on CPU/MPS using identical audited
set-valued policy targets. This does not time metric loss, relative preferences,
optimizer updates, input replay, MCTS or the entire training loop.
"""

import argparse
import gc
import os
import random
import statistics
import time
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins
from benchmark_crossing_preprocessing import bulk_crossing_adjacency, use_adjacency
from build_mcts_training_supervision import atomic, sha
from mcts_joint_training import load_policy_targets
from mcts_training_supervision import set_valued_policy_loss
from run_joint_policy_metric_evaluation import read

from rf_knots import mcts_loop as M
from rf_knots import strand_graph_encoder as graph
from rf_knots.site_head import SiteHead

DATA = Path("artifacts/diverse-mcts-supervision-diamonds-v1")
RUN = Path("artifacts/mcts-policy-metric-joint-v1")
CONFIG = {"seed": 20261028, "batch_sizes": [16, 64, 128], "warmup": 1, "repeats": 3}


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def step(net, selected, device):
    net.zero_grad(set_to_none=True)
    batch = M.build_batch([r[0] for r in selected], device, budget_dim=net.budget_dim)
    logits, _, effective = net(**batch)
    if not torch.equal(effective, batch["site_mask"]):
        raise ValueError("device masked a certified action")
    logp = SiteHead.masked_log_policy(logits, effective)
    positives = torch.zeros_like(logp, dtype=torch.bool)
    for row, (_, indices) in enumerate(selected):
        positives[row, list(indices)] = True
    loss = set_valued_policy_loss(logp, positives)
    loss.backward()
    return loss, logits[effective]


def capture(net, loss, logits):
    values = {"loss": loss.detach().cpu(), "logits": logits.detach().cpu()}
    values.update(
        {
            f"grad/{n}": p.grad.detach().cpu().clone()
            for n, p in net.named_parameters()
            if p.grad is not None
        }
    )
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError("nonfinite outputs or gradients")
    return values


def difference(expected, actual, *, exact):
    if expected.keys() != actual.keys():
        raise ValueError("gradient/output coverage differs")
    maximum = 0.0
    for name, before in expected.items():
        after = actual[name]
        torch.testing.assert_close(
            before, after, rtol=0 if exact else 1e-3, atol=0 if exact else 1e-4
        )
        if before.numel():
            maximum = max(maximum, float((before - after).abs().max()))
    return {"passed": True, "bit_equal_required": exact, "max_absolute_difference": maximum}


def benchmark_arm(checkpoint, selected, device, helper):
    net = M.SitePolicyValueNet(**checkpoint["architecture"]).to(device).train()
    net.load_state_dict(checkpoint["model"], strict=True)
    durations = []
    with use_adjacency(helper):
        for _ in range(CONFIG["warmup"]):
            loss, logits = step(net, selected, device)
        reference = capture(net, loss, logits)
        for _ in range(CONFIG["repeats"]):
            synchronize(device)
            started = time.perf_counter()
            loss, logits = step(net, selected, device)
            synchronize(device)
            durations.append(time.perf_counter() - started)
            difference(reference, capture(net, loss, logits), exact=False)
    for name, value in net.state_dict().items():
        if not torch.equal(value.detach().cpu(), checkpoint["model"][name]):
            raise ValueError("benchmark changed model weights/buffers")
    result = {
        "seconds": durations,
        "median_seconds": statistics.median(durations),
        "policy_states_per_second": len(selected) / statistics.median(durations),
        "loss": float(reference["loss"]),
        "weights_unchanged": True,
    }
    del net, loss, logits
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite device probe")
    model = read(RUN / "manifest.json")
    supervision = read(DATA / "manifest.json")
    audit = read(DATA / "certificate-audit.json")
    if (
        model["status"] != "completed"
        or supervision["status"] != "completed"
        or audit["status"] != "passed"
    ):
        raise ValueError("require completed model, TRAIN supervision and certificate audit")
    if model["checkpoint_sha256"] != sha(RUN / "checkpoint.pt"):
        raise ValueError("model checksum changed")
    paths = [
        Path(__file__).resolve(),
        RUN / "manifest.json",
        RUN / "checkpoint.pt",
        DATA / "manifest.json",
        DATA / "certificate-audit.json",
        Path("scripts/benchmark_crossing_preprocessing.py"),
        Path("scripts/mcts_joint_training.py"),
        Path("scripts/mcts_training_supervision.py"),
        *Path("src/rf_knots").glob("*.py"),
        *(DATA / row["file"] for row in supervision["conditions"]),
    ]
    pins = {**audit["inputs"], **{str(p): sha(p) for p in paths}}
    check_pins(pins)
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    checkpoint = torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=False)
    groups, _ = load_policy_targets(DATA)
    rng = random.Random(CONFIG["seed"])
    selection = [
        rng.choice(groups[c]) for c in rng.sample(range(len(groups)), max(CONFIG["batch_sizes"]))
    ]
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    baseline_helper = graph.crossing_adjacency
    results = {}
    # Failure is recorded without fallback and without touching the active experiment.
    error = None
    try:
        for size in CONFIG["batch_sizes"]:
            selected = selection[:size]
            cpu_reference = None
            for device in devices:
                original = None
                for name, helper in (
                    ("original", baseline_helper),
                    ("bulk", bulk_crossing_adjacency),
                ):
                    result, values = benchmark_arm(checkpoint, selected, device, helper)
                    if cpu_reference is None:
                        cpu_reference = values
                    result["cpu_numerical_gate"] = difference(
                        cpu_reference, values, exact=device == "cpu"
                    )
                    if original is None:
                        original = values
                    result["same_device_preprocessing_gate"] = difference(
                        original, values, exact=True
                    )
                    results[f"batch{size}/{device}/{name}"] = result
                    print(
                        {
                            "completed": f"batch{size}/{device}/{name}",
                            "seconds": result["median_seconds"],
                        },
                        flush=True,
                    )
    except (RuntimeError, ValueError, AssertionError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    check_pins(pins)
    report = {
        "status": "completed" if error is None else "failed_engineering_gate",
        "error": error,
        "inputs": pins,
        "config": CONFIG,
        "results": results,
        "runtime": {
            "torch": str(torch.__version__),
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
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
        "scope": (
            "TRAIN-only set-valued policy forward/backward. No optimizer steps or saved models. "
            "Excludes metric/relative losses, optimizer, data replay and MCTS; "
            "not full-loop throughput."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    atomic(args.out / "report.json", report)
    print(report["status"], error, flush=True)


if __name__ == "__main__":
    main()
