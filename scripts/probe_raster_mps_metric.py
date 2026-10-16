"""Protocol125: isolated CPU/MPS metric backward parity and synchronized cost."""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from diagnose_raster_task_gradients import gradient_snapshot, named_parameters
from probe_raster_cache_full_network import state_digest
from probe_raster_cache_search import load_network
from run_strand_graph_pilot import metric_loss, pair_row

VIEWS = Path("artifacts/raster-zero-cc-view-gate-v1")
VIEWS_SHA = "eaf4e2743545826ec0cf3e334b7cd38045c2b7a8cc9b75abb0bfe2545951a34c"
RUN = Path("artifacts/pretrained-raster-log-training-pair-v1")


def select_pairs(variants):
    selected = []
    for level in (1, 4, 16):
        candidates = [
            (i, v)
            for i, v in enumerate(variants)
            if v["rounds"] == level and 1 <= v["lower"] == v["upper"]
        ]
        if not candidates:
            raise ValueError("missing exact-positive size level")
        index, row = min(
            candidates,
            key=lambda item: (
                -len(item[1]["left"]["word"]) - len(item[1]["right"]["word"]),
                item[1]["parent_index"],
            ),
        )
        selected.append((index, row))
    return selected


def compare_vectors(reference, candidate):
    a, b = reference.double(), candidate.double()
    if (
        a.shape != b.shape
        or a.ndim != 1
        or not torch.isfinite(a).all()
        or not torch.isfinite(b).all()
    ):
        raise ValueError("matching finite CPU gradient vectors required")
    norm, other = float(a.norm()), float(b.norm())
    error = float((a - b).norm())
    cosine = float(torch.dot(a, b)) / (norm * other) if norm and other else None
    return dict(
        reference_norm=norm,
        candidate_norm=other,
        absolute_l2_error=error,
        relative_l2_error=error / norm if norm else None,
        cosine=cosine,
        passed=error <= 1e-5
        if norm <= 1e-6
        else error / norm <= 0.01 and cosine is not None and cosine >= 0.999,
    )


def compare(cpu, mps, a, b):
    left = torch.tensor([cpu["loss"], *cpu["prediction"]], dtype=torch.float64)
    right = torch.tensor([mps["loss"], *mps["prediction"]], dtype=torch.float64)
    scalar_ok = bool(torch.isclose(left, right, rtol=1e-3, atol=1e-4).all())
    layout = cpu["layout"]
    def clean(entries):
        return [{k: v for k, v in p.items() if k != "absent"} for p in entries]
    if clean(layout) != clean(mps["layout"]):
        raise ValueError("backend parameter layout differs")
    groups = {}
    for block in sorted({p["block"] for p in layout}):
        indices = [i for p in layout if p["block"] == block for i in range(p["start"], p["stop"])]
        groups[block] = compare_vectors(a[indices], b[indices])
    all_ = compare_vectors(a, b)
    return dict(
        scalars_close=scalar_ok,
        scalar_max_abs_error=float((left - right).abs().max()),
        gradients=all_,
        blocks=groups,
        passed=scalar_ok and all_["passed"] and all(v["passed"] for v in groups.values()),
    )


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def measure(net, objective, row, device):
    before = state_digest(net, objective)
    net.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)
    synchronize(device)
    start = time.perf_counter()
    loss, prediction = metric_loss(net, objective, [row], device)
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
        raise ValueError("nonfinite forward")
    loss.backward()
    synchronize(device)
    seconds = time.perf_counter() - start
    gradient, layout = gradient_snapshot(named_parameters(net, objective))
    if state_digest(net, objective) != before:
        raise ValueError("metric backward changed weights")
    return dict(
        device=device,
        seconds=seconds,
        loss=float(loss.detach().cpu()),
        prediction=prediction.detach().cpu().tolist(),
        layout=layout,
        state_sha256=before,
        weights_unchanged=True,
    ), gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("automatic MPS fallback must be disabled")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    if sha(VIEWS / "report.json") != VIEWS_SHA:
        raise ValueError("different view gate")
    vr = json.loads((VIEWS / "report.json").read_text())
    if (
        sha(VIEWS / "variants.json") != vr["variants_sha256"]
        or sha(VIEWS / "inputs.json") != vr["inputs_sha256"]
    ):
        raise ValueError("changed view artifacts")
    pins = dict(json.loads((VIEWS / "inputs.json").read_text())["inputs"])
    for path in (
        VIEWS / "report.json",
        VIEWS / "variants.json",
        VIEWS / "inputs.json",
        Path(__file__).resolve(),
        Path("tests/test_raster_mps_metric_probe.py"),
        Path("research/125-raster-mps-metric-training-gate.md"),
    ):
        pins[str(path)] = sha(path)
    check_pins(pins)
    selected = select_pairs(json.loads((VIEWS / "variants.json").read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json",
        dict(
            inputs=pins,
            pid=os.getpid(),
            torch_version=torch.__version__,
            threads=1,
            automatic_mps_fallback=False,
            selected_variants=[i for i, _ in selected],
            warmups=2,
            repetitions=3,
            optimizer_steps=0,
        ),
    )
    results, comparisons, error, phase = {}, {}, None, "availability"
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        for arm_index, arm in enumerate(("q254", "random")):
            base, objective_base = load_network(RUN / arm / "checkpoint.pt", arm)
            for case_index, (variant_index, variant) in enumerate(selected):
                row = pair_row(
                    variant_index,
                    "TRAIN-large-view-cost",
                    (tuple(variant["left"]["word"]), variant["left"]["strands"]),
                    (tuple(variant["right"]["word"]), variant["right"]["strands"]),
                    variant["lower"],
                    variant["upper"],
                    variant["parent_witness_semantic_length"],
                )
                case = f"{arm}-rounds{variant['rounds']}"
                phase = f"{case}-device-copy"
                nets = {
                    d: (
                        copy.deepcopy(base).to(d).train(),
                        copy.deepcopy(objective_base).to(d).train(),
                    )
                    for d in ("cpu", "mps")
                }
                expected_state = state_digest(base, objective_base)
                for device, (net, objective) in nets.items():
                    if state_digest(net, objective) != expected_state:
                        raise ValueError("backend copy differs from checkpoint state")
                    phase = f"{case}-{device}-warmup"
                    for _ in range(2):
                        measure(net, objective, row, device)
                for repetition in range(3):
                    order = (
                        ("cpu", "mps")
                        if (arm_index + case_index + repetition) % 2 == 0
                        else ("mps", "cpu")
                    )
                    measured, gradients = {}, {}
                    for device in order:
                        phase = f"{case}-rep{repetition}-{device}"
                        measured[device], gradients[device] = measure(*nets[device], row, device)
                        vector_path = args.out / f"{phase}.pt"
                        with vector_path.open("xb") as stream:
                            torch.save(gradients[device], stream)
                        results[phase] = {
                            **measured[device],
                            "gradient_sha256": sha(vector_path),
                            "variant_index": variant_index,
                            "repetition": repetition,
                        }
                    key = f"{case}-rep{repetition}"
                    comparisons[key] = compare(
                        measured["cpu"], measured["mps"], gradients["cpu"], gradients["mps"]
                    )
                    comparisons[key]["order"] = order
                    print(
                        json.dumps(
                            dict(
                                case=key,
                                passed=comparisons[key]["passed"],
                                cpu_seconds=measured["cpu"]["seconds"],
                                mps_seconds=measured["mps"]["seconds"],
                            )
                        ),
                        flush=True,
                    )
                del nets
        phase = "final-input-check"
        check_pins(pins)
    except Exception as exc:
        error = dict(phase=phase, type=type(exc).__name__, message=str(exc))
    write_new(args.out / "measurements.json", results)
    passed = (
        error is None and len(comparisons) == 18 and all(c["passed"] for c in comparisons.values())
    )
    write_new(
        args.out / "report.json",
        dict(
            status="passed" if passed else "failed",
            protocol=125,
            inputs_sha256=sha(args.out / "inputs.json"),
            measurements_sha256=sha(args.out / "measurements.json"),
            comparisons=comparisons,
            measured_backward_calls=len(results),
            error=error,
            optimizer_steps=0,
            frozen_search_changed=False,
            scope="Selected TRAIN metric forward/backward only; "
            "not full-training or MCTS throughput.",
        ),
    )
    print(
        json.dumps(
            dict(
                status="passed" if passed else "failed",
                error=error,
                report_sha256=sha(args.out / "report.json"),
            )
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
