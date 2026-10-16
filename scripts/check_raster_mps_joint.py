"""Protocol127: private full-task MPS gradients and saved-AdamW resume gate."""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from diagnose_raster_task_gradients import gradient_snapshot, named_parameters
from pretrained_raster_policy import policy_backward
from probe_raster_cache_full_network import state_digest
from probe_raster_cache_search import load_network
from probe_raster_mps_metric import compare_vectors
from raster_log_training import CONFIG, digest
from run_strand_graph_pilot import metric_loss
from trajectory_log_preference import relative_backward

from rf_knots import mcts_loop as M
from rf_knots.embedding_augmentation import MaterializedSyntheticPair, SyntheticPairSpec
from rf_knots.embedding_metric import witness_priority_weight

BASE = Path("artifacts/raster-task-gradient-diagnostic-v1")
BASE_SHA = "f4911c95002d5db7cc076e07b633b7324bf9f9db06142a77592e391da494fdb2"
RUN = Path("artifacts/pretrained-raster-log-training-pair-v1")
TASKS = ["metric", "positive", *[f"relative{i}" for i in range(4)]]


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def tree_check(a, b, *, exact=False, path=""):
    """Fail closed, with exact optimizer counters even in approximate mode."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"tensor identity differs: {path}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"nonfinite state: {path}")
        if exact or not a.is_floating_point() or path.endswith("/step"):
            if not torch.equal(a, b):
                raise ValueError(f"exact tensor differs: {path}")
        else:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=path)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f"dictionary keys differ: {path}")
        for k in a:
            tree_check(a[k], b[k], exact=exact, path=f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError(f"sequence differs: {path}")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            tree_check(x, y, exact=exact, path=f"{path}/{i}")
    elif type(a) is not type(b) or a != b or isinstance(a, float) and not math.isfinite(a):
        raise ValueError(f"scalar differs: {path}")


def vectors_check(reference, candidate, cpu_layout, mps_layout):
    def clean(rows):
        return [{k: v for k, v in r.items() if k != "absent"} for r in rows]

    if clean(cpu_layout) != clean(mps_layout):
        raise ValueError("CPU/MPS parameter layouts differ")
    covered = [i for p in cpu_layout for i in range(p["start"], p["stop"])]
    if covered != list(range(reference.numel())):
        raise ValueError("layout does not partition vector")
    blocks = {}
    for block in sorted({p["block"] for p in cpu_layout}):
        indices = [
            i for p in cpu_layout if p["block"] == block for i in range(p["start"], p["stop"])
        ]
        blocks[block] = compare_vectors(reference[indices], candidate[indices])
    whole = compare_vectors(reference, candidate)
    return dict(
        whole=whole,
        blocks=blocks,
        passed=whole["passed"] and all(v["passed"] for v in blocks.values()),
    )


def request(record):
    return (
        M.Instance(**{**record["instance"], "word": tuple(record["instance"]["word"])}),
        M.SearchState(**{**record["state"], "word": tuple(record["state"]["word"])}),
        tuple(record["actions"]),
    )


def data():
    selected = json.loads((BASE / "selection.json").read_text())
    rows = [
        MaterializedSyntheticPair(
            **{
                **r,
                "spec": SyntheticPairSpec(**r["spec"]),
                "left_word": tuple(r["left_word"]),
                "right_word": tuple(r["right_word"]),
            }
        )
        for r in selected["metrics"]
    ]
    if digest([asdict(r) for r in rows]) != digest(selected["metrics"]):
        raise ValueError("metric reconstruction differs")
    positives = [(request(r["request"]), tuple(r["good"])) for r in selected["positives"]]
    groups = [
        [(request(r["request"]), r["index"], r["coefficient"]) for r in group]
        for group in selected["full_relative_groups"]
    ]
    if len(rows) != 32 or len(positives) != 8 or [len(g) for g in groups] != [334, 52, 74, 54]:
        raise ValueError("draw/group coverage differs")
    return rows, positives, groups


def backward(task, net, objective, rows, positives, preferences):
    if task == "metric":
        if (
            objective.cc_rank_weight != 0
            or not objective.witness_priority
            or any(r.metric != "cc" for r in rows)
        ):
            raise ValueError("requires97 CC-only weighted rank-free loss")
        weights = [
            witness_priority_weight("cc", r.spec.distance_upper, r.witness_semantic_length)
            for r in rows
        ]
        value = 0.0
        for row, weight in zip(rows, weights, strict=True):
            loss, _ = metric_loss(net, objective, [row], "mps")
            scaled = loss * (weight / sum(weights))
            if not torch.isfinite(scaled):
                raise ValueError("nonfinite metric loss")
            scaled.backward()
            value += float(scaled.detach())
        result = dict(metric_loss=value)
    elif task == "positive":
        result = policy_backward(
            net, positives, None, microbatch=1, positive_weight=CONFIG["positive_weight"]
        )
    else:
        result = relative_backward(
            net,
            preferences[int(task[-1])],
            microbatch=1,
            weight=CONFIG["relative_weight"] * CONFIG["relative_every"],
        )
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite task scalar")
    return result


def snapshot(net, objective, optimizer):
    torch.mps.synchronize()
    return cpu_tree(
        dict(
            model=net.state_dict(),
            objective=objective.state_dict(),
            optimizer=optimizer.state_dict(),
            python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state(),
        )
    )


def save(path, payload):
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return sha(path)


def optimizer_test(arm, rows, positives, preferences, out):
    original = torch.load(RUN / arm / "checkpoint.pt", map_location="cpu", weights_only=False)

    def fresh():
        net, obj = load_network(RUN / arm / "checkpoint.pt", arm)
        net.to("mps").train()
        obj.to("mps").train()
        opt = torch.optim.AdamW([*net.parameters(), *obj.parameters()], lr=CONFIG["lr"])
        opt.load_state_dict(original["optimizer"])
        return net, obj, opt

    def step(net, obj, opt, group):
        opt.zero_grad(set_to_none=True)
        losses = {}
        for task in ("metric", "positive", f"relative{group}"):
            losses.update(backward(task, net, obj, rows, positives, preferences))
        norm = torch.nn.utils.clip_grad_norm_(
            [*net.parameters(), *obj.parameters()], CONFIG["grad_clip"], error_if_nonfinite=True
        )
        opt.step()
        torch.mps.synchronize()
        if not all(torch.isfinite(p).all() for p in [*net.parameters(), *obj.parameters()]):
            raise ValueError("nonfinite optimized weights")
        return {**losses, "gradient_norm_before_clip": float(norm)}

    net, obj, opt = fresh()
    initial = state_digest(net, obj)
    if {int(s["step"]) for s in opt.state.values()} != {2048}:
        raise ValueError("optimizer not at2048")
    first = step(net, obj, opt, 1)
    middle = snapshot(net, obj, opt)
    hashes = {"engineering-resume.pt": save(out / "engineering-resume.pt", middle)}
    second = step(net, obj, opt, 3)
    control = snapshot(net, obj, opt)
    hashes["engineering-control-final.pt"] = save(out / "engineering-control-final.pt", control)
    net2, obj2, opt2 = fresh()
    restored = torch.load(out / "engineering-resume.pt", map_location="cpu", weights_only=False)
    net2.load_state_dict(restored["model"], strict=True)
    obj2.load_state_dict(restored["objective"], strict=True)
    opt2.load_state_dict(restored["optimizer"])
    random.setstate(restored["python_rng"])
    torch.set_rng_state(restored["torch_rng"])
    torch.mps.set_rng_state(restored["mps_rng"])
    tree_check(middle, snapshot(net2, obj2, opt2), exact=True)
    replay = step(net2, obj2, opt2, 3)
    resumed = snapshot(net2, obj2, opt2)
    hashes["engineering-resumed-final.pt"] = save(out / "engineering-resumed-final.pt", resumed)
    tree_check(control, resumed)
    bitwise = True
    try:
        tree_check(control, resumed, exact=True)
    except (ValueError, AssertionError):
        bitwise = False
    steps = sorted({int(s["step"]) for s in resumed["optimizer"]["state"].values()})
    if steps != [2050] or initial == state_digest(net2, obj2):
        raise ValueError("optimizer failed to advance private weights/counters")
    return dict(
        passed=True,
        immediate_restore_bitwise=True,
        final_close=True,
        final_bitwise=bitwise,
        steps=steps,
        physical_optimizer_calls=3,
        checkpoints=hashes,
        losses=[first, second, replay],
    )


def provenance():
    if sha(BASE / "report.json") != BASE_SHA:
        raise ValueError("changed118 report")
    base = json.loads((BASE / "report.json").read_text())
    if (
        sha(BASE / "inputs.json") != base["inputs_sha256"]
        or sha(BASE / "selection.json") != base["selection_sha256"]
    ):
        raise ValueError("changed118 input/selection")
    pins = dict(json.loads((BASE / "inputs.json").read_text())["inputs"])
    for arm in ("q254", "random"):
        report = BASE / f"{arm}-final" / "report.json"
        refs = BASE / f"{arm}-references.json"
        if sha(report) != base["cases"][f"{arm}-final"] or sha(refs) != base["references"][arm]:
            raise ValueError("changed118 case/reference")
        case = json.loads(report.read_text())
        vectors = report.parent / "gradients.pt"
        if sha(vectors) != case["vectors_sha256"]:
            raise ValueError("changed118 vectors")
        for path in (report, refs, vectors):
            pins[str(path.resolve())] = sha(path)
    for path in (
        BASE / "report.json",
        BASE / "selection.json",
        BASE / "inputs.json",
        Path(__file__),
        Path("scripts/probe_raster_mps_metric.py"),
        Path("tests/test_raster_mps_joint.py"),
        Path("research/127-raster-mps-joint-resume-gate.md"),
    ):
        pins[str(path.resolve())] = sha(path)
    check_pins(pins)
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("automatic MPS fallback must be disabled")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    pins = provenance()
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json",
        dict(
            inputs=pins,
            pid=os.getpid(),
            protocol=127,
            torch_version=torch.__version__,
            threads=1,
            automatic_mps_fallback=False,
        ),
    )
    results, error, phase = {}, None, "availability"
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        rows, positives, groups = data()
        for arm in ("q254", "random"):
            out = args.out / arm
            out.mkdir()
            refs = json.loads((BASE / f"{arm}-references.json").read_text())
            if digest(refs["references"]) != refs["sha256"]:
                raise ValueError("reference logical digest differs")
            preferences = [
                (g, torch.tensor(r)) for g, r in zip(groups, refs["references"], strict=True)
            ]
            cpu = json.loads((BASE / f"{arm}-final/report.json").read_text())
            vectors = torch.load(BASE / f"{arm}-final/gradients.pt", weights_only=True)
            if list(vectors) != TASKS:
                raise ValueError("CPU task coverage differs")
            net, obj = load_network(RUN / arm / "checkpoint.pt", arm)
            net.to("mps").train()
            obj.to("mps").train()
            before = state_digest(net, obj)
            if before != cpu["state_sha256"]:
                raise ValueError("CPU/MPS initial state differs")
            result = results[arm] = dict(components={})
            saved = {}
            for task in [*TASKS, "mixed"]:
                phase = f"{arm}/{task}"
                net.zero_grad(set_to_none=True)
                obj.zero_grad(set_to_none=True)
                start = time.perf_counter()
                losses = {}
                for part in ["metric", "positive", "relative0"] if task == "mixed" else [task]:
                    losses.update(backward(part, net, obj, rows, positives, preferences))
                torch.mps.synchronize()
                seconds = time.perf_counter() - start
                saved[task], layout = gradient_snapshot(named_parameters(net, obj))
                expected = (
                    sum(vectors[k].double() for k in ("metric", "positive", "relative0"))
                    if task == "mixed"
                    else vectors[task]
                )
                checked = vectors_check(
                    expected,
                    saved[task],
                    cpu["layout"]["metric" if task == "mixed" else task],
                    layout,
                )
                if task != "mixed":
                    old = cpu["unweighted_losses"][task]
                    if old.keys() != losses.keys():
                        raise ValueError("scalar coverage differs")
                    checked["scalars_close"] = all(
                        math.isclose(old[k], losses[k], rel_tol=1e-3, abs_tol=1e-4) for k in old
                    )
                    checked["passed"] &= checked["scalars_close"]
                if state_digest(net, obj) != before:
                    raise ValueError("gradient-only test changed state")
                result["components"][task] = dict(**checked, losses=losses, seconds=seconds)
                print(
                    json.dumps(dict(phase=phase, passed=checked["passed"], seconds=seconds)),
                    flush=True,
                )
            result["vectors_sha256"] = save(out / "gradients.pt", saved)
            result["weights_unchanged_during_gradient_checks"] = True
            if all(v["passed"] for v in result["components"].values()):
                phase = f"{arm}/optimizer-resume"
                result["optimizer_resume"] = optimizer_test(arm, rows, positives, preferences, out)
            else:
                result["optimizer_resume"] = dict(
                    passed=False, skipped="gradient comparison failed"
                )
            print(
                json.dumps(
                    dict(phase=phase, optimizer_resume=result["optimizer_resume"]["passed"])
                ),
                flush=True,
            )
    except Exception as exc:
        error = dict(phase=phase, type=type(exc).__name__, message=str(exc))
    try:
        check_pins(pins)
    except Exception as exc:
        error = dict(
            phase="final-provenance", type=type(exc).__name__, message=str(exc), prior=error
        )
    passed = (
        error is None
        and len(results) == 2
        and all(r.get("optimizer_resume", {}).get("passed", False) for r in results.values())
    )
    report = dict(
        status="passed" if passed else "failed",
        protocol=127,
        cases=results,
        error=error,
        inputs_sha256=sha(args.out / "inputs.json"),
        engineering_only=True,
        scientific_training=False,
        test_scoring=False,
    )
    write_new(args.out / "report.json", report)
    print(json.dumps(dict(status=report["status"], error=error)), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
