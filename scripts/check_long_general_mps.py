"""Protocol152 private real-weight MPS parity/resume; never promotes weights."""

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from check_raster_mps_joint import cpu_tree, save, tree_check
from graph_metric_sampling import load_graph_sampler
from long_general_training_data import CONFIG, load_long_source, materialize, paired_draw
from mcts_joint_training import load_metric_samplers, load_policy_targets
from pretrained_raster_policy import reference_preferences
from probe_raster_cache_full_network import state_digest
from probe_raster_mps_metric import compare, measure
from raster_log_training import (
    CORPUS,
    GRAPH,
    SUPERVISION,
    digest,
    inputs_and_sources,
    matched_initializations,
)
from train_long_general_pair import checked_loss_contract, optimize
from train_mcts_policy_metric import finite_optimizer, finite_state

from rf_knots.embedding_metric import MetricEmbeddingObjective

SCHEMA = "rf-knots-long-general-mps-gate-v1"


def select_cases(records):
    chosen = []
    for band in range(3):
        for label in range(3):
            eligible = [
                r
                for r in records
                if r["source"]["length_band"] == band
                and r["lower"] == r["upper"] == label
                and r["split"] == "train"
            ]
            if not eligible:
                raise ValueError("missing TRAIN class/band for MPS gate")
            row = min(
                eligible,
                key=lambda r: (
                    -len(r["states"][0]["word"]),
                    -r["states"][0]["strands"],
                    r["index"],
                ),
            )
            chosen.append((f"band{band}-exact{label}", row))
    eligible = [r for r in records if r["split"] == "train"]
    row = min(
        eligible,
        key=lambda r: (-r["states"][0]["strands"], -len(r["states"][0]["word"]), r["index"]),
    )
    return chosen + [("maximum-strands", row)]


def snapshot(net, objective, optimizer, rng, long_rng, trace):
    torch.mps.synchronize()
    return cpu_tree(
        dict(
            model=net.state_dict(),
            objective=objective.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng.getstate(),
            long_rng=long_rng.getstate(),
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state(),
            trace=trace,
        )
    )


def advance_draws(arm, rng, long_rng, data, preferences, trace, count=16):
    original, suffix, graph, long_source, positives, _ = data
    if count != 16:
        raise ValueError("gate requires exactly one16-draw relative period")
    shared, selected = trace
    for _ in range(count):
        draw = paired_draw(
            arm, rng, long_rng, original, suffix, graph, long_source, positives, preferences
        )
        shared = digest([shared, draw.shared_digest])
        selected = digest([selected, draw.selected_digest])
    return draw, (shared, selected)


def optimizer_gate(arm, base, data, preferences, out):
    def fresh():
        net = copy.deepcopy(base).to("mps")
        objective = MetricEmbeddingObjective().to("mps")
        optimizer = torch.optim.AdamW(
            [*net.parameters(), *objective.parameters()],
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        return net, objective, optimizer

    def step(net, objective, optimizer, draw):
        started = time.perf_counter()
        losses = optimize(net, objective, optimizer, draw.metrics, draw.positives, draw.relative)
        torch.mps.synchronize()
        finite_state(net, objective)
        finite_optimizer(optimizer)
        return dict(
            losses=losses,
            seconds=time.perf_counter() - started,
            relative_group=draw.group_index,
            shared_digest=draw.shared_digest,
            selected_digest=draw.selected_digest,
        )

    net, objective, optimizer = fresh()
    initial = state_digest(net, objective)
    rng, long_rng = random.Random(CONFIG["seed"]), random.Random(CONFIG["long_seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.mps.manual_seed(CONFIG["seed"])
    trace = (digest([]), digest([]))
    first_draw, trace = advance_draws(arm, rng, long_rng, data, preferences, trace)
    first = step(net, objective, optimizer, first_draw)
    middle = snapshot(net, objective, optimizer, rng, long_rng, trace)
    checkpoints = {"engineering-resume.pt": save(out / "engineering-resume.pt", middle)}
    second_draw, trace = advance_draws(arm, rng, long_rng, data, preferences, trace)
    second = step(net, objective, optimizer, second_draw)
    control = snapshot(net, objective, optimizer, rng, long_rng, trace)
    checkpoints["engineering-control-final.pt"] = save(
        out / "engineering-control-final.pt", control
    )
    replay_net, replay_obj, replay_opt = fresh()
    restored = torch.load(out / "engineering-resume.pt", map_location="cpu", weights_only=False)
    replay_net.load_state_dict(restored["model"], strict=True)
    replay_obj.load_state_dict(restored["objective"], strict=True)
    replay_opt.load_state_dict(restored["optimizer"])
    rng.setstate(restored["rng"])
    long_rng.setstate(restored["long_rng"])
    torch.set_rng_state(restored["torch_rng"])
    torch.mps.set_rng_state(restored["mps_rng"])
    trace = restored["trace"]
    tree_check(
        middle, snapshot(replay_net, replay_obj, replay_opt, rng, long_rng, trace), exact=True
    )
    draw, trace = advance_draws(arm, rng, long_rng, data, preferences, trace)
    if (draw.shared_digest, draw.selected_digest, draw.group_index) != (
        second_draw.shared_digest,
        second_draw.selected_digest,
        second_draw.group_index,
    ):
        raise ValueError("resumed second batch differs")
    replay = step(replay_net, replay_obj, replay_opt, draw)
    resumed = snapshot(replay_net, replay_obj, replay_opt, rng, long_rng, trace)
    checkpoints["engineering-resumed-final.pt"] = save(
        out / "engineering-resumed-final.pt", resumed
    )
    tree_check(control, resumed)
    bitwise = True
    try:
        tree_check(control, resumed, exact=True)
    except (AssertionError, ValueError):
        bitwise = False
    steps = sorted({int(s["step"]) for s in resumed["optimizer"]["state"].values()})
    if steps != [2] or state_digest(replay_net, replay_obj) == initial:
        raise ValueError("private AdamW did not advance")
    return dict(
        passed=True,
        immediate_restore_bitwise=True,
        final_close=True,
        final_bitwise=bitwise,
        optimizer_steps=steps,
        physical_optimizer_calls=3,
        consumed_data_draws=32,
        shared_draw_digest=trace[0],
        selected_draw_digest=trace[1],
        checkpoints=checkpoints,
        batch_records=[[asdict(r) for r in d.metrics] for d in (first_draw, second_draw)],
        losses=[first, second, replay],
        initial_state_sha256=initial,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-data", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("gate output must be new")
    if (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"
        or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("require MPS, with no automatic fallback")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    checked_loss_contract()
    long_source, _, pins = load_long_source(args.long_data)
    original_pins, sources = inputs_and_sources()
    merge_pins(pins, original_pins)
    merge_pins(pins, sources)
    for name in (
        "scripts/check_long_general_mps.py",
        "tests/test_long_general_mps_gate.py",
        "scripts/train_long_general_pair.py",
        "tests/test_train_long_general_pair.py",
        "scripts/long_general_training_data.py",
        "tests/test_long_general_training_data.py",
        "scripts/train_raster_resource_pair.py",
        "scripts/raster_resource_units.py",
        "scripts/check_raster_mps_joint.py",
        "scripts/probe_raster_mps_metric.py",
        "scripts/diagnose_raster_task_gradients.py",
        "scripts/probe_raster_cache_full_network.py",
        "research/149-long-general-mixture-training-protocol.md",
        "research/151-long-mixture-evaluation-contract.md",
        "research/152-long-mixture-mps-execution-gate.md",
    ):
        path = Path(name).resolve()
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    records = json.loads((args.long_data / "corpus.json").read_text())["records"]["train"]
    selection = select_cases(records)
    base, states, architecture, _ = matched_initializations(pins)
    base.load_state_dict(states["q254"], strict=True)
    objective = MetricEmbeddingObjective()
    initial = state_digest(base, objective)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    data = (
        original,
        suffix,
        load_graph_sampler(GRAPH),
        long_source,
        *load_policy_targets(SUPERVISION),
    )
    args.out.mkdir(exist_ok=False)
    atomic(args.out / "selection.json", [dict(name=n, record=r) for n, r in selection])
    atomic(
        args.out / "inputs.json",
        dict(
            inputs=pins,
            config=CONFIG,
            architecture=architecture,
            initial_state_sha256=initial,
            pid=os.getpid(),
            torch_version=torch.__version__,
        ),
    )
    cases, arms, artifacts = {}, {}, {}
    try:
        mps, mps_obj = copy.deepcopy(base).to("mps"), copy.deepcopy(objective).to("mps")
        for name, record in selection:
            row = materialize(record)
            cpu, cpu_vector = measure(base, objective, row, "cpu")
            gpu, gpu_vector = measure(mps, mps_obj, row, "mps")
            result = compare(cpu, gpu, cpu_vector, gpu_vector)
            vector_path = args.out / f"{name}-gradients.pt"
            artifacts[vector_path.name] = save(vector_path, dict(cpu=cpu_vector, mps=gpu_vector))
            cases[name] = dict(
                passed=result["passed"], index=record["index"], cpu=cpu, mps=gpu, comparison=result
            )
            case_path = args.out / f"{name}.json"
            atomic(case_path, cases[name])
            artifacts[case_path.name] = sha(case_path)
            if not result["passed"]:
                raise ValueError(f"CPU/MPS metric parity failed: {name}")
            atomic(
                args.out / "progress.json",
                dict(phase="metric_parity", cases=list(cases), pid=os.getpid()),
            )
            print(json.dumps(dict(metric_case=name, passed=True)), flush=True)
        preferences = reference_preferences(mps, data[-1], microbatch=1)
        if state_digest(base, objective) != initial or state_digest(mps, mps_obj) != initial:
            raise ValueError("pre-optimizer parity/reference changed initial state")
        for arm in CONFIG["arms"]:
            directory = args.out / arm
            directory.mkdir()
            arms[arm] = optimizer_gate(arm, base, data, preferences, directory)
            for name, checksum in arms[arm]["checkpoints"].items():
                artifacts[str(Path(arm) / name)] = checksum
            atomic(directory / "report.json", arms[arm])
            artifacts[str(Path(arm) / "report.json")] = sha(directory / "report.json")
            print(
                json.dumps(dict(arm=arm, passed=True, final_bitwise=arms[arm]["final_bitwise"])),
                flush=True,
            )
        if arms["control"]["shared_draw_digest"] != arms["long"]["shared_draw_digest"]:
            raise ValueError("engineering arms consumed different data streams")
        if state_digest(base, objective) != initial:
            raise ValueError("private optimization changed shared source weights")
        for name in ("selection.json", "inputs.json"):
            artifacts[name] = sha(args.out / name)
        check_pins(pins)
        atomic(
            args.out / "report.json",
            dict(
                schema=SCHEMA,
                status="passed",
                inputs=pins,
                artifacts=artifacts,
                config=CONFIG,
                torch_version=torch.__version__,
                device="mps",
                threads=1,
                automatic_mps_fallback=False,
                architecture=architecture,
                initial_state_sha256=initial,
                cases=cases,
                arms=arms,
                physical_optimizer_calls=6,
                published_training_updates=0,
                scope="Private bounded backend readiness; not a promotion or learning result",
            ),
        )
    except Exception as error:
        atomic(
            args.out / "failure.json",
            dict(
                status="failed",
                error=repr(error),
                completed_cases=cases,
                completed_arms=list(arms),
                artifacts=artifacts,
            ),
        )
        raise


if __name__ == "__main__":
    main()
