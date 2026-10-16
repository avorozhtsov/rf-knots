"""Protocol149 matched MPS training; requires completed147 and150 consumers."""

import argparse
import fcntl
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_long_mixture_short_u_test import CONFIG as SHORT_CONFIG
from build_long_mixture_short_u_test import LIFT as SHORT_LIFT
from build_mcts_training_supervision import atomic, sha
from evaluate_resource_search import CONFIG as OLD_SEARCH_CONFIG
from graph_metric_sampling import load_graph_sampler
from long_general_training_data import CONFIG, load_long_source, paired_draw
from mcts_joint_training import load_metric_samplers, load_policy_targets
from pretrained_raster_policy import reference_preferences
from probe_raster_cache_full_network import state_digest
from raster_log_training import (
    CORPUS,
    GRAPH,
    SUPERVISION,
    digest,
    inputs_and_sources,
    matched_initializations,
    relative_due,
    validate_relative_counts,
)
from raster_resource_units import with_resource_units
from run_strand_graph_pilot import load_frozen_corpus
from train_mcts_policy_metric import finite_optimizer, finite_state
from train_raster_resource_pair import CONFIG as OPTIMIZE_CONFIG
from train_raster_resource_pair import development, optimize, save_checkpoint

from rf_knots.embedding_metric import MetricEmbeddingObjective

SCHEMA = "rf-knots-long-general-training-pair-v1"
SEARCH_CONFIG = {**OLD_SEARCH_CONFIG, "seed": 20261079}
SEARCH_MATRIX = [
    dict(arm="control", stage="initial", prior="distance"),
    dict(arm="control", stage="final", prior="distance"),
    dict(arm="long", stage="final", prior="distance"),
    dict(arm="control", stage="final", prior="uniform"),
    dict(arm="long", stage="final", prior="uniform"),
]


def read(path):
    return json.loads(path.read_text())


def evaluation_contract(long_data, short_test):
    audit_path, panel_path = short_test / "consumer-audit.json", short_test / "panel.json"
    audit, panel = read(audit_path), read(panel_path)
    pins = dict(audit["inputs"])
    if (
        audit["status"] != "passed"
        or audit["evaluation_ready"] is not True
        or audit["selected_roots"] != 64
        or audit["classes"] != {"1": 32, "2": 32}
        or audit["attempts"] != SHORT_CONFIG["attempts"]
        or audit["panel_sha256"] != sha(panel_path)
        or Path(audit["panel"]).resolve() != panel_path.resolve()
        or pins.get(audit["panel"]) != sha(panel_path)
        or len(panel["validation"]) != 64
        or len(panel["size_shift"]) != 64
        or panel["complete"] is not True
        or panel["lift"] != SHORT_LIFT
        or panel["acquisition_contract"]["config"] != SHORT_CONFIG
    ):
        raise ValueError("149 requires a completed, bound64-root150 consumer")
    for path in (
        short_test / "manifest.json",
        long_data / "corpus.json",
        long_data / "consumer-audit.json",
        Path("scripts/build_long_mixture_short_u_test.py").resolve(),
    ):
        if pins.get(str(path)) != sha(path):
            raise ValueError("150 consumer not bound to required source/data/completion")
    check_pins(pins)
    merge_pins(pins, {str(audit_path): sha(audit_path)})
    contract = dict(
        schema="rf-knots-long-general-evaluation-contract-v1",
        long_corpus=str(long_data / "corpus.json"),
        long_corpus_sha256=sha(long_data / "corpus.json"),
        short_u_panel=str(panel_path),
        short_u_panel_sha256=sha(panel_path),
        search_config=SEARCH_CONFIG,
        search_matrix=SEARCH_MATRIX,
        search_conditions=128,
        search_roots="all64 original150 roots, each at L10 and L1000",
        resource_contract="unchanged139 build_roots native per-root caps; no tail/value oracle",
        metric_matrix=[
            dict(arm=a, stage=s, panel=p)
            for a, s in (("control", "initial"), ("control", "final"), ("long", "final"))
            for p in ("general147-test", "short-u150-original-and-lift")
        ],
        distance_metrics=[
            "interval violation",
            "constant1",
            "exact class",
            "positive-only ranking",
            "query-group averages",
            "word-length and strand strata",
        ],
        model_initial_identity_required=True,
        per_checkpoint_cache_gate_required=True,
        full_search_replay_required=True,
        same_success_set_cost_comparison_only=True,
        scope="One shared initial model only after exact tensor/function identity. "
        "Uniform tree prior is not an embedding-input ablation; "
        "censored searches are not negatives.",
    )
    return contract, pins


def execution_gate(path, long_data):
    """Consume completed152 evidence, including recomputed gradient/replay checks."""
    from check_raster_mps_joint import tree_check
    from probe_raster_mps_metric import compare

    report = read(path)
    cases = {f"band{b}-exact{c}" for b in range(3) for c in range(3)} | {"maximum-strands"}
    if (
        report["schema"] != "rf-knots-long-general-mps-gate-v1"
        or report["status"] != "passed"
        or report["config"] != CONFIG
        or report["torch_version"] != torch.__version__
        or report["device"] != "mps"
        or report["threads"] != 1
        or report["automatic_mps_fallback"] is not False
        or set(report["cases"]) != cases
        or set(report["arms"]) != set(CONFIG["arms"])
        or report["physical_optimizer_calls"] != 6
        or report["published_training_updates"] != 0
    ):
        raise ValueError("149 requires complete matching152 MPS execution evidence")
    pins = dict(report["inputs"])
    for required in (
        long_data / "corpus.json",
        long_data / "consumer-audit.json",
        Path(__file__).resolve(),
        Path("scripts/check_long_general_mps.py").resolve(),
    ):
        if pins.get(str(required)) != sha(required):
            raise ValueError("execution gate source/data binding differs")
    expected = {"inputs.json", "selection.json"}
    expected |= {f"{c}{suffix}" for c in cases for suffix in (".json", "-gradients.pt")}
    expected |= {
        f"{arm}/{name}"
        for arm in CONFIG["arms"]
        for name in (
            "report.json",
            "engineering-resume.pt",
            "engineering-control-final.pt",
            "engineering-resumed-final.pt",
        )
    }
    if set(report["artifacts"]) != expected:
        raise ValueError("execution gate artifact coverage differs")
    for name, checksum in report["artifacts"].items():
        artifact = path.parent / name
        if sha(artifact) != checksum:
            raise ValueError("execution gate artifact hash differs")
        merge_pins(pins, {str(artifact): checksum})
    check_pins(pins)
    for name, case in report["cases"].items():
        if (
            case != read(path.parent / f"{name}.json")
            or case["passed"] is not True
            or any(
                case[device]["state_sha256"] != report["initial_state_sha256"]
                or case[device]["weights_unchanged"] is not True
                for device in ("cpu", "mps")
            )
        ):
            raise ValueError("execution gate case/initial weights differ")
        vectors = torch.load(
            path.parent / f"{name}-gradients.pt", map_location="cpu", weights_only=False
        )
        comparison = compare(case["cpu"], case["mps"], vectors["cpu"], vectors["mps"])
        if comparison != case["comparison"] or comparison["passed"] is not True:
            raise ValueError("execution gate gradient/scalar arithmetic differs")
    for arm, case in report["arms"].items():
        if (
            case != read(path.parent / arm / "report.json")
            or case["passed"] is not True
            or case["immediate_restore_bitwise"] is not True
            or case["final_close"] is not True
            or case["optimizer_steps"] != [2]
            or case["physical_optimizer_calls"] != 3
            or case["consumed_data_draws"] != 32
            or case["initial_state_sha256"] != report["initial_state_sha256"]
        ):
            raise ValueError("execution gate resume coverage differs")
        saved = [
            torch.load(
                path.parent / arm / f"engineering-{name}.pt", map_location="cpu", weights_only=False
            )
            for name in ("control-final", "resumed-final")
        ]
        tree_check(*saved)
        final_bitwise = True
        try:
            tree_check(*saved, exact=True)
        except (AssertionError, ValueError):
            final_bitwise = False
        if case["final_bitwise"] is not final_bitwise:
            raise ValueError("execution gate bitwise replay claim differs")
        middle = torch.load(
            path.parent / arm / "engineering-resume.pt", map_location="cpu", weights_only=False
        )
        for payload, step in ((middle, 1), (saved[0], 2), (saved[1], 2)):
            if {int(s["step"]) for s in payload["optimizer"]["state"].values()} != {step}:
                raise ValueError("execution gate saved optimizer steps differ")
        if any(
            payload["trace"] != (case["shared_draw_digest"], case["selected_draw_digest"])
            for payload in saved
        ):
            raise ValueError("execution gate saved data trace differs")
    if (
        report["arms"]["control"]["shared_draw_digest"]
        != report["arms"]["long"]["shared_draw_digest"]
    ):
        raise ValueError("execution gate paired data streams differ")
    merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    return report, pins


def verify_checkpoint(directory, entry, contract, arm):
    if Path(entry["file"]).name != entry["file"]:
        raise ValueError("checkpoint name escapes its arm directory")
    path = directory / entry["file"]
    if sha(path) != entry["sha256"]:
        raise ValueError("checkpoint identity differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = payload["steps"]
    if (
        payload["contract_sha256"] != contract
        or payload["arm"] != arm
        or arm not in CONFIG["arms"]
        or payload["config"] != CONFIG
        or step != entry["steps"]
        or type(step) is not int
        or not 0 <= step <= CONFIG["steps"]
        or step % CONFIG["checkpoint_interval"]
    ):
        raise ValueError("checkpoint training contract differs")
    for field in ("rng", "long_rng"):
        random.Random().setstate(payload[field])
    validate_relative_counts(payload["relative_counts"], payload["relative_groups"], step)
    return payload


def checked_loss_contract():
    # Reuse only unchanged mathematical helpers, not135's runner/config globals.
    for key in ("positive_weight", "relative_weight", "relative_every", "grad_clip", "device"):
        if CONFIG[key] != OPTIMIZE_CONFIG[key]:
            raise ValueError("149 requested a different loss/device; do not silently reuse135")


def verify_history(directory, state):
    entries = state.get("metrics_history", [])
    expected = list(
        range(
            CONFIG["checkpoint_interval"],
            state["completed_steps"] + 1,
            CONFIG["checkpoint_interval"],
        )
    )
    if [entry["steps"] for entry in entries] != expected:
        raise ValueError("committed metrics coverage differs")
    for entry in entries:
        if (
            Path(entry["file"]).name != entry["file"]
            or sha(directory / entry["file"]) != entry["sha256"]
        ):
            raise ValueError("committed metrics identity differs")


def verify_completed_arm(directory, contract, arm, dev_labels=("historical", "fresh_general")):
    state = read(directory / "manifest.json")
    if (
        state["status"] != "completed"
        or state["phase"] != "completed"
        or state["arm"] != arm
        or state["contract_sha256"] != contract
        or state["completed_steps"] != CONFIG["steps"]
    ):
        raise ValueError("arm is not completely committed")
    initial = verify_checkpoint(directory, state["initial_checkpoint"], contract, arm)
    final = verify_checkpoint(directory, state["checkpoint"], contract, arm)
    if initial["steps"] != 0 or final["steps"] != CONFIG["steps"]:
        raise ValueError("completed checkpoint stage differs")
    expected = {
        f"dev-{label}-{step:06d}.json" for label in dev_labels for step in CONFIG["dev_steps"]
    }
    if set(state["development_sha256"]) != expected:
        raise ValueError("completed arm lacks DEV coverage")
    for name, checksum in state["development_sha256"].items():
        if sha(directory / name) != checksum:
            raise ValueError("completed DEV report differs")
    verify_history(directory, state)
    return state


def size_counters():
    return dict(
        metric_pairs=0,
        metric_word_tokens=0,
        policy_states=0,
        policy_word_tokens=0,
        metric_max_length=0,
        metric_max_strands=0,
        policy_max_length=0,
        policy_max_strands=0,
    )


def count_batch(counters, draw, relative):
    counters["metric_pairs"] += len(draw.metrics)
    for row in draw.metrics:
        counters["metric_word_tokens"] += len(row.left_word) + len(row.right_word)
        counters["metric_max_length"] = max(
            counters["metric_max_length"], len(row.left_word), len(row.right_word)
        )
        counters["metric_max_strands"] = max(
            counters["metric_max_strands"], row.left_strands, row.right_strands
        )
    requests = [r[0] for r in draw.positives]
    if relative is not None:
        requests += [r[0] for r in relative[0]]
    counters["policy_states"] += len(requests)
    counters["policy_word_tokens"] += sum(len(r[1].word) for r in requests)
    for request in requests:
        counters["policy_max_length"] = max(counters["policy_max_length"], len(request[1].word))
        counters["policy_max_strands"] = max(counters["policy_max_strands"], request[1].strands)


def run_arm(arm, root, net0, architecture, data, dev, contract, pins):
    directory = root / arm
    directory.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    state = (
        read(manifest_path)
        if manifest_path.exists()
        else dict(
            arm=arm,
            contract_sha256=contract,
            status="preparing",
            completed_steps=0,
            development_sha256={},
            seconds_committed=0.0,
            metrics_history=[],
        )
    )
    if state["contract_sha256"] != contract or state["arm"] != arm:
        raise ValueError("arm resume contract differs")
    for name, expected in state["development_sha256"].items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise ValueError("committed DEV report differs")
    if state["status"] == "completed":
        return verify_completed_arm(directory, contract, arm, dev)
    if state["status"] not in ("preparing", "training"):
        raise ValueError("unknown arm state")
    verify_history(directory, state)
    state.update(pid=os.getpid(), status="preparing", phase="initial_reference")
    atomic(manifest_path, state)
    net = with_resource_units(net0, "raw").to("mps")
    objective = MetricEmbeddingObjective().to("mps")
    original, suffix, graph, long_source, positives, raw_groups = data
    preferences = reference_preferences(net, raw_groups, microbatch=1)
    reference_digest = digest([old.tolist() for _, old in preferences])
    if state.get("reference_sha256", reference_digest) != reference_digest:
        raise ValueError("initial policy reference differs")
    state["reference_sha256"] = reference_digest
    optimizer = torch.optim.AdamW(
        [*net.parameters(), *objective.parameters()],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    rng, long_rng = random.Random(CONFIG["seed"]), random.Random(CONFIG["long_seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.mps.manual_seed(CONFIG["seed"])
    completed, shared_trace, selected_trace = 0, digest([]), digest([])
    counts, counters = [0] * len(preferences), size_counters()
    if "checkpoint" in state:
        payload = verify_checkpoint(directory, state["checkpoint"], contract, arm)
        if payload["architecture"] != architecture or payload["relative_groups"] != len(
            preferences
        ):
            raise ValueError("resume architecture or preference coverage differs")
        if payload["steps"] != state["completed_steps"]:
            raise ValueError("committed step disagreement")
        net.load_state_dict(payload["model"], strict=True)
        objective.load_state_dict(payload["objective"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        rng.setstate(payload["rng"])
        long_rng.setstate(payload["long_rng"])
        torch.set_rng_state(payload["torch_rng"])
        torch.mps.set_rng_state(payload["mps_rng"])
        completed, shared_trace, selected_trace, counts, counters = (
            payload[k]
            for k in (
                "steps",
                "shared_draw_digest",
                "selected_draw_digest",
                "relative_counts",
                "size_counters",
            )
        )

    def checkpoint(step):
        finite_state(net, objective)
        finite_optimizer(optimizer)
        torch.mps.synchronize()
        return save_checkpoint(
            directory,
            dict(
                model=net.state_dict(),
                objective=objective.state_dict(),
                optimizer=optimizer.state_dict(),
                architecture=architecture,
                config=CONFIG,
                arm=arm,
                steps=step,
                rng=rng.getstate(),
                long_rng=long_rng.getstate(),
                torch_rng=torch.get_rng_state(),
                mps_rng=torch.mps.get_rng_state(),
                shared_draw_digest=shared_trace,
                selected_draw_digest=selected_trace,
                relative_counts=counts.copy(),
                relative_groups=len(preferences),
                size_counters=counters.copy(),
                contract_sha256=contract,
            ),
        )

    def score_development(step):
        for label, value in dev.items():
            name = f"dev-{label}-{step:06d}.json"
            if name in state["development_sha256"]:
                continue
            report = development(net, objective, *value)
            report.update(
                role=f"{label} DEV; not TEST and not checkpoint selection",
                arm=arm,
                checkpoint=state["checkpoint"],
                contract_sha256=contract,
            )
            target = directory / name
            if target.exists() and read(target) != report:
                raise ValueError("uncommitted DEV report differs; preserve for diagnosis")
            if not target.exists():
                atomic(target, report)
            state["development_sha256"][name] = sha(target)
            atomic(manifest_path, state)

    if "initial_checkpoint" not in state:
        if completed:
            raise ValueError("trained state lacks an initial identity")
        state["initial_checkpoint"] = checkpoint(0)
        state["checkpoint"] = state["initial_checkpoint"]
        atomic(manifest_path, state)
    if completed and any(
        f"dev-{label}-000000.json" not in state["development_sha256"] for label in dev
    ):
        raise ValueError("optimizer advanced before initial DEV")
    if not completed:
        score_development(0)
    started, previous_seconds = time.monotonic(), state["seconds_committed"]
    state.update(status="training", phase="optimizer")
    atomic(manifest_path, state)
    for step in range(completed, CONFIG["steps"]):
        draw = paired_draw(
            arm, rng, long_rng, original, suffix, graph, long_source, positives, preferences
        )
        relative = draw.relative if relative_due(step) else None
        if relative is not None:
            counts[draw.group_index] += 1
        losses = optimize(net, objective, optimizer, draw.metrics, draw.positives, relative)
        shared_trace = digest([shared_trace, draw.shared_digest])
        selected_trace = digest([selected_trace, draw.selected_digest])
        count_batch(counters, draw, relative)
        if (step + 1) % CONFIG["checkpoint_interval"] == 0:
            entry = checkpoint(step + 1)
            if (step + 1) % 256 == 0:
                check_pins(pins)
            state.update(
                checkpoint=entry,
                completed_steps=step + 1,
                **losses,
                shared_draw_digest=shared_trace,
                selected_draw_digest=selected_trace,
                relative_counts=counts.copy(),
                size_counters=counters.copy(),
                seconds_committed=previous_seconds + time.monotonic() - started,
            )
            metric_path = directory / f"metrics-{step + 1:06d}-{digest(state)[:16]}.json"
            if metric_path.exists() and read(metric_path) != state:
                raise ValueError("immutable metrics identity collision")
            if not metric_path.exists():
                atomic(metric_path, state)
            state["metrics_history"].append(
                dict(file=metric_path.name, sha256=sha(metric_path), steps=step + 1)
            )
            atomic(manifest_path, state)
            print(
                json.dumps(
                    {
                        k: state[k]
                        for k in (
                            "arm",
                            "completed_steps",
                            "metric_loss",
                            "positive_loss",
                            "seconds_committed",
                        )
                    }
                ),
                flush=True,
            )
    state.update(phase="final_development")
    atomic(manifest_path, state)
    score_development(CONFIG["steps"])
    check_pins(pins)
    state.update(status="completed", phase="completed")
    atomic(manifest_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--long-data", type=Path, required=True)
    parser.add_argument("--short-test", type=Path, required=True)
    parser.add_argument("--execution-gate", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    checked_loss_contract()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    if (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"
        or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("require native MPS with fallback disabled")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    long_source, fresh_dev, pins = load_long_source(args.long_data)
    evaluation, evaluation_pins = evaluation_contract(args.long_data, args.short_test)
    merge_pins(pins, evaluation_pins)
    gate, gate_pins = execution_gate(args.execution_gate, args.long_data)
    merge_pins(pins, gate_pins)
    old_pins, sources = inputs_and_sources()
    merge_pins(pins, old_pins)
    for name in (
        "scripts/train_long_general_pair.py",
        "scripts/long_general_training_data.py",
        "scripts/train_raster_resource_pair.py",
        "scripts/raster_resource_units.py",
        "scripts/evaluate_resource_search.py",
        "scripts/evaluate_graph_mixture_cost_mcts.py",
        "tests/test_train_long_general_pair.py",
        "tests/test_long_general_training_data.py",
        "research/149-long-general-mixture-training-protocol.md",
        "research/151-long-mixture-evaluation-contract.md",
    ):
        path = Path(name).resolve()
        merge_pins(sources, {str(path): sha(path)})
    merge_pins(pins, sources)
    check_pins(pins)
    contract_payload = dict(
        config=CONFIG,
        inputs=pins,
        evaluation=evaluation,
        torch_version=torch.__version__,
        execution_gate=dict(report=str(args.execution_gate), sha256=sha(args.execution_gate)),
    )
    contract = digest(contract_payload)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    manifest = (
        read(path)
        if args.resume
        else dict(
            schema=SCHEMA, contract=contract_payload, contract_sha256=contract, completed_arms=[]
        )
    )
    if (
        manifest["schema"] != SCHEMA
        or manifest["contract"] != contract_payload
        or manifest["contract_sha256"] != contract
    ):
        raise ValueError("training resume inputs/config/evaluation differs")
    if not args.resume:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i, name in enumerate(sources):
            shutil.copy2(name, snapshot / f"{i:03d}-{Path(name).name}")
    for i, name in enumerate(sources):
        if sha(args.out / "source_snapshot" / f"{i:03d}-{Path(name).name}") != sources[name]:
            raise ValueError("training source snapshot differs")
    if manifest.get("status") == "completed":
        if manifest["completed_arms"] != CONFIG["arms"]:
            raise ValueError("completed pair lacks required arms")
        results = [verify_completed_arm(args.out / arm, contract, arm) for arm in CONFIG["arms"]]
        if (
            results[0]["shared_draw_digest"] != results[1]["shared_draw_digest"]
            or results[0]["reference_sha256"] != results[1]["reference_sha256"]
        ):
            raise ValueError("completed paired data traces/reference differ")
        print("completed training verified; no restart or manifest mutation", flush=True)
        return
    manifest.update(status="preparing", pid=os.getpid(), phase="historical_data_and_initialization")
    atomic(path, manifest)
    raw, corpus = load_frozen_corpus(CORPUS)
    if len(raw["validation"]) != 256:
        raise ValueError("historical DEV coverage differs")
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    data = (
        original,
        suffix,
        load_graph_sampler(GRAPH),
        long_source,
        *load_policy_targets(SUPERVISION),
    )
    net, states, architecture, _ = matched_initializations(pins)
    net.load_state_dict(states["q254"], strict=True)
    if digest(contract_payload) != contract:
        raise ValueError("initialization added unbound inputs after contract freeze")
    if state_digest(net, MetricEmbeddingObjective()) != gate["initial_state_sha256"] or digest(
        architecture
    ) != digest(gate["architecture"]):
        raise ValueError("training initialization differs from the real-weight execution gate")
    dev = dict(historical=(corpus["validation"][0], raw["validation"]), fresh_general=fresh_dev)
    manifest.update(
        long_support=long_source.support, development_counts={k: len(v[0]) for k, v in dev.items()}
    )
    atomic(path, manifest)
    results = []
    for arm in CONFIG["arms"]:
        manifest.update(status="training", phase="arm", current_arm=arm)
        atomic(path, manifest)
        results.append(run_arm(arm, args.out, net, architecture, data, dev, contract, pins))
        if arm not in manifest["completed_arms"]:
            manifest["completed_arms"].append(arm)
        atomic(path, manifest)
    if (
        results[0]["shared_draw_digest"] != results[1]["shared_draw_digest"]
        or results[0]["reference_sha256"] != results[1]["reference_sha256"]
    ):
        raise ValueError("paired sampling or initial reference differs")
    check_pins(pins)
    manifest.update(
        status="completed",
        phase="training_completed_not_promoted",
        current_arm=None,
        independent_training_audit="pending",
        independent_distance_evaluation="pending",
        independent_search_evaluation="pending",
    )
    atomic(path, manifest)


if __name__ == "__main__":
    main()
