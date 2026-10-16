"""Post-exit149 audit: all commits, actual data draws, initial weights and DEV."""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_resource_training_pair import check_optimizer, check_tensors
from build_mcts_training_supervision import atomic, sha
from graph_metric_sampling import load_graph_sampler
from long_general_training_data import CONFIG, load_long_source, paired_draw
from mcts_joint_training import load_metric_samplers, load_policy_targets
from raster_log_training import (
    CORPUS,
    GRAPH,
    SUPERVISION,
    digest,
    matched_initializations,
    relative_due,
)
from run_strand_graph_pilot import load_frozen_corpus
from train_long_general_pair import (
    SCHEMA,
    checked_loss_contract,
    evaluation_contract,
    execution_gate,
    verify_checkpoint,
    verify_completed_arm,
)

from rf_knots.embedding_metric import MetricEmbeddingObjective


def read(path):
    return json.loads(path.read_text())


def replay_draws(arm, data):
    rng, long_rng = random.Random(CONFIG["seed"]), random.Random(CONFIG["long_seed"])
    original, suffix, graph, long_source, positives, raw_groups = data
    preferences = [(g, torch.zeros(len(g))) for g in raw_groups]
    shared, selected, counts = digest([]), digest([]), [0] * len(preferences)
    counters = dict(
        metric_pairs=0,
        metric_word_tokens=0,
        metric_max_length=0,
        metric_max_strands=0,
        policy_states=0,
        policy_word_tokens=0,
        policy_max_length=0,
        policy_max_strands=0,
    )

    def state():
        return dict(
            shared_draw_digest=shared,
            selected_draw_digest=selected,
            relative_counts=counts.copy(),
            size_counters=counters.copy(),
            rng=rng.getstate(),
            long_rng=long_rng.getstate(),
        )

    yield 0, state()
    for step in range(CONFIG["steps"]):
        draw = paired_draw(
            arm, rng, long_rng, original, suffix, graph, long_source, positives, preferences
        )
        shared = digest([shared, draw.shared_digest])
        selected = digest([selected, draw.selected_digest])
        endpoints = [(r.left_word, r.left_strands) for r in draw.metrics] + [
            (r.right_word, r.right_strands) for r in draw.metrics
        ]
        counters["metric_pairs"] += len(draw.metrics)
        counters["metric_word_tokens"] += sum(len(w) for w, _ in endpoints)
        counters["metric_max_length"] = max(
            counters["metric_max_length"], *(len(w) for w, _ in endpoints)
        )
        counters["metric_max_strands"] = max(
            counters["metric_max_strands"], *(n for _, n in endpoints)
        )
        requests = [request for request, _ in draw.positives]
        if relative_due(step):
            counts[draw.group_index] += 1
            requests.extend(request for request, _, _ in draw.relative[0])
        counters["policy_states"] += len(requests)
        counters["policy_word_tokens"] += sum(len(state.word) for _, state, _ in requests)
        counters["policy_max_length"] = max(
            counters["policy_max_length"], *(len(s.word) for _, s, _ in requests)
        )
        counters["policy_max_strands"] = max(
            counters["policy_max_strands"], *(s.strands for _, s, _ in requests)
        )
        if (step + 1) % CONFIG["checkpoint_interval"] == 0:
            yield step + 1, state()


def verify_development(report, rows, records):
    if (
        report["count"] != len(rows)
        or len(report["records"]) != len(rows)
        or len(rows) != len(records)
    ):
        raise ValueError("DEV record coverage differs")
    grouped, exact, errors, constants = defaultdict(list), defaultdict(list), [], []
    for actual, row, source in zip(report["records"], rows, records, strict=True):
        lo, hi, p = row.spec.distance_lower, row.spec.distance_upper, actual["prediction"]
        g = source["left_fingerprint"][0], abs(source["left_fingerprint"][1])
        violation, constant = max(lo - p, p - hi, 0), max(lo - 1, 1 - hi, 0)
        if not math.isfinite(p) or actual != dict(
            index=row.spec.index,
            lower=lo,
            upper=hi,
            prediction=p,
            violation=violation,
            constant1_violation=constant,
            query_group=list(g),
        ):
            raise ValueError("DEV row identity or arithmetic differs")
        errors.append(violation)
        constants.append(constant)
        grouped[g].append(violation)
        if lo == hi:
            exact[str(lo)].append(violation)
    expected = dict(
        groups=len(grouped),
        mean_interval_violation=sum(errors) / len(errors),
        equal_group_interval_violation=sum(sum(v) / len(v) for v in grouped.values())
        / len(grouped),
        constant1_mean_violation=sum(constants) / len(constants),
        exact_groups={
            k: dict(count=len(v), mean_interval_violation=sum(v) / len(v)) for k, v in exact.items()
        },
    )
    if any(report[k] != value for k, value in expected.items()):
        raise ValueError("DEV aggregate arithmetic differs")


def audit_arm(directory, arm, parent, expected_state, architecture, parameters, data, dev):
    contract = parent["contract_sha256"]
    state = verify_completed_arm(directory, contract, arm, dev)
    pins = {str(directory / "manifest.json"): sha(directory / "manifest.json")}
    entries = {0: state["initial_checkpoint"]}
    for entry in state["metrics_history"]:
        path = directory / entry["file"]
        metadata = read(path)
        if (
            metadata["arm"] != arm
            or metadata["contract_sha256"] != contract
            or metadata["completed_steps"] != entry["steps"]
        ):
            raise ValueError("metrics commit identity differs")
        entries[entry["steps"]] = metadata["checkpoint"]
        pins[str(path)] = entry["sha256"]
    if entries[CONFIG["steps"]] != state["checkpoint"]:
        raise ValueError("final checkpoint not in committed history")
    initial, final = None, None
    objective_state = MetricEmbeddingObjective().state_dict()
    for step, expected in replay_draws(arm, data):
        entry = entries[step]
        payload = verify_checkpoint(directory, entry, contract, arm)
        if payload["steps"] != step or payload["architecture"] != architecture:
            raise ValueError("checkpoint step/architecture differs")
        if any(payload[k] != value for k, value in expected.items()):
            raise ValueError("checkpoint RNG/draw/counter replay differs")
        check_tensors(payload["model"], expected_state, exact=step == 0)
        check_tensors(payload["objective"], objective_state, exact=step == 0)
        check_optimizer(payload, parameters, step)
        if not torch.equal(payload["objective"]["rm_scale_raw"], objective_state["rm_scale_raw"]):
            raise ValueError("unused RM objective scale changed")
        pins[str(directory / entry["file"])] = entry["sha256"]
        if step == 0:
            initial = payload
        final = payload
    if any(
        final[k] != state[k]
        for k in ("shared_draw_digest", "selected_draw_digest", "relative_counts", "size_counters")
    ):
        raise ValueError("final checkpoint/manifest counters differ")
    changed = [k for k, v in expected_state.items() if not torch.equal(v, final["model"][k])]
    for prefix in ("encoder.tiles.", "encoder.heads.", "head."):
        if not any(k.startswith(prefix) for k in changed):
            raise ValueError(f"no training update in {prefix}")
    for label, (rows, records) in dev.items():
        for step, entry in (
            (0, state["initial_checkpoint"]),
            (CONFIG["steps"], state["checkpoint"]),
        ):
            path = directory / f"dev-{label}-{step:06d}.json"
            report = read(path)
            if (
                report["arm"] != arm
                or report["checkpoint"] != entry
                or report["contract_sha256"] != contract
            ):
                raise ValueError("DEV checkpoint binding differs")
            verify_development(report, rows, records)
            pins[str(path)] = sha(path)
    return (
        dict(initial=initial, final=final),
        pins,
        dict(
            initial=state["initial_checkpoint"],
            final=state["checkpoint"],
            steps=CONFIG["steps"],
            checkpoints_checked=len(entries),
            changed_model_tensor_keys=changed,
            shared_draw_digest=final["shared_draw_digest"],
            selected_draw_digest=final["selected_draw_digest"],
            relative_counts=final["relative_counts"],
            size_counters=final["size_counters"],
            reference_sha256=state["reference_sha256"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        raise ValueError("audit output must be new")
    parent_path = args.run / "manifest.json"
    parent = read(parent_path)
    if (
        parent["schema"] != SCHEMA
        or parent["status"] != "completed"
        or parent["completed_arms"] != CONFIG["arms"]
        or parent["current_arm"] is not None
        or parent["contract"]["config"] != CONFIG
        or digest(parent["contract"]) != parent["contract_sha256"]
    ):
        raise ValueError("require a completely committed149 pair")
    checked_loss_contract()
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    pins = dict(parent["contract"]["inputs"])
    for path in (
        parent_path,
        Path(__file__).resolve(),
        Path("scripts/audit_resource_training_pair.py").resolve(),
        Path("tests/test_long_general_training_audit.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    contract = parent["contract"]
    long_data = Path(contract["evaluation"]["long_corpus"]).parent
    short_test = Path(contract["evaluation"]["short_u_panel"]).parent
    long_source, fresh_dev, data_pins = load_long_source(long_data)
    evaluation, eval_pins = evaluation_contract(long_data, short_test)
    gate_path = Path(contract["execution_gate"]["report"])
    if sha(gate_path) != contract["execution_gate"]["sha256"]:
        raise ValueError("execution gate hash differs from the training contract")
    gate, gate_pins = execution_gate(gate_path, long_data)
    if evaluation != contract["evaluation"]:
        raise ValueError("frozen evaluation matrix differs")
    for extra in (data_pins, eval_pins, gate_pins):
        merge_pins(pins, extra)
    net, states, architecture, _ = matched_initializations(pins)
    if digest(architecture) != digest(gate["architecture"]):
        raise ValueError("execution/training architectures differ")
    objective = MetricEmbeddingObjective()
    parameters = list(net.named_parameters()) + [
        ("objective." + n, p) for n, p in objective.named_parameters()
    ]
    raw, corpus = load_frozen_corpus(CORPUS)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    data = (
        original,
        suffix,
        load_graph_sampler(GRAPH),
        long_source,
        *load_policy_targets(SUPERVISION),
    )
    dev = dict(historical=(corpus["validation"][0], raw["validation"]), fresh_general=fresh_dev)
    results, payloads = {}, {}
    for arm in CONFIG["arms"]:
        payloads[arm], more, results[arm] = audit_arm(
            args.run / arm, arm, parent, states["q254"], architecture, parameters, data, dev
        )
        merge_pins(pins, more)
    a, b = (payloads[arm]["final"] for arm in CONFIG["arms"])
    if any(a[k] != b[k] for k in ("rng", "long_rng", "shared_draw_digest", "relative_counts")) or (
        results["control"]["reference_sha256"] != results["long"]["reference_sha256"]
    ):
        raise ValueError("paired historical/new-data streams or initial references differ")
    check_pins(pins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(
        args.out,
        dict(
            status="passed",
            inputs=pins,
            arms=results,
            initial_tensors_identical=True,
            data_rng_aligned=True,
            backend_final_rng_aligned={
                k: torch.equal(a[k], b[k]) for k in ("torch_rng", "mps_rng")
            },
            scope="All committed optimizer states and draws checked; no TEST/MCTS quality claim. "
            "Different input shapes can consume backend randomness differently; paired-data "
            "identity refers to the two Python sampling streams, not equal hardware work.",
        ),
    )
    print(
        json.dumps(
            dict(
                status="passed",
                checkpoints_per_arm=1 + CONFIG["steps"] // CONFIG["checkpoint_interval"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
