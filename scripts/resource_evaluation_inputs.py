"""Fail-closed135 checkpoint feature identity and136/138 panel consumers."""

import json
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import sha
from build_resource_general_test import CONFIG as GENERAL_CONFIG
from build_resource_general_test import audit_row, provenance, select
from build_resource_short_u_test import CONFIG as SHORT_CONFIG
from burau_integer_check import determinant
from raster_log_training import matched_initializations
from raster_resource_units import ResourceUnitsHead, with_resource_units
from train_raster_resource_pair import CONFIG, verify_checkpoint

from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import canonical, invariants, owner, verify_record

PANEL_ROOTS = {
    "short-u": Path("artifacts/resource-units-fresh-short-u-test-v1"),
    "general": Path("artifacts/resource-units-fresh-general-test-v1"),
}


def model_entry(run, audit, arm, stage):
    if arm not in CONFIG["arms"] or stage not in ("initial", "final"):
        raise ValueError("unknown checkpoint arm/stage")
    if (
        audit["status"] != "passed"
        or set(audit["arms"]) != set(CONFIG["arms"])
        or audit.get("initial_tensors_identical") is not True
        or audit.get("data_rng_aligned") is not True
    ):
        raise ValueError("require complete verified135 pair")
    entry = audit["arms"][arm][stage]
    expected_step = 0 if stage == "initial" else CONFIG["steps"]
    if entry["steps"] != expected_step or Path(entry["file"]).name != entry["file"]:
        raise ValueError("checkpoint stage/file differs")
    path = run / arm / entry["file"]
    if audit["inputs"].get(str(path)) != entry["sha256"] or sha(path) != entry["sha256"]:
        raise ValueError("checkpoint not bound to the paired audit")
    return entry, path


def load_model(run, pair_path, arm, stage):
    audit = json.loads(pair_path.read_text())
    entry, path = model_entry(run, audit, arm, stage)
    pins = dict(audit["inputs"])
    merge_pins(
        pins, {str(pair_path): sha(pair_path), str(Path(__file__).resolve()): sha(Path(__file__))}
    )
    check_pins(pins)
    parent = json.loads((run / "manifest.json").read_text())
    payload = verify_checkpoint(run / arm, entry, parent["contract_sha256"], arm)
    net, _, architecture, _ = matched_initializations(pins)
    if payload["architecture"] != architecture:
        raise ValueError("checkpoint architecture differs")
    net = with_resource_units(net, arm)
    net.load_state_dict(payload["model"], strict=True)
    objective = MetricEmbeddingObjective()
    objective.load_state_dict(payload["objective"], strict=True)
    if (arm == "units") != isinstance(net.head, ResourceUnitsHead):
        raise ValueError("policy feature mode differs from checkpoint arm")
    net.eval()
    objective.eval()
    return net, objective, pins, path


def load_panel(kind):
    if kind not in PANEL_ROOTS:
        raise ValueError("unknown panel kind")
    root = PANEL_ROOTS[kind]
    state_path = root / ("manifest.json" if kind == "short-u" else "progress.json")
    state = json.loads(state_path.read_text())
    if state["status"] != "completed" or state.get("evaluation_ready") is not True:
        raise ValueError("panel is not completed and adequate")
    path = root / "generator" / "panel.json" if kind == "short-u" else root / "panel.json"
    if sha(path) != state["panel_sha256"]:
        raise ValueError("panel hash differs")
    panel = json.loads(path.read_text())
    if not panel["complete"] or not panel["construction_complete"] or panel["partition"] != "test":
        raise ValueError("panel construction is incomplete")
    pins = dict(state["contract"]["inputs"])
    merge_pins(pins, panel["contract"]["inputs"])
    merge_pins(pins, {str(path): sha(path), str(state_path): sha(state_path)})
    check_pins(pins)
    if kind == "general":
        extra, groups, training, excluded = provenance()
        merge_pins(pins, extra)
        if (
            state["contract"]["config"] != GENERAL_CONFIG
            or panel["contract"] != state["contract"]
            or state["next_index"] != GENERAL_CONFIG["attempts"]
            or len(state["rows"]) != GENERAL_CONFIG["attempts"]
            or panel["excluded_query_groups"] != [list(g) for g in sorted(excluded)]
            or panel["training_groups"]
            != {k: [list(g) for g in sorted(v)] for k, v in groups.items()}
        ):
            raise ValueError("general attempt/partition contract differs")
        records, decisions, classes, count, adequate = select(
            state["rows"], training, excluded, GENERAL_CONFIG
        )
        if (
            not adequate
            or records != panel["validation"]
            or panel["size_shift"] != []
            or panel["selection_decisions"] != decisions
            or panel["classes"] != classes
            or panel["groups"] != count
            or panel["selection_counts"] != dict(Counter(decisions))
        ):
            raise ValueError("general selection differs from frozen rule")
        for row in records:
            audit_row(row, row["index"])
    else:
        registry_path = root / "excluded-groups.json"
        progress_path = root / "generator" / "progress.json"
        if (
            sha(registry_path) != state["registry_sha256"]
            or sha(progress_path) != state["progress_sha256"]
            or state["contract"]["config"] != SHORT_CONFIG
            or {k: panel["contract"][k] for k in SHORT_CONFIG} != SHORT_CONFIG
        ):
            raise ValueError("short-u config/exclusion/progress identity differs")
        progress = json.loads(progress_path.read_text())
        if (
            progress["next_index"] != SHORT_CONFIG["attempts"]
            or progress["status"] != "completed"
            or sum(progress["counts"].values()) != SHORT_CONFIG["attempts"]
            or progress["records"] != panel["validation"]
        ):
            raise ValueError("short-u fixed attempt coverage differs")
        registry = json.loads(registry_path.read_text())
        excluded = {tuple(r["left_fingerprint"]) for r in registry["validation"]}
        if len(panel["validation"]) != 64 or len(panel["size_shift"]) != 64:
            raise ValueError("short-u requires64 roots and64 corresponding lifts")
        counts, seen = Counter(), set()
        for row, lifted in zip(panel["validation"], panel["size_shift"], strict=True):
            witness, large = verify_record(row), verify_record(lifted)
            fp = invariants(witness.start.word, witness.start.strands)
            group = fp[0], abs(fp[1])
            key = canonical(witness.start.word, witness.start.strands)
            if (
                fp[0] != determinant(witness.start.word, witness.start.strands)
                or group in excluded
                or owner(fp) != "test"
                or key in seen
                or row["index"] != lifted["index"]
                or row["seed"] != lifted["seed"]
                or row["left_fingerprint"] != lifted["left_fingerprint"]
                or large.crossing_changes != witness.crossing_changes
            ):
                raise ValueError("short-u replay/partition/lift binding differs")
            seen.add(key)
            counts[str(witness.crossing_changes)] += 1
        if counts != {"1": 32, "2": 32}:
            raise ValueError("short-u witnessed CC coverage differs")
        merge_pins(
            pins, {str(registry_path): sha(registry_path), str(progress_path): sha(progress_path)}
        )
    check_pins(pins)
    return panel, pins, path
