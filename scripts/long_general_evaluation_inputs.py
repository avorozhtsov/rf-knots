"""Fail-closed149 checkpoints and frozen147/150 independent evaluation inputs."""

import copy
import json
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_resource_training_pair import check_tensors
from build_long_general_cc_supply import group
from build_mcts_training_supervision import sha
from long_general_training_data import CONFIG, load_long_source, materialize
from pretrained_raster_policy import policy_log_probabilities
from probe_raster_cache_full_network import state_digest
from raster_log_training import digest, matched_initializations
from train_long_general_pair import SCHEMA, evaluation_contract, verify_checkpoint

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import owner
from rf_knots.site_head import SiteHead


def read(path):
    return json.loads(path.read_text())


def model_entry(run, audit, arm, stage):
    if arm not in CONFIG["arms"] or stage not in ("initial", "final"):
        raise ValueError("unknown149 arm/stage")
    if (
        audit["status"] != "passed"
        or set(audit["arms"]) != set(CONFIG["arms"])
        or audit.get("initial_tensors_identical") is not True
        or audit.get("data_rng_aligned") is not True
        or any(
            a["steps"] != CONFIG["steps"]
            or a["checkpoints_checked"] != 1 + CONFIG["steps"] // CONFIG["checkpoint_interval"]
            for a in audit["arms"].values()
        )
    ):
        raise ValueError("require a complete independently audited149 pair")
    entry = audit["arms"][arm][stage]
    if (
        entry["steps"] != (0 if stage == "initial" else CONFIG["steps"])
        or Path(entry["file"]).name != entry["file"]
    ):
        raise ValueError("checkpoint stage/file differs")
    path = run / arm / entry["file"]
    if audit["inputs"].get(str(path)) != entry["sha256"] or sha(path) != entry["sha256"]:
        raise ValueError("checkpoint not bound to149 audit")
    return entry, path


def load_inputs(run, audit_path):
    parent_path = run / "manifest.json"
    parent, audit = read(parent_path), read(audit_path)
    if (
        parent["schema"] != SCHEMA
        or parent["status"] != "completed"
        or parent["completed_arms"] != CONFIG["arms"]
        or parent["current_arm"] is not None
        or parent["contract"]["config"] != CONFIG
        or digest(parent["contract"]) != parent["contract_sha256"]
    ):
        raise ValueError("require completely committed149 training")
    pins = dict(audit["inputs"])
    for path in (parent_path, Path("scripts/audit_long_general_training_pair.py").resolve()):
        if pins.get(str(path)) != sha(path):
            raise ValueError("training audit lacks required source/manifest binding")
    merge_pins(pins, parent["contract"]["inputs"])
    for arm in CONFIG["arms"]:
        for stage in ("initial", "final"):
            model_entry(run, audit, arm, stage)
    evaluation = parent["contract"]["evaluation"]
    actual, more = evaluation_contract(
        Path(evaluation["long_corpus"]).parent, Path(evaluation["short_u_panel"]).parent
    )
    if actual != evaluation:
        raise ValueError("frozen evaluation matrix differs")
    merge_pins(pins, more)
    for path in (audit_path, Path(__file__).resolve()):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    return parent, audit, pins


def test_records(corpus):
    """Consume passed mathematical audit, then recheck query/partition structure."""
    forbidden = set(map(tuple, corpus["excluded_query_groups"]))
    for groups in corpus["training_groups"].values():
        forbidden.update(map(tuple, groups))
    forbidden.update(group(fp) for r in corpus["records"]["train"] for fp in r["fingerprints"])
    forbidden.update(group(r["left_fingerprint"]) for r in corpus["records"]["validation"])
    rows = corpus["records"]["test"]
    if not rows:
        raise ValueError("empty general TEST")
    keys = set()
    for row in rows:
        materialize(row)
        if (
            row["split"] != "test"
            or owner(row["left_fingerprint"]) != "test"
            or group(row["left_fingerprint"]) in forbidden
            or any(owner(fp) == "quarantine" for fp in row["fingerprints"])
            or row["pair_key"] in keys
        ):
            raise ValueError("general TEST query exposure/partition/uniqueness differs")
        keys.add(row["pair_key"])
    return rows


def load_panel(kind, parent, pins):
    evaluation = parent["contract"]["evaluation"]
    if kind == "general":
        path = Path(evaluation["long_corpus"])
        _, _, more = load_long_source(path.parent)
        merge_pins(pins, more)
        views = {"test": test_records(read(path))}
    elif kind == "short-u":
        path = Path(evaluation["short_u_panel"])
        panel = read(path)
        views = {k: panel[k] for k in ("validation", "size_shift")}
        if any(len(rows) != 64 for rows in views.values()):
            raise ValueError("require all64 original roots and64 lifts")
    else:
        raise ValueError("unknown evaluation panel")
    expected = evaluation["long_corpus_sha256" if kind == "general" else "short_u_panel_sha256"]
    if sha(path) != expected or pins.get(str(path)) != expected:
        raise ValueError("panel is not frozen in149 contract")
    check_pins(pins)
    return views, path


@torch.no_grad()
def initial_identity(net, objective, other, endpoint):
    """Exact state/architecture identity plus deterministic TRAIN forward probes."""
    check_tensors(other["model"], net.state_dict(), exact=True)
    check_tensors(other["objective"], objective.state_dict(), exact=True)
    twin = copy.deepcopy(net).eval()
    twin.load_state_dict(other["model"], strict=True)
    word, strands = tuple(endpoint["word"]), endpoint["strands"]
    instance = M.Instance("149-initial-identity-TRAIN", word, strands, 2)
    state = M.SearchState(word, strands, 0, 0)
    positions = sorted({0, len(word) // 2, len(word) - 1})
    actions = tuple(instance.spec.encode(CROSSING_CHANGE, i) for i in positions)
    before = state_digest(net, objective)
    tensors = (torch.tensor([word]), torch.tensor([len(word)]), torch.tensor([strands]))
    vectors = [model.encoder(*tensors)["knot"] for model in (net, twin)]
    if not all(torch.isfinite(v).all() for v in vectors) or not torch.equal(*vectors):
        raise ValueError("shared initial encoder function differs")
    for head in positions:
        policies = [
            policy_log_probabilities(model, [(instance, state, actions)], heads=[head])
            for model in (net, twin)
        ]
        if not torch.equal(*policies):
            raise ValueError("shared initial position-aware policy differs")
    if state_digest(net, objective) != before or state_digest(twin, objective) != before:
        raise ValueError("initial forward verification changed weights")
    return dict(
        exact_tensors=True,
        encoder_equal=True,
        policy_equal=True,
        heads=positions,
        endpoint=endpoint,
        state_sha256=before,
        scope="Identical architecture/raw feature path and tensors; TRAIN forward spot-checks",
    )


def load_model(run, parent, audit, pins, arm, stage):
    entry, path = model_entry(run, audit, arm, stage)
    payload = verify_checkpoint(run / arm, entry, parent["contract_sha256"], arm)
    net, _, architecture, _ = matched_initializations(pins)
    if payload["architecture"] != architecture or type(net.head) is not SiteHead:
        raise ValueError("149 architecture/raw feature implementation differs")
    net.load_state_dict(payload["model"], strict=True)
    objective = MetricEmbeddingObjective()
    objective.load_state_dict(payload["objective"], strict=True)
    net.eval()
    objective.eval()
    shared = None
    if stage == "initial":
        other_arm = "long" if arm == "control" else "control"
        other_entry, _ = model_entry(run, audit, other_arm, "initial")
        other = verify_checkpoint(
            run / other_arm, other_entry, parent["contract_sha256"], other_arm
        )
        if other["architecture"] != architecture:
            raise ValueError("shared initial architecture differs")
        corpus = read(Path(parent["contract"]["evaluation"]["long_corpus"]))
        shared = initial_identity(net, objective, other, corpus["records"]["train"][0]["states"][0])
    check_pins(pins)
    return net, objective, path, shared
