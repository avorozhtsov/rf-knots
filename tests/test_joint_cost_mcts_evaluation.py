import importlib
import json
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    torch.set_num_threads(1)
    return importlib.import_module("evaluate_joint_cost_mcts")


def root(weight=10):
    instance = M.Instance("engineering", (1, -1, 1), 2, 0, cap_moves=8, cc_weight=weight)
    return {"condition": 0, "panel_index": 0, "instance": instance.to_dict()}


def checkpoint():
    torch.manual_seed(902)
    architecture = dict(
        width=16,
        depth=2,
        embedding_dim=8,
        budget_dim=5,
        encoder_kind="strand_graph",
        normalized_knot=True,
    )
    net = M.SitePolicyValueNet(**architecture)
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    return {
        "architecture": architecture,
        "model": net.state_dict(),
        "objective": objective.state_dict(),
    }


@pytest.mark.parametrize("learned", [False, True])
def test_batch_replay_compute_and_repeatability(evaluator, learned):
    config = {**evaluator.CONFIG, "simulations": 3, "horizon": 8}
    model = checkpoint() if learned else None
    rows, counts = evaluator.run_batch([root()], model, 8, config)
    again, counts_again = evaluator.run_batch([root()], model, 8, config)
    assert rows == again and counts == counts_again
    assert rows[0]["search_seed"] == config["seed"] + 8
    assert rows[0]["tree_seed"] == (config["seed"] + 8) * 9973
    assert counts["conditions"] == 1 and counts["episodes"] > 0
    assert counts["continuation_moves"] > 0
    assert counts["policy_neural_evaluations"] == (counts["continuation_moves"] if learned else 0)
    if rows[0]["solved"]:
        witness = UnknotWitness.from_dict(rows[0]["witness"])
        witness.verify()
        assert rows[0]["L10"] == rows[0]["L1000"] == witness.moves


def test_checkpoint_nan_is_not_hidden_by_uniform_fallback(evaluator):
    model = checkpoint()
    next(iter(model["model"].values())).fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite checkpoint"):
        evaluator.run_batch([root()], model, 0)


@pytest.mark.parametrize("full_schema", [False, True])
def test_root_pairing_does_not_change_resources_or_inject_witness(
    evaluator, monkeypatch, full_schema
):
    i = M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=3)
    s = i.spec
    witness = UnknotWitness.from_actions(
        i.word,
        i.strands,
        s,
        [s.encode(CROSSING_CHANGE, 0), s.encode(REDUCE, 0), s.encode(DESTABILIZE)],
    )
    monkeypatch.setattr(evaluator, "verify_evaluation_pair", lambda record: witness)
    panel = {
        "complete": True,
        "pair_kind": "short-u",
        "validation": [
            {"left_fingerprint": [3, -2], "interval": {"right": {"word": [], "strands": 1}}}
        ],
    }
    if full_schema:
        panel["validation"][0]["interval"]["right"]["cyclic_band_generators"] = False
    a, b = evaluator.build_roots(panel)
    assert (a["condition"], b["condition"]) == (0, 1)
    assert [a["instance"]["cc_weight"], b["instance"]["cc_weight"]] == [10, 1000]

    def strip(row):
        return {k: v for k, v in row["instance"].items() if k not in ("uid", "cc_weight")}

    assert strip(a) == strip(b)
    assert "witness" not in a and "actions" not in a["instance"]
    assert a["reference_L10"] == 13 and a["reference_L1000"] == 1003
    panel["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        evaluator.build_roots(panel)


def test_resume_coverage_and_hashes_fail_closed(evaluator, tmp_path):
    shard = tmp_path / "conditions.json"
    shard.write_text(json.dumps([{"condition": 0}, {"condition": 1}]))
    manifest = {
        "next_condition": 2,
        "shards": [{"file": shard.name, "sha256": evaluator.sha(shard)}],
    }
    assert len(evaluator.committed_rows(manifest, tmp_path)) == 2
    manifest["next_condition"] = 3
    with pytest.raises(ValueError, match="coverage"):
        evaluator.committed_rows(manifest, tmp_path)
    manifest["next_condition"] = 2
    shard.write_text("[]")
    with pytest.raises(ValueError, match="shard changed"):
        evaluator.committed_rows(manifest, tmp_path)


@pytest.mark.parametrize(
    "endpoint",
    [
        {"word": [], "strands": 1, "cyclic_band_generators": True},
        {"word": [], "strands": 2},
        {"word": [1], "strands": 2},
        {"word": [], "strands": 1, "unknown_field": 0},
    ],
)
def test_root_reader_does_not_accept_other_endpoint_semantics(evaluator, endpoint):
    panel = {
        "complete": True,
        "pair_kind": "short-u",
        "validation": [{"interval": {"right": endpoint}}],
    }
    with pytest.raises(ValueError, match="explicit-unknot"):
        evaluator.build_roots(panel)


def test_final_auditor_replays_actual_batch(evaluator):
    auditor = importlib.import_module("audit_joint_cost_mcts")
    config = {**evaluator.CONFIG, "simulations": 3, "horizon": 8}
    source = root()
    rows, counts = evaluator.run_batch([source], checkpoint(), 0, config)
    i = M.Instance(**{**source["instance"], "word": tuple(source["instance"]["word"])})
    checked = auditor.audit_condition(i, rows[0], config)
    assert checked["continuation_moves"] == counts["continuation_moves"]
    assert checked["policy_action_scores"] == counts["policy_action_scores"]
    rows[0]["episodes"][0]["continuation"]["probabilities"][0] = 0
    with pytest.raises(AssertionError, match="probability"):
        auditor.audit_condition(i, rows[0], config)


@pytest.mark.parametrize("fault", ["cost", "weight", "incumbent", "depth", "absent"])
def test_final_auditor_rejects_success_summary_corruption(evaluator, fault):
    auditor = importlib.import_module("audit_joint_cost_mcts")
    i = M.Instance("t", (1, 1, 1), 2, 1, cap_moves=4, cc_weight=10)
    s = i.spec
    witness = UnknotWitness.from_actions(
        i.word,
        i.strands,
        s,
        [s.encode(CROSSING_CHANGE, 0), s.encode(REDUCE, 0), s.encode(DESTABILIZE)],
    )
    row = {
        "cc_weight": 10,
        "simulations": 3,
        "maximum_tree_depth": 2,
        "maximum_observed_depth": 3,
        "episodes": [],
        "solved": True,
        "witness": witness.to_dict(),
        "L10": 13,
        "L1000": 1003,
        "incumbents": [{"simulation": 3, "cost": 13, "cc": 1, "moves": 3}],
    }
    assert auditor.audit_condition(i, row, evaluator.CONFIG)["solved_conditions"] == 1
    if fault == "cost":
        row["L10"] = 12
    elif fault == "weight":
        row["cc_weight"] = 1000
    elif fault == "incumbent":
        row["incumbents"] *= 2
    elif fault == "depth":
        row["maximum_observed_depth"] = 5
    else:
        row["witness"] = None
    with pytest.raises(ValueError):
        auditor.audit_condition(i, row, evaluator.CONFIG)
