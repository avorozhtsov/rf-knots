import importlib.util
import hashlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def evaluation(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "panel_evaluation", scripts / "evaluate_strand_graph_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_determinant_nonidentity_is_only_a_one_step_lower_bound(evaluation):
    pair = {"lower": 0, "upper": 2}
    assert evaluation.certified_lower(pair, (9, 0), (1, 0), determinant=True) == 1
    assert evaluation.certified_lower(pair, (101, 0), (1, 0), determinant=True) == 1
    assert evaluation.certified_lower(pair, (9, 0), (1, 0), determinant=False) == 0
    # Equal fingerprints do not imply equal knots or strengthen the bound.
    assert evaluation.certified_lower(pair, (9, 0), (9, 0), determinant=True) == 0
    assert (
        evaluation.certified_lower({"lower": 2, "upper": 2}, (3, 2), (3, -2), determinant=True) == 2
    )


def test_lower_bound_contradiction_fails_closed(evaluation):
    with pytest.raises(ValueError, match="contradicts"):
        evaluation.certified_lower({"lower": 0, "upper": 0}, (9, 0), (1, 0), determinant=True)


def test_constant_control_exposes_interval_imbalance(evaluation):
    report = evaluation.constant_baseline([{"lower": 1, "upper": 2}, {"lower": 2, "upper": 2}], 2)
    assert report["mean_interval_violation"] == 0
    assert report["exact_groups"] == {"2": {"count": 1, "mae": 0}}


def test_saved_initial_state_is_pinned(evaluation, tmp_path):
    architecture = {"width": 48}
    path = tmp_path / "initial.pt"
    torch.save({"architecture": architecture, "model": {"w": torch.tensor([3.0])}}, path)
    manifest = {"initial_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    state = evaluation.load_initial_checkpoint(tmp_path, manifest, architecture)
    assert torch.equal(state["model"]["w"], torch.tensor([3.0]))
    with pytest.raises(ValueError, match="architecture"):
        evaluation.load_initial_checkpoint(tmp_path, manifest, {"width": 49})
    with pytest.raises(ValueError, match="SHA"):
        evaluation.load_initial_checkpoint(tmp_path, {"initial_checkpoint_sha256": "bad"}, architecture)
    assert evaluation.load_initial_checkpoint(tmp_path, {}, architecture) is None


def test_group_report_does_not_count_representations_as_independent(evaluation):
    predictions = [{"lower": 1, "upper": 1, "prediction": 0, "violation": 1} for _ in range(3)]
    predictions.append({"lower": 0, "upper": 2, "prediction": 1, "violation": 0})
    records = [{"left_fingerprint": [5, -4], "right_fingerprint": [3, -2]} for _ in range(3)]
    records.append({"left_fingerprint": [7, 0], "right_fingerprint": [1, 0]})
    report = evaluation.group_report(predictions, records)
    assert report["group_count"] == 2
    assert report["mean_group_interval_violation"] == 0.5
    assert report["groups"]["5:4"]["count"] == 3
    assert predictions[0]["panel_index"] == 0
    assert report["groups"]["7:0"]["constant_baselines"]["2"]["mean_interval_violation"] == 0
