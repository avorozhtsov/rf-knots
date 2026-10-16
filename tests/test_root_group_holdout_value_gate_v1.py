"""Fail-closed tests for the prospective root-group-disjoint value gate."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from root_group_holdout_value_gate_v1 import metrics, partition, records_for, stratum


ROOTS = Path("artifacts/diverse-cost-mcts-training-v1/roots.json")
LABELS = Path("artifacts/root-budget16-multioutput-labels-v1/labels.json")


def test_stratum_boundary():
    assert stratum({"word": [1] * 99, "cap_crossing_changes": 1}) == ("1", "<100")
    assert stratum({"word": [1] * 100, "cap_crossing_changes": 2}) == ("2", ">=100")
    assert stratum({"word": [1], "cap_crossing_changes": 8}) == ("3+", "<100")


def test_partition_and_attempt_ownership():
    roots = json.loads(ROOTS.read_text())
    labels = json.loads(LABELS.read_text())["labels"]
    selections = []
    for salt in (20261005, 20261006):
        holdout = partition(roots, salt)
        fit, test = records_for(roots, labels, holdout)
        assert len(holdout) == 32
        assert len(fit) == 1920 and len(test) == 640
        fit_indices = {r["root_index"] // 2 for r in fit}
        test_indices = {r["root_index"] // 2 for r in test}
        assert not fit_indices & test_indices
        assert len(fit_indices) == 96 and len(test_indices) == 32
        assert sorted(holdout) == partition(roots, salt)
        selections.append(holdout)
    assert selections[0] != selections[1]


def test_partition_rejects_paired_identity_change():
    roots = json.loads(ROOTS.read_text())
    roots[1]["instance"]["strands"] += 1
    with pytest.raises(ValueError, match="paired root identity"):
        partition(roots, 20261005)


def test_probability_scores():
    (brier, loss), (base_brier, base_loss) = metrics({"solved": True}, 0.8, 0.5)
    assert brier == pytest.approx(0.04)
    assert base_brier == pytest.approx(0.25)
    assert loss < base_loss
