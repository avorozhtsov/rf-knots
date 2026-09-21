import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rf_knots import mcts_loop as M
from rf_knots.actions import DESTABILIZE, REDUCE
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def augmenter(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/augment_mcts_training_diamonds.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("proof_based_augmentation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equivalent_reduction_positions_are_all_certified_without_cost_change(augmenter):
    from mcts_training_supervision import success_supervision

    i = M.Instance("reductions", (1, -1) * 3 + (1,), 2, 0, cap_moves=8)
    w = UnknotWitness.from_actions(
        i.word, i.strands, i.spec, [i.spec.encode(REDUCE, 0)] * 3 + [i.spec.encode(DESTABILIZE)]
    )
    data = success_supervision(i, [w])
    for section in ("positives", "preferences"):
        for row in data[section]:
            row["state"] = asdict(row["state"])
    data = json.loads(json.dumps({"condition": 0, **data}))
    original_costs = [r["known_suffix_cost"] for r in data["positives"]]
    certificates = augmenter.add_alternatives(
        i, data, [w], {augmenter.digest(w.to_dict()): w.to_dict()}, set()
    )
    root = data["positives"][0]
    assert set(root["positive_actions"]) == {i.spec.encode(REDUCE, p) for p in range(6)}
    assert len(certificates) == 9
    assert all(c["method"] == "same-successor" for c in certificates)
    assert [r["known_suffix_cost"] for r in data["positives"]] == original_costs
    assert data["preferences"] == []


def test_missing_source_witness_is_rejected(augmenter):
    i = M.Instance("one", (1,), 2, 0)
    w = UnknotWitness.from_actions(i.word, i.strands, i.spec, [i.spec.encode(DESTABILIZE)])
    with pytest.raises(ValueError, match="registry"):
        augmenter.add_alternatives(i, {"positives": [], "preferences": []}, [w], {}, set())
