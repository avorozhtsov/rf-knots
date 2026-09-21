import importlib.util
import json
from pathlib import Path

import pytest

from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_normalization import fast_normalize_global_view
from rf_knots.embedding_replay_corpus import invariants, path_fingerprints


@pytest.fixture
def audit(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/audit_short_u_training_supply.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("supply_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate():
    w = conjugate_witness(20261017, 16, max_strands=17, min_depth=8, max_depth=48)
    fp = invariants(w.start.word, w.start.strands)
    assert fp == (2395733, 0)
    return json.loads(json.dumps({"index": 16, "fingerprint": fp,
        "path_fingerprints": path_fingerprints(w),
        "normalized": fast_normalize_global_view(w.start.word, w.start.strands),
        "signature_interval": [0, 2], "determinant_strengthened_interval": [1, 2],
        "witness": w.to_dict()}))


def test_complete_replay_and_independent_determinant(audit, candidate):
    r = audit.validate(candidate, set())
    assert r["determinant"] == 2395733
    assert (r["lower"], r["upper"]) == (1, 2)


def test_known_train_group_rejected(audit, candidate):
    with pytest.raises(ValueError, match="root fingerprint"):
        audit.validate(candidate, {(2395733, 0)})


@pytest.mark.parametrize("field,value,match", [
    ("determinant_strengthened_interval", [2, 2], "bound contract"),
    ("normalized", [[], 1], "normalization"),
    ("path_fingerprints", [[1, 0]], "ownership"),
])
def test_forged_bound_normalization_or_path_rejected(audit, candidate, field, value, match):
    candidate[field] = value
    with pytest.raises(ValueError, match=match):
        audit.validate(candidate, set())
