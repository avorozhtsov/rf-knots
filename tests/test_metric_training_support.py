import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit():
    path = Path(__file__).parents[1] / "scripts/audit_metric_training_support.py"
    spec = importlib.util.spec_from_file_location("metric_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(word=(1, 1, 1), lower=1, upper=1):
    return {"split": "train", "left_fingerprint": [3, -2], "right_fingerprint": [1, 0],
            "interval": {"metric": "cc", "lower": lower, "upper": upper,
                         "left": {"word": list(word), "strands": 2},
                         "right": {"word": [], "strands": 1}}}


def test_repeated_records_are_not_new_groups_or_states(audit):
    report = audit.describe([row(), row()])
    support = report["exact_explicit_unknot"]["1"]
    assert support["records"] == 2
    assert support["fingerprint_groups"] == support["raw_states"] == support["normalized_states"] == 1
    assert support["groups_by_count"] == [((3, 2), 2)]


def test_intervals_not_mislabeled_exact_and_sides_symmetric(audit):
    reverse = row()
    reverse["interval"]["left"], reverse["interval"]["right"] = reverse["interval"]["right"], reverse["interval"]["left"]
    reverse["left_fingerprint"], reverse["right_fingerprint"] = reverse["right_fingerprint"], reverse["left_fingerprint"]
    report = audit.describe([reverse, row(lower=0, upper=2)])
    assert report["records"] == 2
    assert report["exact_explicit_unknot"]["1"]["records"] == 1
    assert len(report["exact_explicit_unknot"]) == 1


def test_nontrain_rows_rejected(audit):
    record = row()
    record["split"] = "test"
    with pytest.raises(ValueError, match="train rows"):
        audit.describe([record])
