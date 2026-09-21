import importlib
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("build_joint_gordian_panel")
    monkeypatch.setattr(module, "owner", lambda fp: "test")
    return module


def row(lower=1, upper=1):
    return {
        "left_fingerprint": [31, -8],
        "right_fingerprint": [29, -6],
        "interval": {"left": {"word": [1, 1, 1], "strands": 2}, "lower": lower, "upper": upper},
    }


def test_fresh_gordian_admission_requires_two_nontrivial_endpoints(builder):
    r = row()
    assert builder.admission_reason(r, set(), set(), Counter(), Counter()) == "accepted"
    r["right_fingerprint"] = [1, 0]
    assert (
        builder.admission_reason(r, set(), set(), Counter(), Counter())
        == "insufficient_endpoint_u_lower"
    )


def test_frozen_group_class_caps_and_exclusion(builder):
    r = row()
    group = (31, 8)
    assert builder.admission_reason(r, {group}, set(), Counter(), Counter()) == "previous_group"
    assert (
        builder.admission_reason(r, set(), set(), Counter({(group, "1"): 4}), Counter())
        == "group_cap"
    )
    assert builder.admission_reason(r, set(), set(), Counter(), Counter({"1": 32})) == "class_cap"
    key = builder.canonical((1, 1, 1), 2)
    assert builder.admission_reason(r, set(), {key}, Counter(), Counter()) == "duplicate"


def test_gate_does_not_replace_missing_exact_two_with_intervals(builder):
    rows = []
    for n in range(32):
        r = row(n % 2, n % 2)
        r["left_fingerprint"] = [31 + 2 * n, -8]
        rows.append(r)
    assert not builder.adequate(rows)
    for r in rows[:4]:
        r["interval"].update(lower=0, upper=2)
    assert not builder.adequate(rows)
    for r in rows[:4]:
        r["interval"].update(lower=2, upper=2)
    assert builder.adequate(rows)


def test_exclusions_include_all_thirteen_existing_panels(builder):
    names = set(
        (*builder.EXCLUSIONS, *builder.EXTRA, "embedding-common-block-gordian-supplement-v1")
    )
    assert len(names) == 13
    assert "embedding-diverse-metric-gordian-test-v1" in names
