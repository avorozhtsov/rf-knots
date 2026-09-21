import importlib
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_cost_resource_envelope")


def root(weight, cc=1, moves=30, length=3, strands=2):
    return {
        "condition": 0,
        "panel_index": 0,
        "stratum": "within",
        "instance": {
            "word": [1] * length,
            "strands": strands,
            "cc_weight": weight,
            "cap_crossing_changes": cc,
        },
        "reference_L10": 10 * cc + moves,
        "reference_L1000": 1000 * cc + moves,
    }


def test_same_witness_extra_cc_can_be_relevant_for_l10_but_not_l1000(diagnostic):
    a = diagnostic.envelope(root(10), 10)
    b = diagnostic.envelope(root(1000), 1000)
    assert a["extra_cc_improvement_not_ruled_out"]
    assert not b["extra_cc_improvement_not_ruled_out"]
    assert a["maximum_cc_for_strict_improvement_by_necessary_bound"] == 3
    assert b["maximum_cc_for_strict_improvement_by_necessary_bound"] == 1


def test_integer_bound_matches_exhaustive_cost_arithmetic(diagnostic):
    for weight in (10, 1000):
        for length, strands in ((0, 1), (3, 2), (40, 3), (50, 30)):
            q = max((length + 1) // 2, strands - 1)
            for cc in range(3):
                for slack in (0, 1, 11, 1002):
                    r = root(weight, cc, cc + q + slack, length, strands)
                    got = diagnostic.envelope(r, weight)
                    limit = got["maximum_cc_for_strict_improvement_by_necessary_bound"]
                    bound_cost = r[f"reference_L{weight}"]
                    # Enumerate count pairs, not asserted-realizable knot paths.
                    for c in range(110):
                        admissible = any(
                            weight * c + m < bound_cost for m in range(c + q, c + q + 4)
                        )
                        assert admissible == (c <= limit)
                    assert got["extra_cc_improvement_not_ruled_out"] == (limit > cc)


def test_equal_cost_is_not_a_strict_improvement(diagnostic):
    # Extra CC=2 gives a necessary lower cost 2*11+2 = 24, equal to reference.
    got = diagnostic.envelope(root(10, moves=14), 10)
    assert got["first_excluded_cc_path_cost_lower_bound"] == got["reference_cost"]
    assert not got["extra_cc_improvement_not_ruled_out"]


def test_reject_mismatched_weights_and_impossible_reference_counts(diagnostic):
    with pytest.raises(ValueError, match="matching"):
        diagnostic.envelope(root(10), 1000)
    with pytest.raises(ValueError, match="lower bound"):
        diagnostic.envelope(root(10, moves=1, length=50), 10)
