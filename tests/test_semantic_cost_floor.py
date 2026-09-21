import importlib
from pathlib import Path

import pytest

from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    REDUCE,
    STABILIZE_NEG,
    STABILIZE_POS,
    ActionSpec,
)
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_semantic_cost_floor")


def test_exact_trefoil_floor_and_unknown_interval_are_distinct(audit):
    spec = ActionSpec(8, 4)
    witness = UnknotWitness.from_actions(
        (1, 1, 1),
        2,
        spec,
        [spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)],
    )
    row = audit.witness_identity(witness)
    assert row["q0"] == 2 and row["excess"] == 0
    bounds = audit.reference_bounds(row, 1, 64)
    assert bounds["10"]["reference_certifiably_optimal"]
    assert bounds["10"]["certified_lower_cost"] == 13
    assert not audit.reference_bounds(row, 0, 64)["10"]["reference_certifiably_optimal"]
    with pytest.raises(ValueError, match="lower bound"):
        audit.reference_bounds(row, 2, 64)


@pytest.mark.parametrize("stabilize", [STABILIZE_POS, STABILIZE_NEG])
def test_growth_has_the_exact_accounted_cost(audit, stabilize):
    spec = ActionSpec(10, 4)
    actions = [
        spec.encode(stabilize),
        spec.encode(DESTABILIZE),
        spec.encode(INSERT, 0, 1, 1),
        spec.encode(REDUCE, 0),
        spec.encode(DESTABILIZE),
    ]
    witness = UnknotWitness.from_actions((1,), 2, spec, actions)
    row = audit.witness_identity(witness)
    assert row["moves"] == 5 and row["q0"] == 1 and row["excess"] == 4
    assert row["prefixes_checked"] == 5


def test_braid_and_commute_excess_is_not_claimed_globally_unnecessary(audit):
    spec = ActionSpec(10, 5)
    witness = UnknotWitness.from_actions(
        (1, 2, 1, 2),
        3,
        spec,
        [
            spec.encode(BRAID, 0),
            spec.encode(CROSSING_CHANGE, 2),
            spec.encode(REDUCE, 2),
            spec.encode(DESTABILIZE),
            spec.encode(DESTABILIZE),
        ],
    )
    row = audit.witness_identity(witness)
    assert row["q0"] == 3 and row["cc"] == 1 and row["excess"] == 1
    other = UnknotWitness.from_actions(
        (1, 3, 2),
        4,
        spec,
        [
            spec.encode(COMMUTE, 0),
            spec.encode(DESTABILIZE),
            spec.encode(DESTABILIZE),
            spec.encode(DESTABILIZE),
        ],
    )
    assert audit.witness_identity(other)["excess"] == 1


def test_lower_cc_can_allow_more_moves_without_exceeding_reference_cost(audit):
    row = {"q0": 100, "cc": 2, "moves": 102, "excess": 0}
    bounds = audit.reference_bounds(row, 1, 102)
    assert bounds["10"]["envelope_beyond_current_cap"] == 10
    assert bounds["1000"]["envelope_beyond_current_cap"] == 1000
    assert not bounds["1000"]["reference_certifiably_optimal"]
