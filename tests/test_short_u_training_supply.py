import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).parents[1] / "scripts/probe_short_u_training_supply.py"
    spec = importlib.util.spec_from_file_location("supply_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_cc_witness_remains_interval_without_lower_certificate(probe):
    assert probe.bounds((3, 0), 2) == ([0, 2], [1, 2])
    assert probe.bounds((3, 0), 1) == ([0, 1], [1, 1])
    assert probe.bounds((9, -4), 2) == ([2, 2], [2, 2])
    with pytest.raises(ValueError):
        probe.bounds((9, -4), 1)


def test_group_is_mirror_folded_not_identity(probe):
    assert probe.group((9, -4)) == probe.group((9, 4))


def test_empty_supply_is_reported_honestly(probe):
    assert probe.describe([]) == {"accepted": 0, "groups": 0, "classes": {},
        "max_raw_length": 0, "max_normalized_length": 0,
        "max_raw_strands": 0, "max_normalized_strands": 0}
