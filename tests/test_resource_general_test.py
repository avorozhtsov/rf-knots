import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("build_resource_general_test")


@pytest.mark.parametrize("index", [0, 1, 2])
def test_new_seed_real_certificate_and_repeatability(module, index):
    row = module.make_row(index)
    module.audit_row(row, index)
    assert row == module.make_row(index)
    assert row["source"] != module.construct(index)[0]
    assert 0 <= row["lower"] <= row["upper"] <= 2


@pytest.mark.parametrize("field", ["source", "fingerprints", "upper", "cyclic_rotation", "index"])
def test_rejects_changed_evidence(module, field):
    row = deepcopy(module.make_row(1))
    if field == "source":
        row[field]["selected_blocks"] = []
    elif field == "fingerprints":
        row[field][0][0] += 2
    elif field == "upper":
        row[field] = 1.0
    elif field == "cyclic_rotation":
        row[field] = 0
    else:
        row[field] = 2
    with pytest.raises(ValueError):
        module.audit_row(row, 1)


def test_new_configuration_does_not_mutate_frozen74(module):
    old = importlib.import_module("build_pure_braid_distance_panel")
    assert module.CONFIG["seed"] == 20261073
    assert old.CONFIG["seed"] == 20261036
    assert module.CONFIG is not old.CONFIG
