import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_raster_mps_metric")


def test_gradient_gate_zero_and_direction(module):
    assert module.compare_vectors(torch.zeros(3), torch.zeros(3))["passed"]
    assert module.compare_vectors(torch.zeros(3), torch.zeros(3))["cosine"] is None
    a = torch.tensor([1.0, 2.0, 3.0])
    assert module.compare_vectors(a, a * 1.001)["passed"]
    assert not module.compare_vectors(a, -a)["passed"]
    assert not module.compare_vectors(a, a * 1.1)["passed"]
    with pytest.raises(ValueError):
        module.compare_vectors(a, a * float("nan"))


def test_geometry_selection_excludes_intervals_and_tie_breaks(module):
    variants = []
    for level in (1, 4, 16):
        for index, length, hi in ((3, 10, 1), (2, 10, 1), (1, 100, 2)):
            variants.append(
                dict(
                    rounds=level,
                    parent_index=index,
                    lower=1,
                    upper=hi,
                    left=dict(word=[1] * length),
                    right=dict(word=[1]),
                )
            )
    assert [r["parent_index"] for _, r in module.select_pairs(variants)] == [2, 2, 2]
