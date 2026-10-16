import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_raster_mps_joint")


def test_cpu_copy_is_independent_and_strict(module):
    source = {"weights": torch.tensor([1.0]), "nested": [(2, None)]}
    clone = module.cpu_tree(source)
    module.tree_check(source, clone, exact=True)
    source["weights"].add_(1e-6)
    module.tree_check(source, clone)
    with pytest.raises(ValueError):
        module.tree_check(source, clone, exact=True)
    assert clone["weights"].item() == 1.0
    with pytest.raises(ValueError):
        module.tree_check({"a": 1}, {"b": 1})


def test_steps_rng_and_nonfinite_fail_closed(module):
    with pytest.raises(ValueError):
        module.tree_check({"step": torch.tensor(2048.0)}, {"step": torch.tensor(2048.001)})
    with pytest.raises(ValueError):
        module.tree_check(
            torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)
        )
    with pytest.raises(ValueError):
        module.tree_check(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    with pytest.raises(ValueError):
        module.tree_check({"x": float("inf")}, {"x": float("inf")})


def test_partition_and_direction(module):
    layout = [
        dict(name="a", shape=[1], block="a", start=0, stop=1, absent=False),
        dict(name="b", shape=[2], block="b", start=1, stop=3, absent=False),
    ]
    vector = torch.tensor([1.0, 2.0, 3.0])
    assert module.vectors_check(vector, vector, layout, layout)["passed"]
    assert not module.vectors_check(vector, -vector, layout, layout)["passed"]
    with pytest.raises(ValueError):
        module.vectors_check(vector, vector, layout[:1], layout[:1])
    with pytest.raises(ValueError):
        module.vectors_check(vector, vector, layout, list(reversed(layout)))
