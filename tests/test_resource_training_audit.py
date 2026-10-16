import importlib
from copy import deepcopy
from pathlib import Path

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_resource_training_pair")


def test_shape_dtype_nonfinite_and_exact_initialization(module):
    expected = {"w": torch.tensor([1.0, 2.0])}
    module.check_tensors(expected, expected, exact=True)
    module.check_tensors({"w": torch.tensor([3.0, 4.0])}, expected)
    for actual in (
        {},
        {"w": torch.ones(3)},
        {"w": torch.ones(2).double()},
        {"w": torch.tensor([float("nan"), 2.0])},
    ):
        with pytest.raises(ValueError):
            module.check_tensors(actual, expected)
    with pytest.raises(ValueError, match="initial tensor differs"):
        module.check_tensors({"w": torch.zeros(2)}, expected, exact=True)


def optimizer_example(module):
    p, rm = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.AdamW(
        [p, rm], lr=module.CONFIG["lr"], weight_decay=module.CONFIG["weight_decay"]
    )
    p.sum().backward()
    optimizer.step()
    return {"optimizer": optimizer.state_dict()}, [("w", p), ("objective.rm_scale_raw", rm)]


def test_optimizer_has_exact_used_parameter_coverage_and_steps(module):
    payload, names = optimizer_example(module)
    module.check_optimizer(payload, names, 1)
    for mutation in ("step", "missing", "moment", "lr"):
        bad = deepcopy(payload)
        state = bad["optimizer"]["state"]
        if mutation == "step":
            state[0]["step"] = torch.tensor(0.5)
        elif mutation == "missing":
            state.clear()
        elif mutation == "moment":
            state[0]["exp_avg"][0] = float("inf")
        else:
            bad["optimizer"]["param_groups"][0]["lr"] *= 2
        with pytest.raises(ValueError):
            module.check_optimizer(bad, names, 1)
