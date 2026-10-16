import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def attribute(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("report_raster_resource_gradients").attribute


def test_single_lambda_column_and_coordinate_change(attribute):
    w = torch.zeros(64, 208)
    w[0, 196] = 1000
    b = torch.zeros(64)
    b[0] = 1
    result = attribute(w, b, 1_000_000.0)
    assert result["raw_lambda_share_linear"] == 1
    assert result["raw_lambda_share_whole"] == 1
    assert result["coordinate_only_norm_ratio"] == 0.001
    assert result["lambda_1000_bias_identity_max_abs_error"] == 0
    assert sum(result["column_squared_norms"].values()) == 1_000_000.0


def test_feature_partition_and_full_network_denominator(attribute):
    w = torch.ones(64, 208)
    result = attribute(w, torch.zeros(64), 2 * w.numel())
    assert sum(result["column_squared_norms"].values()) == w.numel()
    assert result["raw_resources_share_linear"] == pytest.approx(5 / 208)
    assert result["raw_lambda_share_whole"] == pytest.approx(1 / 416)


def test_zero_and_invalid_inputs(attribute):
    result = attribute(torch.zeros(64, 208), torch.zeros(64), 0.0)
    assert result["raw_lambda_share_linear"] is None
    assert result["coordinate_only_norm_ratio"] is None
    with pytest.raises(ValueError):
        attribute(torch.zeros(64, 207), torch.zeros(64), 0.0)
    with pytest.raises(ValueError):
        attribute(torch.ones(64, 208), torch.zeros(64), 0.0)
