import importlib.util
import math
import random
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_embedding_training.py"
SPEC = importlib.util.spec_from_file_location("embedding_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rng_state_round_trip_on_cpu() -> None:
    device = torch.device("cpu")
    random.seed(17)
    torch.manual_seed(23)
    state = MODULE.capture_rng_state(device)
    expected_python = random.random()
    expected_torch = torch.rand(4)
    random.random()
    torch.rand(4)

    assert MODULE.restore_rng_state(state, device)
    assert random.random() == expected_python
    assert torch.equal(torch.rand(4), expected_torch)


def test_legacy_checkpoint_reports_rng_state_not_restored() -> None:
    assert not MODULE.restore_rng_state({}, torch.device("cpu"))


def test_cosine_learning_rate_reaches_requested_floor() -> None:
    start = MODULE.scheduled_learning_rate(1e-4, "cosine", 0.1, 0, 11)
    middle = MODULE.scheduled_learning_rate(1e-4, "cosine", 0.1, 5, 11)
    end = MODULE.scheduled_learning_rate(1e-4, "cosine", 0.1, 10, 11)

    assert start == pytest.approx(1e-4)
    assert middle == pytest.approx(5.5e-5)
    assert end == pytest.approx(1e-5)
    assert math.isfinite(
        MODULE.compute_selection_score(
            0.1,
            {
                "group_macro_interval_violation": 0.2,
                "exact_mae": 0.3,
            },
        )
    )


def test_empty_scaler_state_is_ignored_across_devices() -> None:
    class EnabledScaler:
        def is_enabled(self) -> bool:
            return True

        def load_state_dict(self, state: dict) -> None:
            raise AssertionError("an empty scaler state must not be loaded")

    assert not MODULE.restore_scaler_state(EnabledScaler(), {})


def test_legacy_checkpoint_geometry_defaults_to_chord() -> None:
    source_config = {"width": 96, "depth": 4, "embedding_dim": 64}
    assert source_config.get("cc_geometry", "chord") == "chord"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_rng_state_round_trip_on_mps() -> None:
    device = torch.device("mps")
    torch.mps.manual_seed(29)
    state = MODULE.capture_rng_state(device)
    expected = torch.rand(4, device=device).cpu()
    torch.rand(4, device=device)

    assert MODULE.restore_rng_state(state, device)
    assert torch.equal(torch.rand(4, device=device).cpu(), expected)
