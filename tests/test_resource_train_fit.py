import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("diagnose_resource_train_fit")


def test_fixed_sample_keeps_complete_actual_batches(module):
    assert module.UPDATES == tuple(range(0, 2048, 128))
    assert len(module.UPDATES) * 32 == 512
    assert module.MODELS == (("units", "initial"), ("units", "final"), ("raw", "final"))


def test_batch_metadata_preserves_native_priority_and_duplicate_draws(module):
    from run_strand_graph_pilot import pair_row

    short = pair_row(0, "test-fixture", ((1, 1, 1), 2), ((), 1), 1, 1, 30)
    other = pair_row(1, "test-fixture", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    rows = module.make_batch(128, {"original": [short, short], "sentinel": [other]})
    assert len(rows) == 3
    assert rows[0]["pair_sha256"] == rows[1]["pair_sha256"]
    assert [r["batch_weight"] for r in rows] == pytest.approx([8 / 17, 8 / 17, 1 / 17])
    assert [r["offset"] for r in rows] == [0, 1, 2]
    assert rows[0]["pair"]["left_word"] == (1, 1, 1)


def test_summary_retains_draw_denominator_and_constant_baselines(module):
    rows = [
        dict(
            pair_sha256="same",
            lower=1,
            upper=1,
            prediction=0.5,
            native_prediction=0.5,
            native_loss=0.25,
            violation=0.5,
            max_length=12,
            max_strands=3,
        ),
    ] * 2
    result = module.summarize(rows)
    assert result["draws"] == 2 and result["unique_raw_pairs"] == 1
    assert result["mean_interval_violation"] == 0.5
    assert result["constants"] == {"0": 1.0, "1": 0.0, "2": 1.0}
    assert result["clamp_count"] == 0


@pytest.mark.parametrize("prediction,lo,hi", [(float("nan"), 0, 1), (1, 2, 1), (1, -1, 1)])
def test_invalid_predictions_and_bounds_fail_closed(module, prediction, lo, hi):
    with pytest.raises(ValueError, match="invalid"):
        module.interval_error(prediction, lo, hi)
