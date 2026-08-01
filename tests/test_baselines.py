import pytest

from rf_knots.baselines import BaselineUnavailable, run_reapr, run_regina, run_snappy


@pytest.mark.parametrize("runner", [run_snappy, run_regina])
def test_exact_empty_braid_shortcut_needs_no_optional_dependency(runner):
    result = runner((), 1)
    assert result.status == "unknot"
    assert result.output_crossings == 0


def test_reapr_reports_missing_executable(monkeypatch):
    monkeypatch.setattr("rf_knots.baselines.shutil.which", lambda _: None)
    with pytest.raises(BaselineUnavailable, match="knoodlesimplify"):
        run_reapr((1, 1, 1), 2)
