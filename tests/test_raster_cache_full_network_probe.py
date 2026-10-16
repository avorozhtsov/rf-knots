import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("weight", [10, 1000])
def test_fixtures_cover_all_moves_with_legal_candidates(monkeypatch, weight):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("probe_raster_cache_full_network")
    bridge = importlib.import_module("pretrained_raster_policy")
    rows = module.fixtures(weight)
    bridge.validate_requests([request for _, request in rows])
    assert len(rows) == 17
    assert len({label for label, _ in rows}) == 17
    assert {label for label, _ in rows if label.startswith("after-")} == {
        f"after-{name}" for name in module.KIND_NAMES
    }
    for label, (instance, state, actions) in rows:
        kinds = {instance.spec.decode(a)[0] for a in actions}
        if label == "all-kinds":
            assert kinds == set(range(9)) - {module.PASS}
        if label in ("pass-only", "after-PASS"):
            assert kinds == {module.PASS}
        assert instance.cc_weight == weight
        assert state.used_moves == int(label.startswith("after-"))
