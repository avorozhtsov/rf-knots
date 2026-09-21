import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def panel(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/build_common_block_gordian_panel.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("common_panel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adequacy_requires_counts_groups_and_all_exact_classes(panel):
    rows = [{"left_fingerprint": [2 * i + 1, 2], "interval": {"lower": i % 3, "upper": i % 3}}
            for i in range(32)]
    assert panel.adequate(rows)
    assert not panel.adequate(rows[:31])
    assert not panel.adequate([{**r, "left_fingerprint": [3, 2]} for r in rows])
    assert not panel.adequate([{**r, "interval": {"lower": 0, "upper": 2}} for r in rows])
