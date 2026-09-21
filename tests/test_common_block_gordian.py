import importlib.util
from pathlib import Path

import pytest

from rf_knots.embedding_replay_corpus import invariants


@pytest.fixture
def construction():
    path = Path(__file__).parents[1] / "scripts/common_block_gordian.py"
    spec = importlib.util.spec_from_file_location("common_block", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index", range(9))
def test_short_prefix_between_nontrivial_endpoints_with_full_replay(construction, index):
    w, prefix, info = construction.make_pair(20261021, index)
    terminal = w.verify()
    assert terminal.word == () and terminal.strands == 1
    endpoint = w.steps[prefix - 1].after
    assert endpoint.word
    upper = sum(s.action.kind == "CROSSING_CHANGE" for s in w.steps[:prefix])
    assert upper == index % 3 == info["attempt_kind"]
    a, b = invariants(w.start.word, w.start.strands), invariants(endpoint.word, endpoint.strands)
    assert min(abs(a[1]), abs(b[1])) // 2 >= 3
    assert abs(a[1] - b[1]) // 2 <= upper
    assert w.crossing_changes > upper  # Full witness also unknots the common block.


def test_construction_deterministic(construction):
    assert construction.make_pair(20261021, 2) == construction.make_pair(20261021, 2)
