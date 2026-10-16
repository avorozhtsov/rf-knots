import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("check_resource_search_cache")


def test_cache_parity_ignores_only_physical_cache_counts(module):
    rows = [
        {"solved": True, "episodes": [{"continuation": {"actions": [1], "probabilities": [0.5]}}]}
    ]
    counts = dict(conditions=1, tree_embedding_evaluations=2)
    assert module.compare_paths(rows, counts, rows, {**counts, "cache_hits": 10}) == 0
    with pytest.raises(ValueError, match="logical work"):
        module.compare_paths(rows, counts, rows, {**counts, "tree_embedding_evaluations": 0})
    bad = deepcopy(rows)
    bad[0]["episodes"][0]["continuation"]["actions"] = [2]
    with pytest.raises(ValueError, match="discrete search"):
        module.compare_paths(rows, counts, bad, counts)
    bad = deepcopy(rows)
    bad[0]["episodes"][0]["continuation"]["probabilities"] = [0.9]
    with pytest.raises(AssertionError):
        module.compare_paths(rows, counts, bad, counts)
