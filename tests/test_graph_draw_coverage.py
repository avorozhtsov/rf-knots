import importlib
import random
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_graph_draw_coverage")


def test_cycle_complete_before_repeat_and_deterministic(module):
    a = module.Cycle([1, 5, 8], 19, ["test"])
    b = module.Cycle([1, 5, 8], 19, ["test"])
    output = [a.next() for _ in range(30)]
    assert output == [b.next() for _ in range(30)]
    for start in range(0, len(output), 3):
        assert set(output[start : start + 3]) == {1, 5, 8}
    with pytest.raises(ValueError):
        module.Cycle([], 1, [])
    with pytest.raises(ValueError):
        module.Cycle([1, 1], 1, [])


def test_original_rng_consumption_and_complete_cycles(module):
    bins = [[[0], [1, 2, 3]], [[4, 5], [6]], [[7, 8, 9]]]
    choices, contexts = module.draw_traces(bins, 1000, 17)
    expected, rng = [], random.Random(17)
    for i in range(1000):
        ci = int(rng.random() * len(bins))
        gi = int(rng.random() * len(bins[ci]))
        row = int(rng.random() * len(bins[ci][gi]))
        swap = rng.random() < 0.5
        assert tuple(contexts[i]) == (ci, gi, swap)
        expected.append(bins[ci][gi][row])
    assert choices[0].tolist() == expected
    for ci, category in enumerate(bins):
        all_records = {v for group in category for v in group}
        seen = choices[2, contexts[:, 0] == ci]
        for start in range(0, len(seen) - len(all_records) + 1, len(all_records)):
            assert set(seen[start : start + len(all_records)]) == all_records
        for gi, group in enumerate(category):
            seen = choices[1, (contexts[:, 0] == ci) & (contexts[:, 1] == gi)]
            for start in range(0, len(seen) - len(group) + 1, len(group)):
                assert set(seen[start : start + len(group)]) == set(group)
    a, b = module.draw_traces(bins, 1000, 17)
    assert np.array_equal(a, choices) and np.array_equal(b, contexts)


@pytest.mark.parametrize("bins,count", [([], 1), ([[[]]], 1), ([[[0]]], 0)])
def test_invalid_support(module, bins, count):
    with pytest.raises(ValueError):
        module.draw_traces(bins, count, 1)
