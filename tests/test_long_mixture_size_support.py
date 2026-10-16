import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_long_mixture_size_support")


def test_full_pool_sizes_distinguish_appearances_and_raw_words(module):
    states = [((), 1), ((1,) * 385, 2), ((33,), 34), ((1,) * 385, 2)]
    result = module.sizes(states)
    assert result["appearances"] == 4 and result["distinct_raw_representations"] == 3
    assert result["maximum_word_length"] == 385 and result["maximum_strands"] == 34
    assert result["at_least_385_letters"] == 2 and result["at_least_34_strands"] == 1
    assert result["length_histogram"] == {"0": 1, "1": 1, "385": 2}
    assert (
        result["raw_representation_set_sha256"]
        == module.sizes(list(reversed(states)))["raw_representation_set_sha256"]
    )


@pytest.mark.parametrize("states", [[], [((0,), 2)], [((2,), 2)], [((1,), 1)], [((), 0)]])
def test_invalid_pool_refused(module, states):
    with pytest.raises(ValueError):
        module.sizes(states)
