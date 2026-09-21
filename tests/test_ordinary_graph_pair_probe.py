import importlib
import struct
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("probe_ordinary_graph_pairs")


def test_replay_requires_literal_cc_not_free_mirror_or_rotation(probe):
    a = ((1, 2, 1), 3)
    assert probe.replay_cc(a, ((1, -2, 1), 3)) == 1
    assert probe.replay_cc(((1, -2, 1), 3), a) == 1
    for b in (((-1, -2, -1), 3), ((2, 1, 1), 3), ((1, 2, 1), 4), a):
        with pytest.raises(ValueError):
            probe.replay_cc(a, b)


def test_codec_flags_and_alphabet_fail_closed(probe):
    encoded = b"UKB0" + bytes([0, 0]) + struct.pack("<HI3h", 3, 3, 1, 2, -1)
    assert probe.ordinary(encoded) == ((1, 2, -1), 3)
    for flag in (1, 2, 255):
        with pytest.raises(ValueError, match="without flags"):
            probe.ordinary(encoded[:5] + bytes([flag]) + encoded[6:])
    with pytest.raises(ValueError, match="invalid ordinary"):
        probe.ordinary(b"UKB0" + bytes([0, 0]) + struct.pack("<HIh", 2, 1, 2))


def test_fixed_sampling_distinct_neighbors_reproducible_not_exact_two(probe):
    edges = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    config = {"seed": 41, "single_paths": 2, "double_paths": 2}
    selected = probe.select_paths(edges, config)
    assert selected == probe.select_paths(edges, config)
    assert len(selected) == 4
    assert {p["representations"][1] for p in selected[2:]} == {2, 3}
    assert all(len(set(p["representations"])) == 3 for p in selected[2:])
    with pytest.raises(ValueError, match="not enough"):
        probe.select_paths(edges, {**config, "double_paths": 3})
