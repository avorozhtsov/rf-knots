import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("probe_fresh_ordinary_neighborhood")


def test_draw_is_reproducible_reversed_and_nonmutating(probe):
    pool = {10: {"word": [1, 2, -1, 2], "strands": 3}, 20: {"word": [1, 1, 1], "strands": 2}}
    before = deepcopy(pool)
    for i in range(32):
        selected, states, positions = probe.draw(i, pool)
        assert (selected, states, positions) == probe.draw(i, pool)
        assert states[-1] == pool[selected["reference_rep"]]
        assert len(positions) == len(set(positions)) == 1 + i % 2
        assert positions == list(reversed(selected["positions_from_reference"]))
        for a, b, position in zip(states[:-1], states[1:], positions, strict=True):
            w = a["word"].copy()
            w[position] *= -1
            assert b == {"word": w, "strands": a["strands"]}
    assert pool == before


def test_rejection_reasons_do_not_treat_source_split_as_test_ownership(probe, monkeypatch):
    monkeypatch.setattr(probe, "owner", lambda f: {1: "train", 11: "test", 13: "quarantine"}[f[0]])
    assert probe.reason([[1, 0], [11, 2]], set(), set()) == "other_owner"
    assert probe.reason([[11, 2], [1, 0]], {(11, 2)}, set()) == "optimizer_collision"
    assert probe.reason([[11, -2], [1, 0]], set(), {(11, 2)}) == "historical_query"
    assert probe.reason([[11, 2], [13, 0]], set(), set()) == "quarantine_reference"
    assert probe.reason([[11, 2], [1, 0]], set(), set()) == "eligible"


def test_source_pool_refuses_inconsistent_representation(probe):
    record = {
        "source": {"representations": [10, 20]},
        "states": [{"word": [1, 1, 1], "strands": 2}, {"word": [1], "strands": 2}],
    }
    pool = probe.source_pool({"rows": [record, deepcopy(record)]})
    assert list(pool) == [10]
    other = deepcopy(record)
    other["states"][0]["word"][0] = -1
    with pytest.raises(ValueError, match="inconsistent"):
        probe.source_pool({"rows": [record, other]})


def test_independent_draw_audit_rejects_source_and_position_changes(probe):
    audit = importlib.import_module("audit_fresh_ordinary_neighborhood")
    pool = {10: {"word": [1, 2, -1, 2], "strands": 3}, 20: {"word": [1, 1, 1], "strands": 2}}
    references = {k: (tuple(s["word"]), s["strands"]) for k, s in pool.items()}
    for i in (0, 1, 10, 1023):
        selected, states, positions = probe.draw(i, pool)
        row = {"index": i, "source": selected, "states": states, "cc_positions": positions}
        audit.check_draw(row, i, references)
        bad = deepcopy(row)
        bad["source"]["reference_rep"] = 999
        with pytest.raises(ValueError, match="source/draw"):
            audit.check_draw(bad, i, references)
        bad = deepcopy(row)
        bad["cc_positions"][0] += 1
        with pytest.raises(ValueError, match="source/draw"):
            audit.check_draw(bad, i, references)


def test_independent_key_keeps_mirror_sign_and_strand_count(probe):
    audit = importlib.import_module("audit_fresh_ordinary_neighborhood")
    a = {"word": [1, 2, -1], "strands": 3}
    assert audit.endpoint_key(a) == audit.endpoint_key({"word": [-1, 1, 2], "strands": 3})
    assert audit.endpoint_key(a) != audit.endpoint_key({"word": [-1, -2, 1], "strands": 3})
    assert audit.endpoint_key(a) != audit.endpoint_key({"word": [1, 2, -1], "strands": 4})
