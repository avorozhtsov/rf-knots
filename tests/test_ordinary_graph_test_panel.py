import importlib
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("build_ordinary_graph_test_panel")


def row(index, left=(11, 2), right=(3, 2)):
    return {
        "path_index": index,
        "source": {"representations": [1, 2], "pairs": [7]},
        "states": [{"word": [1, 1, 1], "strands": 2}, {"word": [-1, 1, 1], "strands": 2}],
        "fingerprints": [list(left), list(right)],
        "cc_positions": [0],
        "lower": 1,
        "upper": 1,
    }


def test_orientation_respects_membership_reference_and_exclusions(builder, monkeypatch):
    monkeypatch.setattr(
        builder, "owner", lambda fp: {11: "test", 3: "train", 7: "quarantine"}[fp[0]]
    )
    r = row(0)
    assert builder.orient(r["states"], r["fingerprints"], set()) is False
    assert builder.orient(r["states"], list(reversed(r["fingerprints"])), set()) is True
    assert builder.orient(r["states"], r["fingerprints"], {(11, 2)}) is None
    assert builder.orient(r["states"], [[11, 2], [7, 2]], set()) is None


def test_dedup_is_unordered_and_does_not_ignore_mirror_signs(builder, monkeypatch):
    monkeypatch.setattr(builder, "owner", lambda fp: "test")
    a, b, c = row(0), row(1), row(2)
    b["states"].reverse()
    b["fingerprints"].reverse()
    c["states"] = [{**s, "word": [-g for g in s["word"]]} for s in c["states"]]
    records, decisions, _, _, adequate = builder.select([a, b, c], set(), builder.CONFIG)
    assert decisions == ["accepted", "duplicate_pair", "accepted"]
    assert len(records) == 2 and adequate is False


def test_group_cap_is_global_across_categories(builder, monkeypatch):
    monkeypatch.setattr(builder, "owner", lambda fp: "test")
    a, b = row(0), row(1)
    b["states"][0]["word"] = [1] * 5
    b["upper"] = 2
    _, decisions, _, _, _ = builder.select(
        [a, b], set(), {"per_query_group": 1, "per_category": 32}
    )
    assert decisions == ["accepted", "group_cap"]


def test_explicit_cyclic_zero_preserves_sign_and_strands(builder):
    a = {"word": [1, 2, -1], "strands": 3}
    assert builder.cyclic_rotation(a, {"word": [-1, 1, 2], "strands": 3}) == 2
    assert builder.cyclic_rotation(a, {"word": [1, -1, -2], "strands": 3}) is None
    assert builder.cyclic_rotation(a, {"word": [1, 2, -1], "strands": 4}) is None
    assert builder.cyclic_rotation({"word": [], "strands": 1}, {"word": [], "strands": 1}) == 0


def test_source_sql_excludes_train_and_mixed_splits(builder, tmp_path, monkeypatch):
    import sqlite3

    from test_full_ordinary_graph_corpus import source_database

    source_database(tmp_path / "source.sqlite")
    db = sqlite3.connect(tmp_path / "source.sqlite")
    assert list(db.execute(builder.EDGE_SQL)) == []
    db.execute("UPDATE pairs SET split='test'")
    assert list(db.execute(builder.EDGE_SQL)) == []
    db.execute("UPDATE representations SET split='test'")
    assert len(list(db.execute(builder.EDGE_SQL))) == 2
    db.close()


def test_independent_auditor_reconstructs_enumeration(builder, tmp_path):
    import random
    import sqlite3

    from test_full_ordinary_graph_corpus import source_database

    audit = importlib.import_module("audit_ordinary_graph_test_panel")
    source_database(tmp_path / "source.sqlite")
    db = sqlite3.connect(tmp_path / "source.sqlite")
    db.execute("UPDATE pairs SET split='validation'")
    db.execute("UPDATE representations SET split='validation'")
    edges = list(db.execute(builder.EDGE_SQL))
    expected = builder.enumerate_paths(edges)
    random.Random(builder.CONFIG["seed"]).shuffle(expected)
    assert audit.independent_paths(db) == (2, expected)
    db.close()


@pytest.mark.parametrize("corruption", [None, "mirror", "position", "fingerprint", "bound"])
def test_independent_literal_replay_and_zero_certificate(builder, corruption):
    audit = importlib.import_module("audit_ordinary_graph_test_panel")
    states = [{"word": w, "strands": 2} for w in ([-1, 1, 1], [1, 1, 1], [1, -1, 1])]
    trace = {
        "states": states,
        "cc_positions": [0, 1],
        "lower": 0,
        "upper": 0,
        "fingerprints": [list(builder.invariants(tuple(s["word"]), s["strands"])) for s in states],
    }
    if corruption is None:
        assert audit.trace_bounds(trace) == (0, 0)
        return
    if corruption == "mirror":
        states[-1]["word"] = [-g for g in states[-1]["word"]]
    elif corruption == "position":
        trace["cc_positions"][1] = 2
    elif corruption == "fingerprint":
        trace["fingerprints"][1][1] *= -1
    else:
        trace["upper"] = 2
    with pytest.raises(ValueError):
        audit.trace_bounds(trace)
