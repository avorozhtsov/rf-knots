import importlib
import json
import sqlite3
import struct
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("build_ordinary_graph_metric_corpus")


def source_database(path):
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE meta(key TEXT,value TEXT);
      INSERT INTO meta VALUES('schema','unknotdb-embedding-pairs-v3');
      CREATE TABLE representations(representation_id INTEGER PRIMARY KEY,encoding BLOB,split TEXT);
      CREATE TABLE pairs(pair_id INTEGER PRIMARY KEY,left_representation INTEGER,
        right_representation INTEGER,metric TEXT,split TEXT,witness_semantic_length INTEGER,
        provenance TEXT);
    """)
    for i, word in enumerate(((1, 1, 1), (-1, 1, 1), (1, -1, 1))):
        encoded = b"UKB0" + bytes([0, 0]) + struct.pack("<HI3h", 2, 3, *word)
        db.execute("INSERT INTO representations VALUES(?,?,'train')", (i, encoded))
    for pair, a, b in ((0, 0, 1), (1, 0, 2)):
        db.execute(
            "INSERT INTO pairs VALUES(?,?,?,'cc','train',1,'replayed_cc_upper_bound')", (pair, a, b)
        )
    db.commit()
    db.close()


def contract(builder, source):
    return {
        "schema": "test",
        "inputs": {str(source): builder.sha(source)},
        "config": {**builder.CONFIG, "commit_paths": 1, "single_edges": 2, "double_paths": 1},
    }


def test_enumeration_all_wedges_sorted_and_no_self_backtracking(builder):
    paths = builder.enumerate_paths([(7, 1, 2), (8, 2, 3), (9, 2, 4)])
    assert len(paths) == 6
    assert [p["representations"] for p in paths[3:]] == [[1, 2, 3], [1, 2, 4], [3, 2, 4]]
    with pytest.raises(ValueError, match="duplicate"):
        builder.enumerate_paths([(1, 0, 1), (2, 1, 0)])


def test_full_preparation_dedups_cyclic_pairs_and_certifies_exact_zero(builder, tmp_path):
    source = tmp_path / "source.sqlite"
    source_database(source)
    out = tmp_path / "out"
    spec = contract(builder, source)
    builder.prepare(out, source, spec, set())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["train_ready"] is False
    assert manifest["next_index"] == 3
    assert manifest["summary"]["paths"] == {"eligible": 3}
    assert manifest["summary"]["deduplicated_pairs"] == 2
    assert manifest["summary"]["intervals"] == {"0-0": 1, "1-1": 1}
    db = sqlite3.connect(out / "corpus.sqlite")
    assert db.execute(
        "SELECT upper_bound,witness_cc,cyclic_zero FROM paths WHERE path_index=2"
    ).fetchone() == (0, 2, 1)
    db.close()
    before = (out / "corpus.sqlite").read_bytes()
    builder.prepare(out, source, spec, set(), resume=True)
    assert (out / "corpus.sqlite").read_bytes() == before
    with pytest.raises(ValueError, match="new output"):
        builder.prepare(out, source, spec, set())


def test_interrupted_manifest_publication_recovers_committed_db(builder, tmp_path, monkeypatch):
    source = tmp_path / "source.sqlite"
    source_database(source)
    out = tmp_path / "out"
    spec = contract(builder, source)
    original = builder.atomic

    def interrupted(path, value):
        if value["next_index"] == 1:
            raise RuntimeError("simulate interruption after DB commit")
        original(path, value)

    monkeypatch.setattr(builder, "atomic", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        builder.prepare(out, source, spec, set())
    assert json.loads((out / "manifest.json").read_text())["next_index"] == 0
    monkeypatch.setattr(builder, "atomic", original)
    builder.prepare(out, source, spec, set(), resume=True)
    db = sqlite3.connect(out / "corpus.sqlite")
    assert db.execute("SELECT count(*) FROM paths").fetchone()[0] == 3
    db.close()


def test_midpoint_query_exclusion_rejects_full_path(builder, tmp_path):
    source = tmp_path / "source.sqlite"
    source_database(source)
    out = tmp_path / "out"
    builder.prepare(out, source, contract(builder, source), {(3, 2)})
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["summary"]["paths"] == {"historical_query": 3}
    assert manifest["summary"]["deduplicated_pairs"] == 0
    assert manifest["summary"]["support"]["eligible_max_length"] is None
