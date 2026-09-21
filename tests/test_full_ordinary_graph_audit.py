import importlib
import sqlite3
from pathlib import Path

import pytest
from test_full_ordinary_graph_corpus import contract, source_database


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return (
        importlib.import_module("build_ordinary_graph_metric_corpus"),
        importlib.import_module("audit_full_ordinary_graph_corpus"),
    )


@pytest.mark.parametrize("corruption", [None, "position", "bound", "dedup", "canonical", "owner"])
def test_full_audit_rejects_corruptions(modules, tmp_path, corruption):
    builder, audit = modules
    source_path = tmp_path / "source.sqlite"
    source_database(source_path)
    out = tmp_path / "out"
    spec = {**contract(builder, source_path), "excluded_groups": []}
    builder.prepare(out, source_path, spec, set())
    source, db = sqlite3.connect(source_path), sqlite3.connect(out / "corpus.sqlite")
    if corruption is None:
        result = audit.inspect_database(db, source, spec)
        assert result["deduplicated_pairs"] == 2
        assert result["intervals"] == {"1-1": 1, "0-0": 1}
    else:
        statements = {
            "position": "UPDATE paths SET cc_positions='[2]' WHERE path_index=0",
            "bound": "UPDATE paths SET upper_bound=2 WHERE path_index=0",
            "dedup": "DELETE FROM training_pairs WHERE upper_bound=0",
            "canonical": "UPDATE states SET canonical='mirror-free' WHERE rep=0",
            "owner": "UPDATE states SET owner='test' WHERE rep=0",
        }
        db.execute(statements[corruption])
        db.commit()
        with pytest.raises(ValueError):
            audit.inspect_database(db, source, spec)
    db.close()
    source.close()
