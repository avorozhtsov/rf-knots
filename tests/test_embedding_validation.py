import sqlite3

import pytest

from rf_knots.embedding_validation import load_training_quarantine


def test_loads_disjoint_validation_and_test_quarantine(tmp_path) -> None:
    path = tmp_path / "protocol.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO meta VALUES('schema','rf-knots-embedding-validation-protocol-v0');
        CREATE TABLE quarantine_knot_ids(
          split TEXT,knot_id TEXT,reason TEXT,cc_radius INTEGER
        );
        CREATE TABLE quarantine_representations(
          split TEXT,representation_id INTEGER,reason TEXT,cc_radius INTEGER
        );
        INSERT INTO quarantine_knot_ids VALUES('validation','knot:3_1','anchor',0);
        INSERT INTO quarantine_knot_ids VALUES('test','knot:4_1','anchor',0);
        INSERT INTO quarantine_representations VALUES('validation',10,'anchor',0);
        INSERT INTO quarantine_representations VALUES('test',20,'anchor',0);
        """
    )
    connection.commit()
    connection.close()
    quarantine = load_training_quarantine(path)
    assert quarantine.knot_ids == {"knot:3_1", "knot:4_1"}
    assert quarantine.representation_ids == {10, 20}


def test_rejects_identity_overlap(tmp_path) -> None:
    path = tmp_path / "protocol.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO meta VALUES('schema','rf-knots-embedding-validation-protocol-v0');
        CREATE TABLE quarantine_knot_ids(
          split TEXT,knot_id TEXT,reason TEXT,cc_radius INTEGER
        );
        CREATE TABLE quarantine_representations(
          split TEXT,representation_id INTEGER,reason TEXT,cc_radius INTEGER
        );
        INSERT INTO quarantine_knot_ids VALUES('validation','knot:3_1','anchor',0);
        INSERT INTO quarantine_knot_ids VALUES('test','knot:3_1','buffer',1);
        """
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="overlap"):
        load_training_quarantine(path)
