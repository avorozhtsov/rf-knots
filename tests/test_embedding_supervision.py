from __future__ import annotations

import sqlite3
import struct

from rf_knots.embedding_supervision import UnknotDBEmbeddingSidecar


def _ukb0(word: tuple[int, ...], strands: int) -> bytes:
    return (
        b"UKB0"
        + bytes((0, 0))
        + struct.pack("<H", strands)
        + struct.pack("<I", len(word))
        + struct.pack(f"<{len(word)}h", *word)
    )


def test_loader_never_mixes_cross_split_rows_into_training(tmp_path) -> None:
    path = tmp_path / "pairs.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO meta VALUES ('schema','unknotdb-embedding-pairs-v2');
        CREATE TABLE representations(
          representation_id INTEGER PRIMARY KEY, encoding BLOB NOT NULL,
          knot_id TEXT, split TEXT NOT NULL
        );
        CREATE TABLE pairs(
          pair_id INTEGER PRIMARY KEY, metric TEXT NOT NULL,
          left_representation INTEGER NOT NULL, right_representation INTEGER NOT NULL,
          distance_lower INTEGER NOT NULL, distance_upper INTEGER NOT NULL,
          witness_semantic_length INTEGER NOT NULL, split TEXT NOT NULL
        );
        CREATE TABLE excluded_cross_split_pairs(
          metric TEXT NOT NULL, left_representation INTEGER NOT NULL,
          right_representation INTEGER NOT NULL, distance_lower INTEGER NOT NULL,
          distance_upper INTEGER NOT NULL, witness_semantic_length INTEGER NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO representations VALUES (?,?,?,?)",
        [(0, _ukb0((1,), 2), "knot:3_1", "train"), (1, _ukb0((-1,), 2), None, "test")],
    )
    connection.execute("INSERT INTO pairs VALUES (0,'cc',0,0,0,0,12,'train')")
    connection.execute(
        "INSERT INTO excluded_cross_split_pairs VALUES ('cc',0,1,1,1,17)"
    )
    connection.commit()
    connection.close()

    with UnknotDBEmbeddingSidecar(path) as sidecar:
        train = list(sidecar.pairs("train"))
        excluded = list(sidecar.excluded_evaluation_witnesses())
        representation = sidecar.representation(0)
    assert len(train) == 1
    assert train[0].witness_semantic_length == 12
    assert len(excluded) == 1
    assert excluded[0].split == "excluded_cross_split"
    assert representation.word == (1,)
    assert representation.strands == 2
