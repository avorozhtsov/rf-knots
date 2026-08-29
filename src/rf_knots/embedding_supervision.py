"""Versioned, read-only UnknotDB supervision contract for RF Knots.

The proof graph owns labels and provenance. RF Knots consumes only this compact
sidecar interface and never reaches into proof-graph implementation tables.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SCHEMAS = {
    "unknotdb-embedding-pairs-v2",
    "unknotdb-embedding-pairs-v3",
}


@dataclass(frozen=True)
class EmbeddingRepresentation:
    representation_id: int
    word: tuple[int, ...]
    strands: int
    knot_id: str | None
    split: str


@dataclass(frozen=True)
class DistanceInterval:
    metric: str
    left_representation: int
    right_representation: int
    lower: float
    upper: float
    witness_semantic_length: int
    split: str
    provenance: str = ""

    @property
    def exact(self) -> bool:
        return self.lower == self.upper


def decode_ukb0(
    representation_id: int, encoded: bytes, knot_id: str | None, split: str
) -> EmbeddingRepresentation:
    """Decode the exact physical UKB0 braid representation used by UnknotDB."""
    if len(encoded) < 12 or encoded[:4] != b"UKB0" or encoded[4] != 0:
        raise ValueError("unsupported UKB0 representation")
    strands = struct.unpack_from("<H", encoded, 6)[0]
    length = struct.unpack_from("<I", encoded, 8)[0]
    if len(encoded) != 12 + 2 * length:
        raise ValueError("UKB0 length mismatch")
    word = struct.unpack_from(f"<{length}h", encoded, 12) if length else ()
    return EmbeddingRepresentation(
        representation_id, tuple(word), strands, knot_id, split
    )


class UnknotDBEmbeddingSidecar:
    """Read-only loader that keeps training and excluded witnesses separate."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._connection = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
        )
        metadata = dict(self._connection.execute("SELECT key,value FROM meta"))
        schema = metadata.get("schema")
        if schema not in SUPPORTED_SCHEMAS:
            self._connection.close()
            raise ValueError(f"unsupported UnknotDB embedding schema {schema!r}")
        self.metadata = metadata

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> UnknotDBEmbeddingSidecar:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def representation(self, representation_id: int) -> EmbeddingRepresentation:
        row = self._connection.execute(
            "SELECT encoding,knot_id,split FROM representations "
            "WHERE representation_id=?",
            (representation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(representation_id)
        return decode_ukb0(representation_id, bytes(row[0]), row[1], str(row[2]))

    def pairs(self, split: str) -> Iterator[DistanceInterval]:
        """Yield eligible pairs. Cross-split witnesses are never returned here."""
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split {split!r}")
        provenance = (
            ",provenance"
            if self.metadata["schema"] == "unknotdb-embedding-pairs-v3"
            else ""
        )
        rows = self._connection.execute(
            "SELECT metric,left_representation,right_representation,distance_lower,"
            f"distance_upper,witness_semantic_length,split{provenance} FROM pairs "
            "WHERE split=? ORDER BY pair_id",
            (split,),
        )
        for row in rows:
            yield DistanceInterval(
                str(row[0]),
                int(row[1]),
                int(row[2]),
                float(row[3]),
                float(row[4]),
                int(row[5]),
                str(row[6]),
                str(row[7]) if len(row) > 7 else "",
            )

    def representation_ids_by_domain(
        self,
        *,
        split: str,
        edge_interior: bool | None = None,
        stopping_point: bool | None = None,
        mirror_image: bool | None = None,
    ) -> tuple[int, ...]:
        """Select explicit replay domains from schema v3 sidecars."""
        if self.metadata["schema"] != "unknotdb-embedding-pairs-v3":
            raise ValueError("representation domains require embedding schema v3")
        clauses = ["r.split=?"]
        values: list[object] = [split]
        for column, value in (
            ("is_edge_interior", edge_interior),
            ("is_stopping_point", stopping_point),
            ("is_mirror_image", mirror_image),
        ):
            if value is not None:
                clauses.append(f"d.{column}=?")
                values.append(int(value))
        rows = self._connection.execute(
            "SELECT r.representation_id FROM representations r "
            "JOIN representation_domains d USING(representation_id) WHERE "
            + " AND ".join(clauses)
            + " ORDER BY r.representation_id",
            values,
        )
        return tuple(int(row[0]) for row in rows)

    def excluded_evaluation_witnesses(self) -> Iterator[DistanceInterval]:
        """Yield audit-only cross-split rows; callers must not train on these."""
        rows = self._connection.execute(
            "SELECT metric,left_representation,right_representation,distance_lower,"
            "distance_upper,witness_semantic_length "
            "FROM excluded_cross_split_pairs ORDER BY metric,left_representation,"
            "right_representation"
        )
        for row in rows:
            yield DistanceInterval(
                str(row[0]),
                int(row[1]),
                int(row[2]),
                float(row[3]),
                float(row[4]),
                int(row[5]),
                "excluded_cross_split",
                str(row[6]) if len(row) > 6 else "excluded_cross_split",
            )
