"""Frozen validation/test quarantine consumed by embedding training."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SCHEMA = "rf-knots-embedding-validation-protocol-v0"


@dataclass(frozen=True)
class TrainingQuarantine:
    """Identities and physical representations forbidden in training."""

    knot_ids: frozenset[str]
    representation_ids: frozenset[int]
    validation_knot_ids: frozenset[str]
    test_knot_ids: frozenset[str]


def load_training_quarantine(path: str | Path) -> TrainingQuarantine:
    """Load both validation and test buffers, failing closed on schema drift."""
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    metadata = dict(connection.execute("SELECT key,value FROM meta"))
    if metadata.get("schema") != SUPPORTED_SCHEMA:
        connection.close()
        raise ValueError(
            f"unsupported embedding validation schema {metadata.get('schema')!r}"
        )
    validation = frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT knot_id FROM quarantine_knot_ids WHERE split='validation'"
        )
    )
    test = frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT knot_id FROM quarantine_knot_ids WHERE split='test'"
        )
    )
    representations = frozenset(
        int(row[0])
        for row in connection.execute(
            "SELECT DISTINCT representation_id FROM quarantine_representations"
        )
    )
    connection.close()
    if validation.intersection(test):
        raise ValueError("validation and test knot quarantines overlap")
    return TrainingQuarantine(
        validation | test,
        representations,
        validation,
        test,
    )
