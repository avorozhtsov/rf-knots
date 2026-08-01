"""Frozen benchmark manifests with content-addressed instances and splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rf_knots.evidence import braid_instance_id

SCHEMA = "rf-knots-benchmark-v1"


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for(instance_id: str, salt: str = "rf-knots-v1") -> str:
    """Stable 80/10/10 split independent of input ordering."""
    bucket = int(hashlib.sha256(f"{salt}:{instance_id}".encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


@dataclass(frozen=True)
class BenchmarkInstance:
    instance_id: str
    encoding: str
    payload: dict[str, Any]
    split: str
    source_id: str | None = None
    known_unknotting_number: int | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {self.split!r}")

    @classmethod
    def braid(
        cls,
        word: Iterable[int],
        strands: int,
        *,
        source_id: str | None = None,
        known_unknotting_number: int | None = None,
        split: str | None = None,
        salt: str = "rf-knots-v1",
    ) -> BenchmarkInstance:
        word = tuple(int(x) for x in word if int(x))
        identity = braid_instance_id(word, strands)
        return cls(identity, "braid-word-v1", {"word": list(word), "strands": strands},
                   split or split_for(identity, salt), source_id, known_unknotting_number)

    def to_dict(self) -> dict[str, Any]:
        row = {"instance_id": self.instance_id, "encoding": self.encoding,
               "payload": self.payload, "split": self.split}
        if self.source_id is not None:
            row["source_id"] = self.source_id
        if self.known_unknotting_number is not None:
            row["known_unknotting_number"] = self.known_unknotting_number
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> BenchmarkInstance:
        return cls(str(row["instance_id"]), str(row["encoding"]), dict(row["payload"]),
                   str(row["split"]), row.get("source_id"), row.get("known_unknotting_number"))


@dataclass(frozen=True)
class BenchmarkManifest:
    name: str
    version: str
    source: dict[str, Any]
    instances: tuple[BenchmarkInstance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "name": self.name, "version": self.version,
                "source": self.source,
                "instances": [instance.to_dict() for instance in self.instances]}

    def write(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path | str) -> BenchmarkManifest:
        row = json.loads(Path(path).read_text())
        if row.get("schema") != SCHEMA:
            raise ValueError(f"unsupported benchmark schema {row.get('schema')!r}")
        manifest = cls(str(row["name"]), str(row["version"]), dict(row["source"]),
                       tuple(BenchmarkInstance.from_dict(x) for x in row["instances"]))
        ids = [instance.instance_id for instance in manifest.instances]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark contains duplicate instance ids")
        return manifest
