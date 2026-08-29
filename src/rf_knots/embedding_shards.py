"""Compact sequential shards for high-throughput embedding training."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from rf_knots.embedding_augmentation import MaterializedSyntheticPair

SCHEMA = "rf-knots-embedding-materialized-shard-v0"


def _flatten(words: Sequence[tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flat = []
    for word in words:
        flat.extend(word)
        offsets.append(len(flat))
    return torch.tensor(flat, dtype=torch.int16), torch.tensor(offsets, dtype=torch.int64)


def write_materialized_shard(
    path: str | Path, rows: Sequence[MaterializedSyntheticPair]
) -> None:
    """Atomically store variable-length words without per-example padding."""
    if not rows:
        raise ValueError("cannot write an empty embedding shard")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    left_words, left_offsets = _flatten([row.left_word for row in rows])
    right_words, right_offsets = _flatten([row.right_word for row in rows])
    strata = sorted({row.spec.stratum for row in rows})
    stratum_ids = {value: index for index, value in enumerate(strata)}
    payload = {
        "schema": SCHEMA,
        "count": len(rows),
        "left_words": left_words,
        "left_offsets": left_offsets,
        "right_words": right_words,
        "right_offsets": right_offsets,
        "left_strands": torch.tensor([row.left_strands for row in rows], dtype=torch.int16),
        "right_strands": torch.tensor(
            [row.right_strands for row in rows], dtype=torch.int16
        ),
        "metric": torch.tensor([row.metric != "cc" for row in rows], dtype=torch.uint8),
        "lower": torch.tensor(
            [row.spec.distance_lower for row in rows], dtype=torch.float32
        ),
        "upper": torch.tensor(
            [row.spec.distance_upper for row in rows], dtype=torch.float32
        ),
        "semantic_length": torch.tensor(
            [row.witness_semantic_length for row in rows], dtype=torch.int32
        ),
        "confidence": torch.tensor(
            [row.spec.confidence for row in rows], dtype=torch.float32
        ),
        "strata": strata,
        "stratum": torch.tensor(
            [stratum_ids[row.spec.stratum] for row in rows], dtype=torch.int16
        ),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_materialized_shard(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported materialized shard schema {payload.get('schema')!r}")
    count = int(payload["count"])
    for endpoint in ("left", "right"):
        offsets = payload[f"{endpoint}_offsets"]
        if len(offsets) != count + 1 or int(offsets[-1]) != len(
            payload[f"{endpoint}_words"]
        ):
            raise ValueError(f"corrupt {endpoint} word offsets")
    return payload


def iter_shard_batches(payload: dict, batch_size: int) -> Iterator[dict]:
    """Yield padded batches with one Python loop per batch, not per data source."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    count = int(payload["count"])
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        words = {}
        lengths = {}
        for endpoint in ("left", "right"):
            flat = payload[f"{endpoint}_words"]
            offsets = payload[f"{endpoint}_offsets"]
            sequences = [
                flat[int(offsets[index]) : int(offsets[index + 1])].to(torch.long)
                for index in range(start, end)
            ]
            words[endpoint] = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True
            )
            lengths[endpoint] = torch.tensor(
                [len(sequence) for sequence in sequences], dtype=torch.long
            )
        stratum_names = payload["strata"]
        yield {
            "metric": tuple(
                "rm" if value else "cc" for value in payload["metric"][start:end]
            ),
            "stratum": tuple(
                stratum_names[int(value)] for value in payload["stratum"][start:end]
            ),
            "left_word": words["left"],
            "right_word": words["right"],
            "left_length": lengths["left"],
            "right_length": lengths["right"],
            "left_strands": payload["left_strands"][start:end].to(torch.long),
            "right_strands": payload["right_strands"][start:end].to(torch.long),
            "lower": payload["lower"][start:end],
            "upper": payload["upper"][start:end],
            "witness_semantic_length": payload["semantic_length"][start:end].to(
                torch.long
            ),
            "confidence": payload["confidence"][start:end],
        }
