"""Build a pinned catalogue of open KnotInfo unknotting-number gaps.

The output is a target catalogue, not evidence.  It records every knot whose
2026-08-14 KnotInfo interval is one of [1, 3], [1, 4], or [2, 4], then overlays
the repository's exact stored braid representative when one is available.

Reproduce the committed snapshot with::

    uv run --with pandas --with xlrd --with-editable . \
      python scripts/build_unknotting_gap_candidates.py \
      --knotinfo-xls tmp/knotinfo_data_complete-2026-08-14.xls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_knots.evidence import braid_instance_id  # noqa: E402

KNOTINFO = {
    "name": "KnotInfo",
    "url": "https://knotinfo.org/knotinfo_data_complete.xls",
    "retrieved_at": "2026-08-14",
    "sha256": "1829b7056eefc653f77a42acc9e471df4020e64fe93a8b6a211ca3d49fe86e7b",
}
TARGET_INTERVALS = {(1, 3), (1, 4), (2, 4)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_interval(value: Any) -> tuple[int, int]:
    numbers = [int(number) for number in re.findall(r"\d+", str(value))]
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    if len(numbers) == 2:
        return numbers[0], numbers[1]
    raise ValueError(f"cannot parse unknotting interval {value!r}")


def _crossing_number(name: str) -> int:
    match = re.match(r"^(\d+)", name)
    if match is None:
        raise ValueError(f"cannot read crossing number from {name!r}")
    return int(match.group(1))


def build_catalogue(knotinfo_xls: Path) -> dict[str, Any]:
    import pandas as pd

    table = json.loads((ROOT / "src/rf_knots/data/knot_table.json").read_text())
    stored = table["knots"]
    skipped = {str(row["name"]): row for row in table.get("skipped", [])}
    frame = pd.read_excel(knotinfo_xls, header=0, dtype=str).fillna("")

    rows: list[dict[str, Any]] = []
    for source_row in frame.to_dict(orient="records"):
        name = str(source_row.get("name", ""))
        if not re.fullmatch(r"\d+(?:a|n|_)_?\d+", name):
            continue
        try:
            interval = parse_interval(source_row.get("unknotting_number", ""))
        except ValueError:
            continue
        if interval not in TARGET_INTERVALS:
            continue

        row: dict[str, Any] = {
            "canonical_name": name,
            "crossing_number": _crossing_number(name),
            "knotinfo_interval_at_snapshot": list(interval),
            "strict_upper_bound_target": interval[1] - 1,
            "interval_width": interval[1] - interval[0],
            "external_witness_status": "no_replayable_actions_in_knotinfo_snapshot",
        }
        reference = str(source_row.get("unknotting_number_anon", "")).strip()
        if reference:
            row["knotinfo_unknotting_reference"] = reference
        local = stored.get(name)
        if local is not None:
            word = [int(value) for value in local["braid"]]
            strands = int(local["strands"])
            row.update(
                {
                    "representation_status": "available",
                    "stored_representation": {
                        "encoding": "braid-word-v1",
                        "instance_id": braid_instance_id(word, strands),
                        "word": word,
                        "strands": strands,
                        "word_length": len(word),
                    },
                }
            )
        elif name in skipped:
            row.update(
                {
                    "representation_status": "unsupported_local_strand_cap",
                    "required_strands": int(skipped[name]["strands"]),
                }
            )
        else:
            row["representation_status"] = "outside_local_crossing_table"
        rows.append(row)

    status_order = {
        "available": 0,
        "unsupported_local_strand_cap": 1,
        "outside_local_crossing_table": 2,
    }
    rows.sort(
        key=lambda row: (
            status_order[row["representation_status"]],
            -row["interval_width"],
            row.get("stored_representation", {}).get("strands", 99),
            row.get("stored_representation", {}).get("word_length", 999),
            row["crossing_number"],
            row["canonical_name"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["catalogue_rank"] = rank

    interval_counts = Counter(tuple(row["knotinfo_interval_at_snapshot"]) for row in rows)
    status_counts = Counter(row["representation_status"] for row in rows)
    reference_count = sum("knotinfo_unknotting_reference" in row for row in rows)
    return {
        "schema": "rf-knots-unknotting-gap-candidates-v1",
        "name": "knotinfo-unknotting-gap-candidates-20260814",
        "version": "1",
        "source": KNOTINFO,
        "scope": (
            "All rows in the pinned KnotInfo snapshot whose unknotting-number interval is "
            "[1,3], [1,4], or [2,4]. Stored braids are starting representations only. "
            "They do not certify an upper bound and do not enumerate all representations "
            "or crossing-change neighbours of the knot type."
        ),
        "ranking": (
            "Available local representations first; then wider interval, fewer strands, "
            "shorter stored word, crossing number, and canonical name."
        ),
        "summary": {
            "candidate_count": len(rows),
            "interval_counts": {
                f"[{lo},{hi}]": interval_counts[(lo, hi)]
                for lo, hi in sorted(TARGET_INTERVALS)
            },
            "representation_status_counts": dict(sorted(status_counts.items())),
            "knotinfo_reference_rows": reference_count,
            "replayable_external_witnesses": 0,
        },
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knotinfo-xls", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/knotinfo-unknotting-gap-candidates-20260814.json",
    )
    args = parser.parse_args()
    if file_sha256(args.knotinfo_xls) != KNOTINFO["sha256"]:
        raise ValueError("KnotInfo XLS SHA-256 does not match the pinned snapshot")
    payload = build_catalogue(args.knotinfo_xls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **payload["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
