"""Import provenance-pinned external unknotting claims without overstating replayability.

The 2026 RL-unknotter repositories publish PD-level search artifacts.  Some
records give a crossing flip and a residual PD matched to a knot with a known
upper bound, but they do not give a complete sequence from that residual PD to
the unknot.  Such records are valuable compositional upper-bound claims; they
are deliberately not assigned L10 and are not admitted for distillation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "external-unknotting-evidence-v1"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cell_value(cell: ET.Element, shared: list[str]) -> Any:
    kind = cell.attrib.get("t")
    value = cell.find(f"{NS}v")
    if value is None:
        inline = cell.find(f"{NS}is/{NS}t")
        return inline.text if inline is not None else None
    text = value.text or ""
    if kind == "s":
        return shared[int(text)]
    return text


def workbook_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """Read selected rows from the first XLSX sheet using only the ZIP/XML spec."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{NS}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{NS}t")))
        output = {}
        with archive.open("xl/worksheets/sheet1.xml") as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag != f"{NS}row":
                    continue
                cells = {
                    "".join(character for character in cell.attrib["r"] if character.isalpha()):
                    _cell_value(cell, shared)
                    for cell in element.findall(f"{NS}c")
                }
                name = cells.get("A")
                if name in wanted:
                    output[str(name)] = {
                        "knot_id": str(name),
                        "pd_presentation": (
                            json.loads(cells["C"]) if cells.get("C") else None
                        ),
                        "unknotting_number": cells.get("D"),
                    }
                element.clear()
        return output


def valid_pd(pd: Any) -> bool:
    if not isinstance(pd, list) or not pd:
        return False
    if any(not isinstance(crossing, list) or len(crossing) != 4 for crossing in pd):
        return False
    labels = [int(label) for crossing in pd for label in crossing]
    return sorted(labels) == sorted(label for label in set(labels) for _ in range(2))


def evidence_richness(row: dict[str, Any]) -> tuple[int, int, str]:
    evidence = row.get("evidence") or {}
    return (
        int("best_pd" in evidence) + int("matched_knots_jones" in evidence),
        len(evidence),
        str(row.get("_source_file", "")),
    )


def build(untangling: Path, unknotter: Path, upperbounds: Path) -> dict[str, Any]:
    output_files = sorted((upperbounds / "outputs").glob("*.jsonl"))
    claims = []
    for path in output_files:
        for row in jsonl(path):
            if row.get("improved") and row.get("evidence"):
                claims.append({**row, "_source_file": str(path.relative_to(upperbounds))})
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in claims:
        key = (str(row["knot"]), int(row["new_upper"]))
        if key not in best or evidence_richness(row) > evidence_richness(best[key]):
            best[key] = row

    workbook = upperbounds / "data" / "unknotting.xlsx"
    presentations = workbook_rows(workbook, {name for name, _ in best})
    records = []
    for (name, new_upper), row in sorted(best.items()):
        external = dict(row["evidence"])
        original = presentations.get(name, {})
        source_path = upperbounds / row["_source_file"]
        original_pd = original.get("pd_presentation")
        residual_pd = external.get("best_pd")
        record = {
            "evidence_id": hashlib.sha256(
                json.dumps(
                    {
                        "commit": commit(upperbounds),
                        "source": row["_source_file"],
                        "knot": name,
                        "new_upper": new_upper,
                        "evidence": external,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "knot_name": name,
            "old_bound": [int(row["old_lower"]), int(row["old_upper"])],
            "claimed_new_upper": new_upper,
            "classification": "external-compositional-pd-upper-bound-claim",
            "original_pd": original_pd,
            "original_pd_combinatorially_valid": valid_pd(original_pd),
            "crossing_flip_indices": external.get("flips", [external.get("flip_index")]),
            "residual_pd": residual_pd,
            "residual_pd_combinatorially_valid": valid_pd(residual_pd),
            "residual_identifications": external.get("matched_knots_jones", []),
            "external_record": external,
            "full_unknotting_path_present": False,
            "replay_verified": False,
            "l10": None,
            "l1000": None,
            "distillation_eligible": False,
            "science_heap_external_l10_eligible": False,
            "source": {
                "repository": "https://github.com/dtubbenhauer/upperbounds",
                "commit": commit(upperbounds),
                "file": row["_source_file"],
                "file_sha256": sha256(source_path),
                "workbook": "data/unknotting.xlsx",
                "workbook_sha256": sha256(workbook),
            },
        }
        records.append(record)

    repositories = [
        {
            "url": "https://github.com/annedranowski/untangling-number",
            "commit": commit(untangling),
            "imported_evidence_records": 0,
            "reason": "published sweep outputs contain no named complete unknotting paths",
        },
        {
            "url": "https://github.com/dtubbenhauer/unknotter",
            "commit": commit(unknotter),
            "imported_evidence_records": 0,
            "reason": "PD flip/reduction rows contain no complete verified unknotting paths",
        },
        {
            "url": "https://github.com/dtubbenhauer/upperbounds",
            "commit": commit(upperbounds),
            "imported_evidence_records": len(records),
            "reason": "provenance-complete compositional PD upper-bound claims",
        },
    ]
    return {
        "schema": SCHEMA,
        "policy": {
            "l10_requires_complete_replayable_path": True,
            "compositional_claims_are_not_distillation_examples": True,
            "pd_combinatorial_validation_is_not_knot_identification": True,
        },
        "repositories": repositories,
        "summary": {
            "records": len(records),
            "full_replayable_paths": sum(row["replay_verified"] for row in records),
            "l10_rankable": sum(row["l10"] is not None for row in records),
            "combinatorially_valid_original_pds": sum(
                row["original_pd_combinatorially_valid"] for row in records
            ),
            "combinatorially_valid_residual_pds": sum(
                row["residual_pd_combinatorially_valid"] for row in records
            ),
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--untangling-number", type=Path, required=True)
    parser.add_argument("--unknotter", type=Path, required=True)
    parser.add_argument("--upperbounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.untangling_number, args.unknotter, args.upperbounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
