"""Freeze Table 1 from Dranowski--Kabkov--Tubbenhauer (2026).

The published table is the authority for the target intervals.  When the
authors' ``unknotting.xlsx`` workbook is supplied, this script additionally
turns its PD presentation for each target into a deterministic braid-word test
instance.  The braid manifest is deliberately separate from the target table:
the representation is derived data and is not the inflated diagram or witness
used to obtain the bound in the paper.

Reproduce the committed files with::

    uv run --with snappy --with openpyxl --with pandas --with xlrd --with-editable . \
      python scripts/build_dkt2026_table1_dataset.py \
      --authors-workbook /path/to/upperbounds/data/unknotting.xlsx \
      --paper-pdf /path/to/2603.07955v3.pdf \
      --knotinfo-xls /path/to/knotinfo_data_complete.xls
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_knots.benchmarks import SCHEMA, file_sha256  # noqa: E402
from rf_knots.evidence import braid_instance_id  # noqa: E402

PAPER = {
    "title": "RL unknotter, hard unknots and unknotting number",
    "authors": ["Anne Dranowski", "Yura Kabkov", "Daniel Tubbenhauer"],
    "arxiv_id": "2603.07955",
    "arxiv_version": "v3",
    "url": "https://arxiv.org/pdf/2603.07955v3",
    "retrieved_at": "2026-08-14",
    "sha256": "79116ca7294da7d525c6f35abfc1377fc7f75401e172cdabdcbd99da502b05a2",
    "table": 1,
    "printed_page": 17,
    "caption": "Current upper bound improvements visible in the self-improving workbook.",
}

UPPERBOUNDS = {
    "repository_url": "https://github.com/dtubbenhauer/upperbounds",
    "repository_commit": "de66f29045e804931edd6d1c9735247f81ad68c1",
    "path": "data/unknotting.xlsx",
    "retrieved_at": "2026-08-14",
    "sha256": "5f58b9d6740ed6cdb63cc728abee2ba3ac54f4427fd55127d0d6d5465f3c41d3",
}

KNOTINFO = {
    "name": "KnotInfo",
    "url": "https://knotinfo.org/knotinfo_data_complete.xls",
    "retrieved_at": "2026-08-14",
    "sha256": "1829b7056eefc653f77a42acc9e471df4020e64fe93a8b6a211ca3d49fe86e7b",
}

# Left panel top-to-bottom, then right panel top-to-bottom.  Each tuple is
# (paper label, KnotInfo lower, KnotInfo upper, paper-workbook lower,
# paper-workbook upper).  The values were checked against a rendered image of
# the published table, not accepted from PDF text extraction alone.
TABLE1_ROWS = """
11a14 2 3 2 2
11a18 2 3 2 2
11a83 2 3 2 2
11n23 2 3 2 2
12a41 2 3 2 2
12a49 2 3 2 2
12a107 3 4 3 3
12a240 2 3 2 2
12a244 2 3 2 2
12a262 2 3 2 2
12a639 2 3 2 2
12a680 2 3 2 2
12n90 2 3 2 2
12n135 2 3 2 2
12n208 2 3 2 2
12n212 2 3 2 2
13a12 1 3 1 2
13a15 2 3 2 2
13a19 1 3 1 2
13a55 2 3 2 2
13a120 1 3 1 2
13a121 1 3 1 2
13a133 1 4 1 3
13a157 2 4 2 3
13a160 2 4 2 3
13a162 2 4 2 3
13a174 2 4 2 3
13a179 2 3 2 2
13a195 1 4 1 3
13a203 2 4 2 3
13a236 1 4 1 3
13a247 2 4 2 3
13a271 1 4 1 3
13a275 1 4 1 3
13a314 1 4 1 3
13a329 2 4 2 3
13a339 1 4 1 3
13a358 2 4 2 3
13a369 2 4 2 3
13a422 2 3 2 2
13a523 1 3 1 2
13a568 2 3 2 2
13a579 1 3 1 2
13a616 3 4 3 3
13a650 2 4 2 2
13a656 1 3 1 2
13a660 3 4 3 3
13a825 3 4 3 3
13a828 2 3 2 2
13a863 1 3 1 2
13a1069 1 4 1 3
13a1549 1 3 1 2
13a1656 1 3 1 2
13a1698 1 4 1 3
13a1712 1 3 1 2
13a1997 2 4 2 3
13a3147 1 3 1 2
13a3149 1 3 1 2
13a3177 1 3 1 2
13a3184 1 3 1 2
13a3212 3 4 3 3
13a3236 1 3 1 2
13a3268 1 3 1 2
13n489 1 3 1 2
13n616 3 4 3 3
13n669 3 4 3 3
13n675 2 4 2 3
13n709 3 4 3 3
13n735 2 4 2 3
13n3800 1 3 1 2
13n3801 1 3 1 2
13n4588 1 4 1 3
""".strip()


def canonical_name(label: str) -> str:
    match = re.fullmatch(r"(\d+[an])(\d+)", label)
    if match is None:
        raise ValueError(f"unexpected Table 1 knot label {label!r}")
    return f"{match.group(1)}_{match.group(2)}"


def table_rows() -> list[dict[str, Any]]:
    rows = []
    for order, line in enumerate(TABLE1_ROWS.splitlines(), start=1):
        label, old_lo, old_hi, new_lo, new_hi = line.split()
        old = [int(old_lo), int(old_hi)]
        new = [int(new_lo), int(new_hi)]
        rows.append(
            {
                "table_order": order,
                "panel": "left" if order <= 36 else "right",
                "panel_row": order if order <= 36 else order - 36,
                "paper_label": label,
                "canonical_name": canonical_name(label),
                "knotinfo_interval_at_paper": old,
                "paper_workbook_interval": new,
                "upper_bound_drop": old[1] - new[1],
                "paper_reports_exact_value": new[0] == new[1],
            }
        )
    return rows


def target_dataset() -> dict[str, Any]:
    rows = table_rows()
    return {
        "schema": "rf-knots-upper-bound-targets-v1",
        "name": "dkt2026-table1-upper-bound-targets",
        "version": "1",
        "source": PAPER,
        "scope": (
            "Exactly the 72 rows printed in Table 1. Values from later live workbooks are "
            "not substituted for the published intervals. These are knot-type upper-bound "
            "targets, not starting representations or replayable unknotting witnesses."
        ),
        "summary": {
            "target_count": len(rows),
            "exact_values_reported": sum(row["paper_reports_exact_value"] for row in rows),
            "strict_upper_bound_improvements": sum(row["upper_bound_drop"] > 0 for row in rows),
            "total_upper_bound_drop": sum(row["upper_bound_drop"] for row in rows),
        },
        "targets": rows,
    }


def parse_interval(value: Any) -> list[int]:
    if isinstance(value, int | float):
        number = int(value)
        return [number, number]
    numbers = [int(number) for number in re.findall(r"\d+", str(value))]
    if len(numbers) == 1:
        return [numbers[0], numbers[0]]
    if len(numbers) == 2:
        return numbers
    raise ValueError(f"cannot parse unknotting interval {value!r}")


def workbook_rows(path: Path, names: set[str]) -> dict[str, tuple[Any, ...]]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["Sheet1"]
    found: dict[str, tuple[Any, ...]] = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        name = row[0]
        if name not in names:
            continue
        if name in found:
            raise ValueError(f"duplicate workbook row for {name}")
        found[str(name)] = tuple(row)
    missing = sorted(names - found.keys())
    if missing:
        raise ValueError(f"workbook is missing Table 1 knots: {missing}")
    return found


def braid_manifest(target_path: Path, workbook_path: Path) -> dict[str, Any]:
    import spherogram

    targets = target_dataset()["targets"]
    by_name = workbook_rows(workbook_path, {row["canonical_name"] for row in targets})
    later_updates = []
    instances = []
    for target in targets:
        name = target["canonical_name"]
        workbook_row = by_name[name]
        current_interval = parse_interval(workbook_row[3])
        paper_interval = target["paper_workbook_interval"]
        if current_interval != paper_interval:
            if current_interval[0] < paper_interval[0] or current_interval[1] > paper_interval[1]:
                raise ValueError(
                    f"current workbook regresses the published interval for {name}: "
                    f"{paper_interval} -> {current_interval}"
                )
            later_updates.append(
                {
                    "canonical_name": name,
                    "paper_workbook_interval": paper_interval,
                    "workbook_interval_at_snapshot": current_interval,
                }
            )

        pd = ast.literal_eval(str(workbook_row[2]))
        if not isinstance(pd, list):
            raise ValueError(f"PD presentation for {name} is not a list")
        word = tuple(int(generator) for generator in spherogram.Link(pd).braid_word())
        strands = max((abs(generator) for generator in word), default=0) + 1
        pd_json = json.dumps(pd, separators=(",", ":"))
        instance = {
            "instance_id": braid_instance_id(word, strands),
            "encoding": "braid-word-v1",
            "payload": {
                "word": list(word),
                "strands": strands,
                "source_pd": pd,
                "source_pd_sha256": hashlib.sha256(pd_json.encode()).hexdigest(),
            },
            "split": "test",
            "source_id": name,
        }
        if target["paper_reports_exact_value"]:
            instance["known_unknotting_number"] = paper_interval[0]
        instances.append(instance)

    instance_ids = [instance["instance_id"] for instance in instances]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("two Table 1 knots produced the same braid instance")

    return {
        "schema": SCHEMA,
        "name": "dkt2026-table1-authors-pd-braids",
        "short_name": "DKT72-PD-v1",
        "version": "1",
        "source": {
            "target_catalog": {
                "path": target_path.name,
                "sha256": file_sha256(target_path),
            },
            "authors_workbook": {
                **UPPERBOUNDS,
                "validation": {
                    "targets_found": len(instances),
                    "intervals_equal_to_paper": len(instances) - len(later_updates),
                    "later_monotone_updates": later_updates,
                },
            },
            "representation_derivation": {
                "tool": "Spherogram",
                "version": version("spherogram"),
                "operation": "Link(source_pd).braid_word()",
            },
            "scope": (
                "One deterministic braid representative per Table 1 knot, converted from the "
                "PD presentation in the pinned authors-workbook snapshot. These are test "
                "representations, not the inflated diagrams or crossing-change witnesses used "
                "to establish the paper's bounds."
            ),
        },
        "instances": instances,
    }


def knotinfo_overlay(target_path: Path, knotinfo_path: Path) -> dict[str, Any]:
    import pandas as pd

    targets = target_dataset()["targets"]
    names = {target["canonical_name"] for target in targets}
    frame = pd.read_excel(knotinfo_path, header=0, dtype=str).fillna("")
    found = {
        str(row["name"]): row
        for row in frame.to_dict(orient="records")
        if row["name"] in names
    }
    missing = sorted(names - found.keys())
    if missing:
        raise ValueError(f"KnotInfo snapshot is missing Table 1 knots: {missing}")

    rows = []
    for target in targets:
        name = target["canonical_name"]
        current = parse_interval(found[name]["unknotting_number"])
        paper = target["paper_workbook_interval"]
        if current[0] < paper[0] or current[1] > paper[1]:
            raise ValueError(
                f"KnotInfo interval regresses Table 1 for {name}: {paper} -> {current}"
            )
        rows.append(
            {
                "table_order": target["table_order"],
                "canonical_name": name,
                "paper_workbook_interval": paper,
                "knotinfo_interval_at_snapshot": current,
                "status": "exact" if current[0] == current[1] else "open",
                "strict_upper_bound_target": current[1] - 1,
                "upper_endpoint_changed_since_paper": current[1] != paper[1],
            }
        )

    return {
        "schema": "rf-knots-upper-bound-baseline-v1",
        "name": "dkt2026-table1-knotinfo-baseline-20260814",
        "version": "1",
        "source": {
            **KNOTINFO,
            "target_catalog": {
                "path": target_path.name,
                "sha256": file_sha256(target_path),
            },
        },
        "scope": (
            "Live KnotInfo intervals for the 72 published Table 1 knots at the pinned snapshot. "
            "Use the open subset as contribution targets and the exact subset as calibration."
        ),
        "summary": {
            "target_count": len(rows),
            "exact_count": sum(row["status"] == "exact" for row in rows),
            "open_count": sum(row["status"] == "open" for row in rows),
            "upper_endpoints_changed_since_paper": sum(
                row["upper_endpoint_changed_since_paper"] for row in rows
            ),
        },
        "targets": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets-output",
        type=Path,
        default=ROOT / "benchmarks" / "dkt2026-table1-upper-bounds-v1.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "benchmarks" / "dkt2026-table1-authors-pd-braids-v1.json",
    )
    parser.add_argument(
        "--knotinfo-output",
        type=Path,
        default=ROOT / "benchmarks" / "dkt2026-table1-knotinfo-20260814.json",
    )
    parser.add_argument("--paper-pdf", type=Path)
    parser.add_argument("--authors-workbook", type=Path)
    parser.add_argument("--knotinfo-xls", type=Path)
    args = parser.parse_args()

    if args.paper_pdf is not None and file_sha256(args.paper_pdf) != PAPER["sha256"]:
        raise ValueError("paper PDF SHA-256 does not match the pinned arXiv v3 snapshot")

    write_json(args.targets_output, target_dataset())
    print(f"wrote {args.targets_output}")

    if args.authors_workbook is not None:
        if file_sha256(args.authors_workbook) != UPPERBOUNDS["sha256"]:
            raise ValueError(
                "authors workbook SHA-256 does not match the pinned repository snapshot"
            )
        write_json(
            args.manifest_output,
            braid_manifest(args.targets_output, args.authors_workbook),
        )
        print(f"wrote {args.manifest_output}")
    if args.knotinfo_xls is not None:
        if file_sha256(args.knotinfo_xls) != KNOTINFO["sha256"]:
            raise ValueError("KnotInfo XLS SHA-256 does not match the pinned snapshot")
        write_json(args.knotinfo_output, knotinfo_overlay(args.targets_output, args.knotinfo_xls))
        print(f"wrote {args.knotinfo_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
