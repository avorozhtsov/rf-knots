"""Published upper-bound targets stay distinct from derived braid representatives."""

import hashlib
import json
from pathlib import Path

from rf_knots.benchmarks import BenchmarkManifest
from rf_knots.evidence import braid_instance_id

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "benchmarks" / "dkt2026-table1-upper-bounds-v1.json"
BRAIDS = ROOT / "benchmarks" / "dkt2026-table1-authors-pd-braids-v1.json"
KNOTINFO = ROOT / "benchmarks" / "dkt2026-table1-knotinfo-20260814.json"


def test_table1_target_catalog_is_exact_and_provenance_bearing():
    dataset = json.loads(TARGETS.read_text())
    targets = dataset["targets"]

    assert dataset["schema"] == "rf-knots-upper-bound-targets-v1"
    assert dataset["source"] == {
        "arxiv_id": "2603.07955",
        "arxiv_version": "v3",
        "authors": ["Anne Dranowski", "Yura Kabkov", "Daniel Tubbenhauer"],
        "caption": "Current upper bound improvements visible in the self-improving workbook.",
        "printed_page": 17,
        "retrieved_at": "2026-08-14",
        "sha256": "79116ca7294da7d525c6f35abfc1377fc7f75401e172cdabdcbd99da502b05a2",
        "table": 1,
        "title": "RL unknotter, hard unknots and unknotting number",
        "url": "https://arxiv.org/pdf/2603.07955v3",
    }
    assert len(targets) == 72
    assert len({row["canonical_name"] for row in targets}) == 72
    assert [row["table_order"] for row in targets] == list(range(1, 73))
    assert all(row["knotinfo_interval_at_paper"][0]
               == row["paper_workbook_interval"][0] for row in targets)
    assert all(row["upper_bound_drop"] > 0 for row in targets)
    assert sum(row["paper_reports_exact_value"] for row in targets) == 30
    assert sum(row["upper_bound_drop"] for row in targets) == 73
    assert [row["canonical_name"] for row in targets if row["upper_bound_drop"] == 2] == [
        "13a_650"
    ]


def test_table1_braid_manifest_is_an_external_test_panel():
    targets = json.loads(TARGETS.read_text())["targets"]
    by_name = {row["canonical_name"]: row for row in targets}
    assert json.loads(BRAIDS.read_text())["short_name"] == "DKT72-PD-v1"
    manifest = BenchmarkManifest.read(BRAIDS)

    assert len(manifest.instances) == 72
    assert {instance.source_id for instance in manifest.instances} == set(by_name)
    assert all(instance.split == "test" for instance in manifest.instances)

    for instance in manifest.instances:
        assert instance.source_id is not None
        target = by_name[instance.source_id]
        word = instance.payload["word"]
        strands = instance.payload["strands"]
        source_pd = instance.payload["source_pd"]
        encoded_pd = json.dumps(source_pd, separators=(",", ":"))
        assert instance.instance_id == braid_instance_id(word, strands)
        assert instance.payload["source_pd_sha256"] == hashlib.sha256(
            encoded_pd.encode()
        ).hexdigest()
        assert len(source_pd) == int(instance.source_id[:2])
        expected_exact = (
            target["paper_workbook_interval"][0]
            if target["paper_reports_exact_value"]
            else None
        )
        assert instance.known_unknotting_number == expected_exact


def test_later_workbook_update_does_not_rewrite_the_published_table():
    manifest = json.loads(BRAIDS.read_text())
    validation = manifest["source"]["authors_workbook"]["validation"]

    assert validation == {
        "targets_found": 72,
        "intervals_equal_to_paper": 71,
        "later_monotone_updates": [
            {
                "canonical_name": "13n_489",
                "paper_workbook_interval": [1, 2],
                "workbook_interval_at_snapshot": [2, 2],
            }
        ],
    }


def test_current_knotinfo_overlay_separates_calibration_from_open_targets():
    overlay = json.loads(KNOTINFO.read_text())
    rows = overlay["targets"]

    assert overlay["schema"] == "rf-knots-upper-bound-baseline-v1"
    assert overlay["summary"] == {
        "target_count": 72,
        "exact_count": 48,
        "open_count": 24,
        "upper_endpoints_changed_since_paper": 0,
    }
    assert all(row["status"] in {"exact", "open"} for row in rows)
    assert all(
        row["strict_upper_bound_target"] == row["knotinfo_interval_at_snapshot"][1] - 1
        for row in rows
    )
    assert [
        row["canonical_name"]
        for row in rows
        if row["canonical_name"] == "13n_489"
    ] == ["13n_489"]
    row_13n489 = next(row for row in rows if row["canonical_name"] == "13n_489")
    assert row_13n489["paper_workbook_interval"] == [1, 2]
    assert row_13n489["knotinfo_interval_at_snapshot"] == [2, 2]
