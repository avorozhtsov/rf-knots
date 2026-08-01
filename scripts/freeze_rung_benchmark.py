"""Freeze the 23 distinct current rung knots into content-addressed splits."""

from __future__ import annotations

import json
from pathlib import Path

from rf_knots.benchmarks import BenchmarkInstance, BenchmarkManifest, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "rungs.json"
OUTPUT = ROOT / "benchmarks" / "rungs-v1.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    instances = []
    for name, knot in sorted(source["knots"].items()):
        instances.append(
            BenchmarkInstance.braid(
                knot["braid"],
                knot["strands"],
                source_id=name,
                known_unknotting_number=knot.get("unknotting_exact"),
                salt="rf-knots-rungs-v1",
            )
        )
    manifest = BenchmarkManifest(
        name="rf-knots-rungs",
        version="1",
        source={
            "path": "docs/rungs.json",
            "sha256": file_sha256(SOURCE),
            "note": "23 distinct source knots; rung repetitions are intentionally removed",
        },
        instances=tuple(instances),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.write(OUTPUT)
    counts = {split: sum(x.split == split for x in instances)
              for split in ("train", "validation", "test")}
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(instances)} instances, {counts}")


if __name__ == "__main__":
    main()
