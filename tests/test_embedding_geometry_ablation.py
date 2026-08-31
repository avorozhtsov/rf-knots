from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_embedding_geometry_ablation.py"
SPEC = importlib.util.spec_from_file_location("embedding_geometry_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_geometry_seeds_are_stable_and_distinct() -> None:
    first = [
        MODULE.deterministic_seed("a" * 64, geometry, 20260830) for geometry in MODULE.CC_GEOMETRIES
    ]
    second = [
        MODULE.deterministic_seed("a" * 64, geometry, 20260830) for geometry in MODULE.CC_GEOMETRIES
    ]

    assert first == second
    assert len(set(first)) == len(first)
    assert all(0 <= seed < 2**63 for seed in first)
