from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_embedding_rank_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("embedding_rank_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rank_arm_names_are_stable() -> None:
    arms = [MODULE.Arm(f"rank-weight-{weight:g}", weight) for weight in (0.25, 0.5, 1.0)]
    assert [arm.name for arm in arms] == [
        "rank-weight-0.25",
        "rank-weight-0.5",
        "rank-weight-1",
    ]
