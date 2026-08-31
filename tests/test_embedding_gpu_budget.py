import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_embedding_gpu_budget.py"
SPEC = importlib.util.spec_from_file_location("embedding_gpu_budget", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_default_stage_one_budget_and_seeds_are_deterministic() -> None:
    arms = MODULE.stage_one_arms(500_000)

    assert len(arms) == 6
    assert sum(arm.requested_examples for arm in arms) == 3_000_000
    assert len({arm.name for arm in arms}) == len(arms)
    seed = MODULE.deterministic_seed("abc", 1, arms[0].name, 42)
    assert seed == MODULE.deterministic_seed("abc", 1, arms[0].name, 42)
    assert seed != MODULE.deterministic_seed("abc", 2, arms[0].name, 42)


def test_parent_score_requires_matching_checkpoint_sha(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"examples_seen": 10}, checkpoint)
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": MODULE.sha256(checkpoint),
                "selection_score": 0.75,
            }
        )
    )
    assert MODULE.parent_score(checkpoint) == pytest.approx(0.75)

    torch.save({"examples_seen": 11}, checkpoint)
    with pytest.raises(ValueError, match="SHA mismatch"):
        MODULE.parent_score(checkpoint)
