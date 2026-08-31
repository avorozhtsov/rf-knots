import importlib.util
import json
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_embedding_checkpoint_tournament.py"
SPEC = importlib.util.spec_from_file_location("embedding_tournament", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_branch_seed_is_reproducible_and_branch_specific() -> None:
    first = MODULE.branch_seed("abc", 1, 42, 0)
    assert first == MODULE.branch_seed("abc", 1, 42, 0)
    assert first != MODULE.branch_seed("abc", 1, 42, 1)
    assert first != MODULE.branch_seed("abc", 2, 42, 0)


def test_select_best_uses_score_then_branch() -> None:
    rows = [
        {"branch": 2, "status": "completed", "selection_score": 1.0},
        {"branch": 1, "status": "completed", "selection_score": 1.0},
        {"branch": 0, "status": "failed"},
    ]
    assert MODULE.select_best(rows)["branch"] == 1


def test_parent_lineage_prefers_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"examples_seen": 10, "lineage_examples_seen": 30}, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert MODULE.parent_lineage_examples(checkpoint, payload) == 30


def test_parent_lineage_uses_verified_legacy_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"examples_seen": 10}, checkpoint)
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": MODULE.sha256(checkpoint),
                "lineage_examples_seen": 20,
            }
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert MODULE.parent_lineage_examples(checkpoint, payload) == 20
