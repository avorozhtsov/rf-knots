import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_resource_search")


def test_resume_binds_feature_stage_prior_and_all_roots(module, tmp_path):
    roots = [{"condition": i} for i in range(128)]
    path = tmp_path / "roots.json"
    path.write_text(json.dumps(roots))
    contract = dict(config=module.CONFIG, arm="units", stage="final", prior="distance")
    state = dict(
        schema=module.SCHEMA,
        contract=contract,
        contract_sha256=module.digest(contract),
        status="evaluating",
        value_head_used=False,
        next_condition=0,
        roots_sha256=module.sha(path),
        shards=[],
    )
    module.validate_resume(state, contract, roots, tmp_path)
    for key, value in (
        ("status", "completed"),
        ("next_condition", 1),
        ("value_head_used", True),
        ("contract_sha256", "different"),
    ):
        with pytest.raises(ValueError):
            module.validate_resume({**state, key: value}, contract, roots, tmp_path)
    for key, value in (("arm", "raw"), ("stage", "initial"), ("prior", "uniform")):
        with pytest.raises(ValueError):
            module.validate_resume(state, {**contract, key: value}, roots, tmp_path)
    path.write_text(json.dumps(roots[:-1]))
    with pytest.raises(ValueError, match="roots changed"):
        module.validate_resume(state, contract, roots, tmp_path)
