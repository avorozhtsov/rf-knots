import importlib
import json
import random
from copy import deepcopy
from pathlib import Path

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_long_general_search")


def inputs(module):
    roots = [dict(condition=i, instance={"cap_moves": 32 + i}) for i in range(128)]
    contract = dict(config=module.CONFIG, arm="long", stage="final", prior="distance", inputs={})
    return roots, contract


def fake_batch(batch, net, objective, begin, config, *, prior, cached):
    assert cached and config["horizon"] == max(r["instance"]["cap_moves"] for r in batch)
    rng = random.Random(config["seed"] + begin)
    return [dict(condition=r["condition"], draw=rng.random()) for r in batch], {
        "conditions": len(batch)
    }


def test_declared_matrix_has_only_five_searches(module):
    for row in module.SEARCH_MATRIX:
        module.declared(**row)
    for row in (
        ("long", "initial", "distance"),
        ("control", "initial", "uniform"),
        ("raw", "final", "distance"),
    ):
        with pytest.raises(ValueError):
            module.declared(*row)


def test_saved_batch_resume_and_completed_reentry_are_exact(module, monkeypatch, tmp_path):
    roots, contract = inputs(module)
    monkeypatch.setattr(module, "run_batch", fake_batch)
    full = tmp_path / "full"
    module.run_search(full, roots, None, None, contract, {})
    interrupted = tmp_path / "resumed"

    def stop(batch, net, objective, begin, config, **kwargs):
        if begin == 8:
            raise RuntimeError("simulated interruption")
        return fake_batch(batch, net, objective, begin, config, **kwargs)

    monkeypatch.setattr(module, "run_batch", stop)
    with pytest.raises(RuntimeError, match="simulated"):
        module.run_search(interrupted, roots, None, None, contract, {})
    assert module.read(interrupted / "manifest.json")["next_condition"] == 8
    seen = []

    def resumed(batch, net, objective, begin, config, **kwargs):
        seen.append(begin)
        return fake_batch(batch, net, objective, begin, config, **kwargs)

    monkeypatch.setattr(module, "run_batch", resumed)
    end = module.run_search(interrupted, roots, None, None, contract, {}, resume=True)
    assert seen == list(range(8, 128, 8))
    assert end["status"] == "completed" and end["counts"] == {"conditions": 128}
    assert module.committed_rows(end, interrupted) == module.committed_rows(
        module.read(full / "manifest.json"), full
    )
    before = {p.name: p.read_bytes() for p in interrupted.iterdir()}
    module.run_search(interrupted, roots, None, None, contract, {}, resume=True)
    assert before == {p.name: p.read_bytes() for p in interrupted.iterdir()}
    (interrupted / "conditions-00000.json").write_text("[]")
    with pytest.raises(ValueError, match="shard changed"):
        module.run_search(interrupted, roots, None, None, contract, {}, resume=True)


@pytest.mark.parametrize("tamper", [False, True])
def test_uncommitted_shard_is_reused_only_when_equal(module, monkeypatch, tmp_path, tamper):
    roots, contract = inputs(module)
    monkeypatch.setattr(module, "run_batch", fake_batch)
    checks = []
    original = module.check_pins

    def stop(pins):
        checks.append(1)
        if len(checks) == 2:
            raise RuntimeError("crash after shard before manifest")
        original(pins)

    monkeypatch.setattr(module, "check_pins", stop)
    root = tmp_path / "search"
    with pytest.raises(RuntimeError, match="after shard"):
        module.run_search(root, roots, None, None, contract, {})
    assert module.read(root / "manifest.json")["next_condition"] == 0
    orphan = root / "conditions-00000.json"
    assert orphan.exists()
    if tamper:
        orphan.write_text("[]")
    before = orphan.read_bytes()
    monkeypatch.setattr(module, "check_pins", original)
    if tamper:
        with pytest.raises(ValueError, match="uncommitted"):
            module.run_search(root, roots, None, None, contract, {}, resume=True)
    else:
        assert (
            module.run_search(root, roots, None, None, contract, {}, resume=True)["status"]
            == "completed"
        )
    assert orphan.read_bytes() == before


def cache_fixture(module, tmp_path):
    pins, models = {}, {}
    for arm, stage in module.MODELS:
        p = tmp_path / f"{arm}-{stage}.pt"
        p.write_bytes(p.name.encode())
        pins[str(p)] = module.sha(p)
        models[f"{arm}-{stage}"] = dict(
            status="passed",
            checkpoint=str(p),
            checkpoint_sha256=module.sha(p),
            weights_unchanged=True,
            cache_capacity=512,
            forward_cases=102,
            searches=[
                dict(prior=prior, seed=seed)
                for prior in ("distance", "uniform")
                for seed in module.SEEDS
            ],
            shared_initial_identity=dict(exact_tensors=True, encoder_equal=True, policy_equal=True),
        )
    source = Path("scripts/check_long_general_search_cache.py").resolve()
    pins[str(source)] = module.sha(source)
    return dict(
        schema=module.CACHE_SCHEMA,
        status="passed",
        models=models,
        inputs=pins,
        training_contract_sha256="contract",
        cache_capacity=512,
        device="cpu",
        threads=1,
        torch_version=torch.__version__,
    )


@pytest.mark.parametrize(
    "mutation", [None, "missing_model", "forward", "search", "identity", "hash", "contract"]
)
def test_cache_gate_requires_every_model_and_fixture(module, tmp_path, mutation):
    gate = cache_fixture(module, tmp_path)
    if mutation == "missing_model":
        gate["models"].pop("control-final")
    elif mutation == "forward":
        gate["models"]["control-final"]["forward_cases"] = 101
    elif mutation == "search":
        gate["models"]["long-final"]["searches"].pop()
    elif mutation == "identity":
        gate["models"]["control-initial"]["shared_initial_identity"]["policy_equal"] = False
    elif mutation == "hash":
        gate["models"]["long-final"]["checkpoint_sha256"] = "wrong"
    elif mutation == "contract":
        gate["training_contract_sha256"] = "wrong"
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    checkpoint = tmp_path / "long-final.pt"
    if mutation:
        with pytest.raises(ValueError):
            module.require_cache_gate(path, checkpoint, "long", "final", {}, "contract")
    else:
        pins = {}
        module.require_cache_gate(path, checkpoint, "long", "final", pins, "contract")
        assert pins[str(path)] == module.sha(path)


def test_resume_rejects_changed_contract_roots_and_missing_coverage(module, tmp_path):
    roots, contract = inputs(module)
    path = tmp_path / "roots.json"
    path.write_text(json.dumps(roots))
    state = dict(
        schema=module.SCHEMA,
        contract=contract,
        contract_sha256=module.digest(contract),
        status="evaluating",
        value_head_used=False,
        next_condition=0,
        roots_sha256=module.sha(path),
        shards=[],
        counts={},
        seconds_committed_batches=0.0,
    )
    module.validate_resume(state, contract, roots, tmp_path)
    for key, value in (
        ("status", "completed"),
        ("next_condition", 1),
        ("value_head_used", True),
        ("seconds_committed_batches", float("nan")),
        ("counts", {"conditions": 8}),
        ("counts", {"episodes": -1}),
    ):
        with pytest.raises(ValueError):
            module.validate_resume({**state, key: value}, contract, roots, tmp_path)
    wrong = deepcopy(contract)
    wrong["config"]["seed"] += 1
    with pytest.raises(ValueError):
        module.validate_resume(state, wrong, roots, tmp_path)
    path.write_text(json.dumps(roots[:-1]))
    with pytest.raises(ValueError, match="roots changed"):
        module.validate_resume(state, contract, roots, tmp_path)
