import copy
import importlib
import random
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("train_long_general_pair")


def test_evaluation_matrix_and_loss_unchanged(module):
    module.checked_loss_contract()
    assert module.SEARCH_CONFIG["seed"] == 20261079
    assert module.SEARCH_CONFIG["cc_weights"] == [10, 1000]
    assert module.SEARCH_CONFIG["simulations"] == 8
    assert len(module.SEARCH_MATRIX) == 5
    assert sum(r["stage"] == "initial" for r in module.SEARCH_MATRIX) == 1
    assert {(r["arm"], r["prior"]) for r in module.SEARCH_MATRIX if r["stage"] == "final"} == {
        (a, p) for a in ("control", "long") for p in ("distance", "uniform")
    }


def test_checkpoint_roundtrip_two_rngs_and_rejections(module, tmp_path):
    payload = dict(
        config=module.CONFIG,
        contract_sha256="contract",
        arm="long",
        steps=16,
        model=dict(weight=torch.tensor([1.0])),
        rng=random.Random(4).getstate(),
        long_rng=random.Random(5).getstate(),
        relative_groups=1,
        relative_counts=[1],
    )
    entry = module.save_checkpoint(tmp_path, payload)
    restored = module.verify_checkpoint(tmp_path, entry, "contract", "long")
    assert restored["rng"] == payload["rng"] and restored["long_rng"] == payload["long_rng"]
    assert torch.equal(restored["model"]["weight"], payload["model"]["weight"])
    for changed, contract, arm in (
        ({**entry, "file": "../escape.pt"}, "contract", "long"),
        ({**entry, "sha256": "wrong"}, "contract", "long"),
        ({**entry, "steps": 32}, "contract", "long"),
        (entry, "changed", "long"),
        (entry, "contract", "control"),
    ):
        with pytest.raises(ValueError):
            module.verify_checkpoint(tmp_path, changed, contract, arm)


def test_evaluation_contract_requires_bound_data_and_complete_roots(module, tmp_path):
    # Metadata-only fixture: real mathematical certificates belong to the
    # separate required consumers, not this contract-binding unit test.
    long_data, short = tmp_path / "long", tmp_path / "short"
    long_data.mkdir()
    short.mkdir()
    for path in (
        long_data / "corpus.json",
        long_data / "consumer-audit.json",
        short / "manifest.json",
    ):
        module.atomic(path, dict(fixture=True))
    panel_path, audit_path = short / "panel.json", short / "consumer-audit.json"
    panel = dict(
        validation=[{}] * 64,
        size_shift=[{}] * 64,
        complete=True,
        lift=module.SHORT_LIFT,
        acquisition_contract=dict(config=module.SHORT_CONFIG),
    )
    module.atomic(panel_path, panel)
    required = [
        panel_path,
        short / "manifest.json",
        long_data / "corpus.json",
        long_data / "consumer-audit.json",
        Path("scripts/build_long_mixture_short_u_test.py").resolve(),
    ]
    audit = dict(
        status="passed",
        evaluation_ready=True,
        selected_roots=64,
        classes={"1": 32, "2": 32},
        attempts=module.SHORT_CONFIG["attempts"],
        panel=str(panel_path),
        panel_sha256=module.sha(panel_path),
        inputs={str(p): module.sha(p) for p in required},
    )
    module.atomic(audit_path, audit)
    contract, pins = module.evaluation_contract(long_data, short)
    assert contract["search_conditions"] == 128 and len(contract["search_matrix"]) == 5
    assert len(contract["metric_matrix"]) == 6
    assert pins[str(audit_path)] == module.sha(audit_path)
    module.atomic(long_data / "corpus.json", dict(changed=True))
    with pytest.raises(ValueError, match="required source/data"):
        module.evaluation_contract(long_data, short)
    audit["selected_roots"] = 63
    module.atomic(audit_path, audit)
    with pytest.raises(ValueError, match="64-root"):
        module.evaluation_contract(long_data, short)


def test_history_gap_and_hash_rejected(module, tmp_path):
    state = dict(completed_steps=16, metrics_history=[])
    with pytest.raises(ValueError, match="coverage"):
        module.verify_history(tmp_path, state)
    target = tmp_path / "metrics.json"
    module.atomic(target, dict(steps=16))
    state["metrics_history"] = [dict(file=target.name, sha256=module.sha(target), steps=16)]
    module.verify_history(tmp_path, state)
    module.atomic(target, dict(steps=32))
    with pytest.raises(ValueError, match="identity"):
        module.verify_history(tmp_path, state)


@pytest.fixture
def toy_runner(module, monkeypatch):
    # Test orchestration/RNG/AdamW with tiny CPU parameters, not actual MPS or
    # the real raster architecture. A real-weight149 MPS resume gate is separate.
    monkeypatch.setitem(module.CONFIG, "steps", 32)
    monkeypatch.setitem(module.CONFIG, "dev_steps", [0, 32])
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.tensor([1, 2], dtype=torch.uint8))
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda state: None)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)

    class Net(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1, bias=False)
            with torch.no_grad():
                self.weight.fill_(0.2)

        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(module, "with_resource_units", lambda net, mode: copy.deepcopy(net))
    monkeypatch.setattr(module, "MetricEmbeddingObjective", Net)
    monkeypatch.setattr(
        module,
        "reference_preferences",
        lambda net, groups, microbatch: [(g, torch.full((len(g),), -1.0)) for g in groups],
    )
    monkeypatch.setattr(
        module,
        "development",
        lambda net, obj, *args: {
            "net": float(net.weight.detach()),
            "objective": float(obj.weight.detach()),
        },
    )
    helper = importlib.import_module("long_general_training_data")

    class Sampler:
        def __init__(self, stratum):
            self.stratum = stratum

        def sample(self, rng, count):
            return [
                helper.pair_row(
                    int(rng.random() * 1000),
                    self.stratum,
                    ((1, 1, 1), 2),
                    ((-1, -1, -1), 2),
                    2,
                    2,
                    2,
                )
                for _ in range(count)
            ]

    instance = M.Instance("fixture", (1, 1, 1), 2, 1)
    request = (
        instance,
        M.SearchState(instance.word, 2, 0, 0),
        (instance.spec.encode(CROSSING_CHANGE, 0),),
    )
    data = (
        Sampler("old"),
        Sampler("suffix"),
        Sampler("graph"),
        Sampler("long"),
        [[(request, (0,))]],
        [[(request, 0, 1.0)]],
    )
    control = dict(calls=0, fail_at=None)

    def optimize(net, obj, optimizer, rows, positives, relative):
        control["calls"] += 1
        if control["calls"] == control["fail_at"]:
            raise RuntimeError("simulated preemption")
        optimizer.zero_grad(set_to_none=True)
        target = rows[0].spec.index / 1000
        loss = (net.weight - target).square().sum() + obj.weight.square().sum()
        loss = loss + torch.rand(()) * net.weight.sum()
        loss.backward()
        optimizer.step()
        return dict(metric_loss=float(loss.detach()), positive_loss=0.0)

    monkeypatch.setattr(module, "optimize", optimize)
    return Net(), data, {k: ([], []) for k in ("historical", "fresh_general")}, control


def test_full_arm_matches_checkpoint_resume_and_completed_read_is_immutable(
    module, toy_runner, tmp_path
):
    net, data, dev, control = toy_runner
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    args = (net, dict(architecture="toy"), data, dev, "contract", {})
    a = module.run_arm("long", full, *args)
    assert a["completed_steps"] == 32 and control["calls"] == 32
    control.update(calls=0, fail_at=17)
    with pytest.raises(RuntimeError, match="preemption"):
        module.run_arm("long", resumed, *args)
    state = module.read(resumed / "long" / "manifest.json")
    assert state["completed_steps"] == 16 and len(state["metrics_history"]) == 1
    control.update(calls=0, fail_at=None)
    b = module.run_arm("long", resumed, *args)
    assert control["calls"] == 16
    x = module.verify_checkpoint(full / "long", a["checkpoint"], "contract", "long")
    y = module.verify_checkpoint(resumed / "long", b["checkpoint"], "contract", "long")
    for key in (
        "rng",
        "long_rng",
        "shared_draw_digest",
        "selected_draw_digest",
        "relative_counts",
        "size_counters",
    ):
        assert x[key] == y[key]
    for key in ("torch_rng", "mps_rng"):
        assert torch.equal(x[key], y[key])
    for model in ("model", "objective"):
        assert all(torch.equal(v, y[model][k]) for k, v in x[model].items())
    for index, state in x["optimizer"]["state"].items():
        for k, value in state.items():
            assert torch.equal(value, y["optimizer"]["state"][index][k])
    assert a["size_counters"]["metric_pairs"] == 32 * 32
    assert a["size_counters"]["policy_states"] == 32 * 8 + 2
    before = {str(p): module.sha(p) for p in resumed.rglob("*") if p.is_file()}
    assert module.run_arm("long", resumed, *args) == b
    after = {str(p): module.sha(p) for p in resumed.rglob("*") if p.is_file()}
    assert before == after and control["calls"] == 16


def test_completed_arm_requires_all_dev_reports(module, toy_runner, tmp_path):
    net, data, dev, _ = toy_runner
    state = module.run_arm("control", tmp_path, net, {}, data, dev, "contract", {})
    state["development_sha256"].pop("dev-fresh_general-000032.json")
    module.atomic(tmp_path / "control" / "manifest.json", state)
    with pytest.raises(ValueError, match="DEV coverage"):
        module.verify_completed_arm(tmp_path / "control", "contract", "control")


def test_missing_long_source_does_not_create_training_output(module, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    output = tmp_path / "not-started"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--out",
            str(output),
            "--pgx-root",
            str(tmp_path),
            "--long-data",
            str(tmp_path / "absent"),
            "--short-test",
            str(tmp_path / "absent-short"),
            "--execution-gate",
            str(tmp_path / "absent-gate.json"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        module.main()
    assert not output.exists()
