import importlib
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("check_long_general_mps")


def selection_fixture():
    result = []
    for band in range(3):
        for label in range(3):
            for length, strands in ((50 + band * 70, 5), (55 + band * 70, 4), (55 + band * 70, 6)):
                result.append(
                    dict(
                        index=len(result),
                        split="train",
                        source=dict(length_band=band),
                        states=[dict(word=[1] * length, strands=strands)],
                        lower=label,
                        upper=label,
                    )
                )
    result.append(
        dict(
            index=100,
            split="train",
            source=dict(length_band=0),
            states=[dict(word=[1] * 50, strands=33)],
            lower=0,
            upper=1,
        )
    )
    return result


def test_score_blind_largest_case_selection(module):
    records = selection_fixture()
    chosen = module.select_cases(records)
    assert len(chosen) == 10
    assert [name for name, _ in chosen] == [
        f"band{b}-exact{c}" for b in range(3) for c in range(3)
    ] + ["maximum-strands"]
    assert all(r["index"] % 3 == 2 for _, r in chosen[:-1])
    assert chosen[-1][1]["index"] == 100
    with pytest.raises(ValueError, match="missing TRAIN"):
        module.select_cases(
            [
                r
                for r in records
                if not (r["source"]["length_band"] == 1 and r["lower"] == r["upper"] == 2)
            ]
        )


def test_draw_period_resume_consumes_both_streams(module, monkeypatch):
    calls = []

    def draw(arm, rng, long_rng, *args):
        a, b = rng.random(), long_rng.random()
        calls.append((a, b))
        return SimpleNamespace(
            shared_digest=module.digest([a, b]), selected_digest=module.digest([arm, a, b])
        )

    monkeypatch.setattr(module, "paired_draw", draw)
    rng, lrng = random.Random(1), random.Random(2)
    data = (None,) * 6
    _, trace = module.advance_draws("long", rng, lrng, data, [], ("a", "b"))
    assert len(calls) == 16
    a, b = random.Random(), random.Random()
    a.setstate(rng.getstate())
    b.setstate(lrng.getstate())
    first, t1 = module.advance_draws("long", rng, lrng, data, [], trace)
    repeated, t2 = module.advance_draws("long", a, b, data, [], trace)
    assert t1 == t2 and vars(first) == vars(repeated)
    with pytest.raises(ValueError, match="16-draw"):
        module.advance_draws("long", rng, lrng, data, [], trace, count=15)


def test_private_saved_optimizer_replay_on_cpu_toy(module, monkeypatch, tmp_path):
    # Exercise152 serialization and draw advancement, not MPS numerical parity.
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

    helper = importlib.import_module("long_general_training_data")
    monkeypatch.setattr(module, "MetricEmbeddingObjective", Net)

    def draw(arm, rng, long_rng, *args):
        a, b = rng.random(), long_rng.random()
        row = helper.pair_row(int(a * 1000), "toy", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 2)
        return SimpleNamespace(
            metrics=[row],
            positives=[],
            relative=([], None),
            group_index=0,
            shared_digest=module.digest([a, b]),
            selected_digest=module.digest([arm, a, b]),
        )

    def optimize(net, obj, opt, rows, positives, relative):
        opt.zero_grad(set_to_none=True)
        loss = (net.weight - rows[0].spec.index / 1000).square().sum() + obj.weight.square().sum()
        loss = loss + torch.rand(()) * net.weight.sum()
        loss.backward()
        opt.step()
        return dict(metric_loss=float(loss.detach()))

    monkeypatch.setattr(module, "paired_draw", draw)
    monkeypatch.setattr(module, "optimize", optimize)
    base = Net()
    before = base.weight.detach().clone()
    report = module.optimizer_gate("long", base, (None,) * 6, [], tmp_path)
    assert report["passed"] and report["immediate_restore_bitwise"] and report["final_bitwise"]
    assert report["optimizer_steps"] == [2] and report["physical_optimizer_calls"] == 3
    assert report["consumed_data_draws"] == 32
    assert len(report["batch_records"]) == 2 and torch.equal(base.weight, before)
    assert all(module.sha(tmp_path / name) == sha for name, sha in report["checkpoints"].items())


def test_training_refuses_absent_or_failed_execution_gate(module, tmp_path):
    trainer = importlib.import_module("train_long_general_pair")
    path = tmp_path / "report.json"
    with pytest.raises(FileNotFoundError):
        trainer.execution_gate(path, tmp_path)
    module.atomic(path, dict(schema=module.SCHEMA, status="failed"))
    with pytest.raises(ValueError, match="complete matching152"):
        trainer.execution_gate(path, tmp_path)
    module.atomic(path, dict(schema=module.SCHEMA, status="passed", config={}))
    with pytest.raises(ValueError, match="complete matching152"):
        trainer.execution_gate(path, tmp_path)
