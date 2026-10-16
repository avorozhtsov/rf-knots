import importlib
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from rf_knots import mcts_loop as M


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_long_general_training_pair")


def test_replayed_draws_check_every_commit_and_independent_work_counters(module, monkeypatch):
    helper = importlib.import_module("long_general_training_data")
    runner = importlib.import_module("train_long_general_pair")
    monkeypatch.setitem(module.CONFIG, "steps", 32)
    metric = helper.pair_row(0, "fixture", ((1, 1, 1), 2), ((1,) * 7, 3), 1, 2, 2)
    short = (None, M.SearchState((1,) * 3, 2, 0, 0), ())
    long = (None, M.SearchState((1,) * 7, 3, 0, 0), ())
    relative = [(long, 0, 1.0)]

    def draw(arm, rng, lrng, *args):
        a, b = rng.random(), lrng.random()
        return SimpleNamespace(
            metrics=[metric] * 32,
            positives=[(short, (0,))] * 8,
            relative=(relative, None),
            group_index=0,
            shared_digest=module.digest([a, b]),
            selected_digest=module.digest([arm, a, b]),
        )

    monkeypatch.setattr(module, "paired_draw", draw)
    states = list(module.replay_draws("long", (None, None, None, None, None, [relative])))
    assert [step for step, _ in states] == [0, 16, 32]
    assert states[1][1]["relative_counts"] == [1] and states[2][1]["relative_counts"] == [2]
    rng, lrng = random.Random(module.CONFIG["seed"]), random.Random(module.CONFIG["long_seed"])
    counters = runner.size_counters()
    for step in range(32):
        row = draw("long", rng, lrng)
        runner.count_batch(counters, row, row.relative if (step + 1) % 16 == 0 else None)
    final = states[-1][1]
    assert counters == final["size_counters"]
    assert final["rng"] == rng.getstate() and final["long_rng"] == lrng.getstate()
    assert counters["metric_pairs"] == 1024 and counters["metric_word_tokens"] == 10240
    assert counters["policy_states"] == 258 and counters["policy_word_tokens"] == 32 * 8 * 3 + 14
    assert counters["policy_max_strands"] == 3 and counters["policy_max_length"] == 7


@pytest.mark.parametrize("arm", ["control", "long"])
def test_real_sampler_trace_independent_of_reference_probabilities(module, monkeypatch, arm):
    import torch

    helper = importlib.import_module("long_general_training_data")
    fixtures = importlib.import_module("test_long_general_training_data")
    runner = importlib.import_module("train_long_general_pair")
    monkeypatch.setitem(module.CONFIG, "steps", 32)
    old = fixtures.old_sampler(helper)
    long_source = helper.LongGeneralSampler(fixtures.fixture_records(helper), set())
    positives, preferences = fixtures.policy_fixtures()
    preferences = [(g, torch.arange(len(g), dtype=torch.float32)) for g, _ in preferences]
    data = (old, old, old, long_source, positives, [g for g, _ in preferences])
    replayed = list(module.replay_draws(arm, data))[-1][1]
    rng = random.Random(module.CONFIG["seed"])
    long_rng = random.Random(module.CONFIG["long_seed"])
    shared, selected = module.digest([]), module.digest([])
    counters = runner.size_counters()
    counts = [0] * len(preferences)
    for step in range(32):
        draw = helper.paired_draw(
            arm, rng, long_rng, old, old, old, long_source, positives, preferences
        )
        shared = module.digest([shared, draw.shared_digest])
        selected = module.digest([selected, draw.selected_digest])
        relative = draw.relative if module.relative_due(step) else None
        if relative is not None:
            counts[draw.group_index] += 1
        runner.count_batch(counters, draw, relative)
    assert replayed == dict(
        shared_draw_digest=shared,
        selected_draw_digest=selected,
        relative_counts=counts,
        size_counters=counters,
        rng=rng.getstate(),
        long_rng=long_rng.getstate(),
    )


def dev_fixture(module):
    helper = importlib.import_module("long_general_training_data")
    rows = [helper.pair_row(i, "DEV-fixture", ((1, 1, 1), 2), ((), 1), i, i, i) for i in range(3)]
    records = [dict(left_fingerprint=fp) for fp in ([3, 2], [3, -2], [5, 0])]
    predictions = [0.1, 1.5, 1.8]
    errors = [abs(i - p) for i, p in enumerate(predictions)]
    report = dict(
        count=3,
        groups=2,
        records=[
            dict(
                index=i,
                lower=i,
                upper=i,
                prediction=p,
                violation=errors[i],
                constant1_violation=abs(i - 1),
                query_group=[3, 2] if i < 2 else [5, 0],
            )
            for i, p in enumerate(predictions)
        ],
        mean_interval_violation=sum(errors) / 3,
        equal_group_interval_violation=((errors[0] + errors[1]) / 2 + errors[2]) / 2,
        constant1_mean_violation=2 / 3,
        exact_groups={
            str(i): dict(count=1, mean_interval_violation=e) for i, e in enumerate(errors)
        },
    )
    return report, rows, records


def test_dev_recomputed_record_and_group_arithmetic(module):
    module.verify_development(*dev_fixture(module))


@pytest.mark.parametrize(
    "field", ["count", "group", "prediction", "violation", "mean", "constant", "exact"]
)
def test_dev_rejects_wrong_binding_or_statistics(module, field):
    report, rows, records = dev_fixture(module)
    report = deepcopy(report)
    if field == "count":
        report["count"] = 2
    elif field == "group":
        report["records"][0]["query_group"] = [3, -2]
    elif field == "prediction":
        report["records"][0]["prediction"] = float("nan")
    elif field == "violation":
        report["records"][0]["violation"] = 0
    elif field == "mean":
        report["equal_group_interval_violation"] = report["mean_interval_violation"]
    elif field == "constant":
        report["constant1_mean_violation"] = 0
    else:
        report["exact_groups"]["2"]["count"] = 2
    with pytest.raises(ValueError):
        module.verify_development(report, rows, records)
