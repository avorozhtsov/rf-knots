import importlib.util
from pathlib import Path
import random

import pytest

from rf_knots.embedding_augmentation import MaterializedSyntheticPair, SyntheticPairSpec


@pytest.fixture
def sampler():
    path = Path(__file__).parents[1] / "scripts/balanced_metric_sampling.py"
    spec = importlib.util.spec_from_file_location("metric_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair(det=3, lower=0, upper=2, index=0):
    spec = SyntheticPairSpec(index, "train", "cc", -1, -1, lower, upper, 4, 0, 0, 1)
    row = MaterializedSyntheticPair(spec, "cc", (1, 1, 1), (), 2, 1, 4)
    record = {"split": "train", "left_fingerprint": [det, 0], "right_fingerprint": [1, 0],
              "interval": {"left": {"word": [1, 1, 1], "strands": 2},
                           "right": {"word": [], "strands": 1}, "lower": lower, "upper": upper}}
    return row, record


def test_bound_strengthening_does_not_turn_upper_bound_into_exact_label(sampler):
    rows, records = zip(pair(upper=2), pair(upper=1))
    strengthened = sampler.prepare_rows(rows, records, bounds="determinant")
    assert [(r.spec.distance_lower, r.spec.distance_upper) for r in strengthened] == [(1, 2), (1, 1)]
    assert sampler.prepare_rows(rows, records) == list(rows)
    assert rows[0].spec.distance_lower == 0


def test_group_balance_not_record_multiplicity(sampler):
    examples = [pair(det=3, lower=1, upper=1, index=0)] * 99 + [pair(det=11, lower=1, upper=1, index=1)]
    rows, records = zip(*examples)
    pool = sampler.GroupBalancedMetricSampler(rows, records)
    assert pool.support == {"short-u/exact-1": {"groups": 2, "records": 100}}
    counts = sum(r.spec.index == 1 for r in pool.sample(random.Random(42), 4000))
    assert 1800 < counts < 2200


def test_equal_rng_consumption_and_exact_resume_across_pool_sizes(sampler):
    a = pair(lower=1, upper=1)
    b = pair(det=11, lower=2, upper=2)
    small = sampler.GroupBalancedMetricSampler([a[0]], [a[1]])
    large = sampler.GroupBalancedMetricSampler([a[0], b[0]], [a[1], b[1]])
    left, right = random.Random(192), random.Random(192)
    small.sample(left, 63); large.sample(right, 63)
    assert left.getstate() == right.getstate()
    state = right.getstate()
    expected = large.sample(right, 100)
    resumed = random.Random(); resumed.setstate(state)
    assert expected == large.sample(resumed, 100)


def test_wrong_partition_and_mismatched_row_rejected(sampler):
    row, record = pair()
    record["split"] = "test"
    with pytest.raises(ValueError): sampler.prepare_rows([row], [record])
    with pytest.raises(ValueError): sampler.GroupBalancedMetricSampler([row], [record])
    record["split"] = "train"
    record["interval"]["left"]["word"] = [-1, -1, -1]
    with pytest.raises(ValueError): sampler.prepare_rows([row], [record])


def test_impossible_strengthened_interval_rejected(sampler):
    row, record = pair(lower=0, upper=0)
    with pytest.raises(ValueError, match="upper bound"):
        sampler.prepare_rows([row], [record], bounds="determinant")
