import copy
import importlib

import pytest
import torch
from test_pretrained_raster_policy import bridge as bridge
from test_pretrained_raster_policy import net as net

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective


def test_raw_raster_batch_replays_success_and_censoring_without_value(net):
    module = importlib.import_module("raster_cost_search")
    audit = importlib.import_module("audit_joint_cost_mcts")
    config = {
        "seed": 97,
        "horizon": 3,
        "insertion_candidates": 4,
        "simulations": 4,
        "widening_batch": 4,
        "c_puct": 1.5,
        "temperature": 1.0,
    }
    instances = [
        M.Instance(
            str(i), word, 2, 0, cap_moves=3, growth_len=0, growth_strands=0, cc_weight=weight
        )
        for i, (word, weight) in enumerate([((1,), 10), ((1, 1, 1), 1000)])
    ]
    roots = [
        {"condition": i, "panel_index": i, "instance": ins.to_dict()}
        for i, ins in enumerate(instances)
    ]
    before = copy.deepcopy(net.state_dict())
    rows, counts = module.run_batch(roots, net, MetricEmbeddingObjective(), 0, config)
    assert rows[0]["solved"] and rows[0]["L10"] == 1
    assert not rows[1]["solved"] and rows[1]["L1000"] is None
    assert counts["conditions"] == 2 and counts["solved_conditions"] == 1
    assert counts["policy_neural_evaluations"] == counts["continuation_moves"]
    assert counts["tree_embedding_evaluations"] > 0
    for instance, row in zip(instances, rows, strict=True):
        audit.audit_condition(instance, row, config)
    assert net.training
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())


def test_raw_raster_batch_rejects_empty(net):
    module = importlib.import_module("raster_cost_search")
    with pytest.raises(ValueError, match="empty"):
        module.run_batch([], net, MetricEmbeddingObjective(), 0, {})
