import copy
import importlib

import pytest
import torch
from test_pretrained_raster_policy import bridge as bridge
from test_pretrained_raster_policy import net as net

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective


def cases():
    config = dict(
        seed=20261074,
        horizon=3,
        insertion_candidates=4,
        simulations=4,
        widening_batch=4,
        c_puct=1.5,
        temperature=1.0,
    )
    roots = []
    for i, (word, weight) in enumerate((((1,), 10), ((1, 1, 1), 1000))):
        instance = M.Instance(
            str(i), word, 2, 0, cap_moves=3, growth_len=0, growth_strands=0, cc_weight=weight
        )
        roots.append(dict(condition=i, panel_index=i, instance=instance.to_dict()))
    return roots, config


def test_distance_mode_exactly_preserves_frozen97_batch(net):
    new = importlib.import_module("resource_cost_search")
    old = importlib.import_module("raster_cost_search")
    roots, config = cases()
    objective = MetricEmbeddingObjective()
    expected_rows, expected_counts = old.run_batch(roots, net, objective, 0, config)
    rows, counts = new.run_batch(roots, net, objective, 0, config)
    assert rows == expected_rows and counts == expected_counts
    assert net.training


def test_uniform_tree_keeps_real_policy_and_never_encodes_distances(net, monkeypatch):
    module = importlib.import_module("resource_cost_search")
    audit = importlib.import_module("audit_joint_cost_mcts")
    roots, config = cases()
    before = copy.deepcopy(net.state_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("distance model used by uniform-tree control")

    monkeypatch.setattr(module, "RawKnotDistance", forbidden)
    rows, counts = module.run_batch(
        roots, net, MetricEmbeddingObjective(), 0, config, prior="uniform"
    )
    assert rows[0]["solved"] and not rows[1]["solved"]
    assert counts["tree_embedding_evaluations"] == counts["tree_embedding_batches"] == 0
    assert counts["policy_neural_evaluations"] == counts["continuation_moves"] > 0
    assert counts["tree_action_scores"] > 0
    for root, row in zip(roots, rows, strict=True):
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        audit.audit_condition(instance, row, config)
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())
    assert net.training


def test_invalid_prior_fails_closed(net):
    module = importlib.import_module("resource_cost_search")
    roots, config = cases()
    with pytest.raises(ValueError):
        module.run_batch(roots, net, MetricEmbeddingObjective(), 0, config, prior="oracle")
