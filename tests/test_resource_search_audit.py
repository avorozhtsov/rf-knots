import importlib
from copy import deepcopy

import pytest
from test_pretrained_raster_policy import bridge as bridge
from test_pretrained_raster_policy import net as net

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective


def test_full_replay_rejects_fake_success_cost_and_cache_counters(net):
    runner = importlib.import_module("resource_cost_search")
    audit = importlib.import_module("audit_resource_search")
    instance = M.Instance(
        "tiny", (1,), 2, 0, cap_moves=3, growth_len=0, growth_strands=0, cc_weight=10
    )
    roots = [dict(condition=0, panel_index=0, horizon=3, instance=instance.to_dict())]
    config = {**audit.CONFIG, "horizon": 3}
    rows, counts = runner.run_batch(
        roots, net, MetricEmbeddingObjective(), 0, config, prior="uniform"
    )
    counts.update(cache_logical_tiles=10, cache_computed_tiles=4, cache_hits=6)
    state = dict(
        next_condition=1,
        counts=dict(counts),
        contract={"prior": "uniform"},
        shards=[{"seconds": 1.0}],
        seconds_committed_batches=1.0,
    )
    assert audit.inspect_rows(roots, rows, state)["solved_conditions"] == 1
    for key in ("tree_embedding_evaluations", "cache_hits", "policy_neural_evaluations"):
        bad = deepcopy(state)
        bad["counts"][key] += 1
        with pytest.raises(ValueError):
            audit.inspect_rows(roots, rows, bad)
    bad_rows = deepcopy(rows)
    bad_rows[0]["L10"] += 1
    with pytest.raises(ValueError, match="cost differs"):
        audit.inspect_rows(roots, bad_rows, state)
