import importlib
from copy import deepcopy

import pytest
from test_pretrained_raster_policy import bridge as bridge
from test_pretrained_raster_policy import net as net

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective


@pytest.mark.parametrize("prior", ["distance", "uniform"])
@pytest.mark.parametrize("solved", [True, False])
def test_full149_replay_rejects_false_cost_rng_and_counters(net, prior, solved):
    runner = importlib.import_module("resource_cost_search")
    audit = importlib.import_module("audit_long_general_search")
    instance = M.Instance(
        "tiny149",
        (1,) if solved else (1, 1, 1),
        2,
        0,
        cap_moves=3,
        growth_len=0,
        growth_strands=0,
        cc_weight=10,
    )
    roots = [dict(condition=0, panel_index=0, horizon=3, instance=instance.to_dict())]
    rows, counts = runner.run_batch(
        roots, net, MetricEmbeddingObjective(), 0, {**audit.CONFIG, "horizon": 3}, prior=prior
    )
    counts.update(cache_logical_tiles=10, cache_computed_tiles=4, cache_hits=6)
    state = dict(
        next_condition=1,
        counts=dict(counts),
        contract={"prior": prior},
        shards=[{"seconds": 1.0}],
        seconds_committed_batches=1.0,
    )
    assert audit.inspect_rows(roots, rows, state).get("solved_conditions", 0) == int(solved)
    for key in ("cache_hits", "policy_neural_evaluations", "conditions"):
        bad = deepcopy(state)
        bad["counts"][key] += 1
        with pytest.raises(ValueError):
            audit.inspect_rows(roots, rows, bad)
    for key in ("L10", "L1000", "search_seed", "tree_seed"):
        bad = deepcopy(rows)
        bad[0][key] = 1 if bad[0][key] is None else bad[0][key] + 1
        with pytest.raises(ValueError):
            audit.inspect_rows(roots, bad, state)
    with pytest.raises(ValueError, match="incomplete"):
        audit.inspect_rows(roots, [], state)
    if prior == "uniform":
        bad = deepcopy(state)
        bad["counts"]["tree_embedding_evaluations"] = 1
        with pytest.raises(ValueError, match="uniform"):
            audit.inspect_rows(roots, rows, bad)
