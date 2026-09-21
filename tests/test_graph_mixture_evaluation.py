import importlib
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    torch.set_num_threads(1)
    return importlib.import_module("evaluate_graph_mixture_cost_mcts")


def test_prospective_caps_are_size_scaled_and_witness_free(evaluator, monkeypatch):
    i = M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=3)
    s = i.spec
    witness = UnknotWitness.from_actions(
        i.word,
        i.strands,
        s,
        [s.encode(CROSSING_CHANGE, 0), s.encode(REDUCE, 0), s.encode(DESTABILIZE)],
    )
    monkeypatch.setattr(evaluator, "verify_evaluation_pair", lambda r: witness)
    panel = {
        "complete": True,
        "pair_kind": "short-u",
        "validation": [
            {"left_fingerprint": [3, -2], "interval": {"right": {"word": [], "strands": 1}}}
        ],
    }
    a, b = evaluator.build_roots(panel)
    assert a["q0"] == 2
    assert a["horizon"] == a["instance"]["cap_moves"] == 128
    assert a["instance"]["growth_len"] == 8 and a["instance"]["growth_strands"] == 4
    assert a["reference_L10"] == 13 and b["reference_L1000"] == 1003
    assert "witness" not in a and "actions" not in a["instance"]
    long_i = M.Instance("size-only", tuple(range(1, 300)), 300, 1, cap_moves=3)
    monkeypatch.setattr(evaluator, "query_instance", lambda *a, **k: (long_i, True))
    large = evaluator.build_roots(panel)[0]
    assert large["q0"] == 299
    assert large["horizon"] == 4 * 299 + 32
    assert large["stratum"] == large["policy_stratum"] == "both"


@pytest.mark.parametrize("learned", [False, True])
def test_varying_batch_caps_replay_with_per_root_horizon(evaluator, learned):
    from test_joint_cost_mcts_evaluation import checkpoint

    roots = []
    for index, cap in enumerate((5, 9)):
        i = M.Instance("engineering", (1, -1, 1), 2, 0, cap_moves=cap, cc_weight=10)
        roots.append(
            {"condition": index, "panel_index": index, "horizon": cap, "instance": i.to_dict()}
        )
    config = evaluator.batch_config(roots)
    assert config["horizon"] == 9
    model = checkpoint() if learned else None
    rows, counts = evaluator.run_batch(roots, model, 0, config)
    manifest = {
        "next_condition": 2,
        "counts": counts,
        "arm": "graph" if learned else "uniform",
        "shards": [{"seconds": 1.0}],
        "seconds_committed_batches": 1.0,
    }
    audit = importlib.import_module("audit_graph_mixture_cost_mcts")
    result = audit.inspect_rows(roots, rows, manifest)
    assert result["conditions"] == 2
    for root, row in zip(roots, rows, strict=True):
        assert row["maximum_observed_depth"] <= root["horizon"]
        assert all(e["continuation"]["stop_reason"] != "horizon" for e in row["episodes"])
    bad = deepcopy(rows)
    bad[0]["search_seed"] += 1
    with pytest.raises(ValueError, match="RNG"):
        audit.inspect_rows(roots, bad, manifest)


def test_size_strata_use_both_raw_axes(evaluator):
    assert evaluator.stratum(254, 21) == "within"
    assert evaluator.stratum(255, 21) == "length_only"
    assert evaluator.stratum(254, 22) == "strands_only"
    assert evaluator.stratum(255, 22) == "both"


def test_short_u_partition_extension_rejects_train_overlap(evaluator):
    audit = importlib.import_module("audit_graph_mixture_short_u_panel")
    records = [
        {
            "left_fingerprint": [101 + 2 * i, 2],
            "interval": {
                "left": {"word": [1] * 255, "strands": 22},
                "right": {"word": [], "strands": 1},
                "upper": 1 + i % 2,
            },
        }
        for i in range(32)
    ]
    panel = {
        "complete": True,
        "construction_complete": True,
        "schema": "rf-knots-gordian-pair-panel-v1",
        "pair_kind": "short-u",
        "contract": {
            "seed": 20261032,
            "attempts": 10000,
            "max_strands": 33,
            "min_depth": 24,
            "max_depth": 96,
        },
        "validation": records,
    }
    result = audit.inspect(panel, set(), set())
    assert result["raw_metric_size_strata"] == {"both": 32}
    with pytest.raises(ValueError, match="collides"):
        audit.inspect(panel, {(101, 2)}, set())
    with pytest.raises(ValueError, match="collides"):
        audit.inspect(panel, set(), {(101, 2)})
