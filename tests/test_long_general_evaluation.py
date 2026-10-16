import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from rf_knots.embedding_metric import MetricEmbeddingObjective


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("long_general_evaluation_inputs")


@pytest.fixture
def scoring(inputs):
    return importlib.import_module("evaluate_long_general_distances")


def checkpoint_fixture(module, tmp_path):
    arms, pins = {}, {}
    for arm in ("control", "long"):
        arms[arm] = dict(steps=2048, checkpoints_checked=129)
        for stage, step in (("initial", 0), ("final", 2048)):
            path = tmp_path / arm / f"{stage}.pt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"{arm}-{stage}: fixture, not Torch".encode())
            arms[arm][stage] = dict(file=path.name, steps=step, sha256=module.sha(path))
            pins[str(path)] = module.sha(path)
    return dict(
        status="passed",
        arms=arms,
        inputs=pins,
        initial_tensors_identical=True,
        data_rng_aligned=True,
    )


def test_checkpoint_stage_and_independent_audit_binding(inputs, tmp_path):
    audit = checkpoint_fixture(inputs, tmp_path)
    entry, path = inputs.model_entry(tmp_path, audit, "long", "final")
    assert path.name == "final.pt" and entry["steps"] == 2048
    for arm, stage in (("raw", "final"), ("control", "best")):
        with pytest.raises(ValueError):
            inputs.model_entry(tmp_path, audit, arm, stage)


@pytest.mark.parametrize("field", ["status", "count", "steps", "shared", "rng", "hash", "path"])
def test_reject_partial_or_tampered_pair(inputs, tmp_path, field):
    audit = checkpoint_fixture(inputs, tmp_path)
    if field == "status":
        audit["status"] = "training"
    elif field == "count":
        audit["arms"]["control"]["checkpoints_checked"] = 128
    elif field == "steps":
        audit["arms"]["long"]["final"]["steps"] = 16
    elif field == "shared":
        audit["initial_tensors_identical"] = False
    elif field == "rng":
        audit["data_rng_aligned"] = False
    elif field == "hash":
        (tmp_path / "long" / "final.pt").write_bytes(b"changed")
    else:
        audit["arms"]["long"]["final"]["file"] = "../control/final.pt"
    with pytest.raises(ValueError):
        inputs.model_entry(tmp_path, audit, "long", "final")


def test_incomplete_parent_cannot_begin_evaluation(inputs, tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(schema=inputs.SCHEMA, status="training"))
    )
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    with pytest.raises(ValueError, match="completely committed"):
        inputs.load_inputs(tmp_path, audit)


def corpus_fixture(module):
    helper = importlib.import_module("long_general_training_data")
    fixtures = importlib.import_module("test_long_general_training_data")
    row = fixtures.fixture_records(helper)[0]
    fp = next([d, 0] for d in range(3, 300, 2) if module.owner((d, 0)) == "test")
    row.update(split="test", fingerprints=[fp, fp], left_fingerprint=fp, right_fingerprint=fp)
    return dict(
        records=dict(train=[], validation=[], test=[row]),
        excluded_query_groups=[],
        training_groups={"historical": []},
    )


def test_general_test_materialization_without_training_rows(inputs):
    corpus = corpus_fixture(inputs)
    assert inputs.test_records(corpus) == corpus["records"]["test"]


@pytest.mark.parametrize("mutation", ["train", "dev", "excluded", "duplicate", "split", "bound"])
def test_reject_query_leakage_and_unbound_general_fields(inputs, mutation):
    corpus = corpus_fixture(inputs)
    row = corpus["records"]["test"][0]
    if mutation == "train":
        corpus["records"]["train"].append(deepcopy(row))
    elif mutation == "dev":
        corpus["records"]["validation"].append(deepcopy(row))
    elif mutation == "excluded":
        corpus["excluded_query_groups"].append(row["left_fingerprint"])
    elif mutation == "duplicate":
        corpus["records"]["test"].append(deepcopy(row))
    elif mutation == "split":
        row["split"] = "validation"
    else:
        row["upper"] = 1
    with pytest.raises(ValueError):
        inputs.test_records(corpus)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 0.0]))

    def encoder(self, words, lengths, strands):
        assert not torch.is_grad_enabled()
        return {"knot": self.weight[None]}


def test_shared_initial_checks_exact_state_and_position_aware_forwards(inputs, monkeypatch):
    net, objective = Toy().eval(), MetricEmbeddingObjective()
    calls = []

    def policy(model, requests, *, heads):
        calls.append(heads[0])
        assert not torch.is_grad_enabled()
        return model.weight + heads[0]

    monkeypatch.setattr(inputs, "policy_log_probabilities", policy)
    other = dict(model=deepcopy(net.state_dict()), objective=deepcopy(objective.state_dict()))
    endpoint = dict(word=[1] * 5, strands=2)
    report = inputs.initial_identity(net, objective, other, endpoint)
    assert report["exact_tensors"] and report["encoder_equal"] and report["policy_equal"]
    assert calls == [0, 0, 2, 2, 4, 4]
    other["model"]["weight"][0] = 2
    with pytest.raises(ValueError, match="initial tensor"):
        inputs.initial_identity(net, objective, other, endpoint)


def test_distance_strata_and_positive_ranking_preserve_all_records(scoring):
    rows = []
    # Include an uncertain [0,1] case: it must not count as certified positive.
    for i, (lo, hi, length, strands) in enumerate(
        ((0, 0, 48, 12), (1, 1, 49, 13), (2, 2, 193, 33), (0, 1, 385, 34))
    ):
        endpoint = dict(word=[1] * length, strands=strands)
        rows.append(
            dict(
                index=i,
                interval=dict(left=endpoint, right=endpoint, lower=lo, upper=hi),
                left_fingerprint=[3, 2],
                right_fingerprint=[3, 2],
            )
        )
    net = Toy().train()
    result = scoring.enrich(scoring.score_records(net, MetricEmbeddingObjective(), rows), rows)
    assert net.training
    assert result["encoded_endpoints"] == 8
    assert result["certified_positive"]["rows"] == 2
    assert result["certified_positive"]["ranking"]["comparable_pairs"] == 1
    assert result["certified_positive"]["ranking"]["accuracy_half_credit_ties"] == 0.5
    assert result["certified_positive"]["constant1"]["mean_interval_violation"] == 0.5
    assert set(result["by_maximum_length"]) == {"0..48", "49..96", "193..384", "385+"}
    assert set(result["by_maximum_strands"]) == {"1..12", "13..24", "25..33", "34+"}
    assert sum(r["rows"] for r in result["by_maximum_length"].values()) == 4
    assert result["predictions"][-1]["maximum_length"] == 385
    assert scoring.describe([])["statistics"] is None
    with pytest.raises(ValueError, match="coverage"):
        scoring.enrich(result, rows[:-1])
    result["predictions"][0]["panel_index"] = 1
    with pytest.raises(ValueError, match="order"):
        scoring.enrich(result, rows)
