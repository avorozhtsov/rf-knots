import copy
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def sequence(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/run_diverse_metric_sequence.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("diverse_sequence", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def manifests(s):
    result = []
    for model in ("candidate", "control"):
        cfg = {"steps": 4096, "seed": 20261018, "value_weight": 1, "data": str(s.DATA),
               "metric_sampling": "group-balanced", "metric_bounds": "determinant",
               "metric_corpus": str(s.CORPORA[model]), "out": str(s.RUNS[model]),
               "policy_micro_batch": 64, "min_relative_groups": 32, "min_long_relative_groups": 8,
               "long_root_length": 96, "batch_size": 64, "lr": 2.5e-5, "policy_weight": .1,
               "clip": .2, "train_seeds": list(range(6)), "window_size": 0, "hierarchy_levels": 0}
        result.append({"status": "completed", "completed_steps": 4096, "config": cfg,
                       "relative_groups": 132, "long_relative_groups": 23, "value_rows": 130260,
                       "initial_checkpoint_sha256": "b2cd2502dace3b8a0daddcc7424151c6ef87578f63e3394467aa437f1c6e64c3",
                       "sources": {"code": "pinned"}, "architecture": {"width": 48},
                       "inputs": {str(s.CORPORA[model]): model + "-hash", "shared": "same"}})
    return result


def test_exact_job_matrix_and_no_failed_panel_scoring(sequence, tmp_path):
    jobs = sequence.evaluation_jobs(tmp_path)
    assert len(jobs) == len({j["name"] for j in jobs}) == len({j["output"] for j in jobs}) == 17
    assert sum(j["name"].startswith("metric-") for j in jobs) == 9
    assert sum("-vs-" in j["name"] for j in jobs) == 4
    assert all("mcts" not in j["name"] for j in jobs)
    assert not any("embedding-diverse-metric-gordian-test-v1" in " ".join(j["command"]) for j in jobs)
    for j in jobs:
        if j["name"].startswith("cost-") and "-vs-" not in j["name"]:
            assert all(flag in j["command"] for flag in ("--all-queries", "--witness-caps", "--layout-tiebreak"))


def test_matched_data_intervention_allowed(sequence):
    sequence.check_pair(manifests(sequence))


@pytest.mark.parametrize("key,value", [("seed", 1), ("steps", 2048), ("lr", 1e-3),
    ("metric_bounds", "signature"), ("value_weight", 0), ("policy_micro_batch", 128)])
def test_changed_training_contract_rejected(sequence, key, value):
    pair = manifests(sequence)
    for m in pair:
        m["config"][key] = value
    with pytest.raises(ValueError): sequence.check_pair(pair)


@pytest.mark.parametrize("key,value", [("status", "training"), ("completed_steps", 4080),
    ("initial_checkpoint_sha256", "wrong"), ("long_relative_groups", 0), ("value_rows", 1)])
def test_incomplete_or_mismatched_training_rejected(sequence, key, value):
    pair = manifests(sequence); pair[0][key] = value
    with pytest.raises(ValueError): sequence.check_pair(pair)


def test_only_expected_normal_or_resumed_worker_matches(sequence):
    command = sequence.training_command("control")
    wrong = copy.deepcopy(command); wrong[wrong.index("--steps") + 1] = "1024"
    listing = [(1, command, ""), (2, command + ["--resume"], ""), (3, wrong, "")]
    assert sequence.training_workers("control", listing) == [1, 2]
    assert sequence.training_workers("candidate", listing) == []


def test_recursive_tensor_check_covers_optimizer(sequence):
    values = {"model": {"w": torch.ones(2)}, "optimizer": [{"moment": torch.tensor(float("nan"))}]}
    tensors = list(sequence.tensors(values))
    assert len(tensors) == 2 and any(not torch.isfinite(t).all() for t in tensors)
