import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_pure_braid_distance_evaluation")


def test_data_wait_requires_live_auditor_or_exact_passed_certificate(controller):
    c = controller
    assert c.data_gate(None, [7], "abc") == [{"kind": "pure-braid-audit", "pid": 7}]
    with pytest.raises(RuntimeError, match="no live auditor"):
        c.data_gate(None, [], "abc")
    with pytest.raises(RuntimeError, match="duplicate"):
        c.data_gate(None, [7, 8], "abc")
    audit = {"status": "passed", "evaluation_ready": True, "inputs": {str(c.PANEL): "abc"}}
    assert c.data_gate(audit, [], "abc") == []
    audit["evaluation_ready"] = False
    with pytest.raises(ValueError, match="inadequate"):
        c.data_gate(audit, [], "abc")


def test_final_pair_wait_respects_serial_parent_and_failure(controller):
    c = controller
    assert c.pair_gate(None, {"status": "waiting_for_gates"}, [9])[0]["controller_pid"] == 9
    with pytest.raises(RuntimeError, match="not live"):
        c.pair_gate(None, {"status": "waiting_for_gates"}, [])
    with pytest.raises(RuntimeError, match="needs attention"):
        c.pair_gate(None, {"status": "needs_attention"}, [9])
    with pytest.raises(ValueError, match="lacks training-pair"):
        c.pair_gate(None, {"status": "completed"}, [9])
    audit = {
        "status": "passed",
        "same_initial_checkpoint": c.INITIAL_SHA,
        "arms": {"graph": {}, "control": {}},
    }
    assert c.pair_gate(audit, {"status": "completed"}, []) == []
    audit["same_initial_checkpoint"] = "different"
    with pytest.raises(ValueError, match="differs"):
        c.pair_gate(audit, {"status": "completed"}, [])


def test_jobs_use_three_fixed_final_models_and_no_training(controller, tmp_path):
    jobs = controller.jobs(tmp_path)
    assert [j["name"] for j in jobs] == ["initial", "graph", "control"]
    assert len({j["output"] for j in jobs}) == 3
    assert all(j["command"][2] == "scripts/evaluate_pure_braid_distance_panel.py" for j in jobs)
    assert all("--resume" not in j["command"] for j in jobs)


def reports():
    result = {}
    for arm in ("initial", "graph", "control"):
        rows = []
        for i, target in enumerate((0, 1, 2)):
            prediction = target if arm == "graph" else 1
            rows.append(
                {
                    "panel_index": i,
                    "lower": target,
                    "upper": target,
                    "prediction": prediction,
                    "violation": abs(target - prediction),
                    "fingerprint_group": f"{101 + i}:2",
                    "left_fingerprint": [101 + i, 2],
                }
            )
        result[arm] = {"validation": {"predictions": rows}, "summary": {}}
    return result


def test_comparison_keeps_zero_and_positive_quality_separate_and_counts_overlap(controller):
    short = {"validation": [{"left_fingerprint": [101, -2]}, {"left_fingerprint": [999, 0]}]}
    result = controller.compare(reports(), short)
    delta = result["comparisons"]["graph-vs-initial"]
    assert delta["exact_two"]["mean_group_error_delta"] == -1
    assert delta["positive_lower_bound"]["mean_group_error_delta"] == -0.5
    assert result["query_groups_also_in_simultaneous_short_u_panel"] == [(101, 2)]


def test_paired_comparison_rejects_different_query_identity(controller):
    data = deepcopy(reports())
    data["graph"]["validation"]["predictions"][0]["panel_index"] = 9
    with pytest.raises(ValueError, match="labels/queries"):
        controller.compare(data, {"validation": []})
