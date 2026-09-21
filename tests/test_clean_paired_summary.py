import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def summary():
    path = Path(__file__).parents[1]/"scripts/summarize_clean_paired_evaluation.py"
    spec = importlib.util.spec_from_file_location("paired_summary",path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_failures_not_converted_into_costs(summary):
    result = summary.paired_stats([
        {"baseline_cost":None,"candidate_cost":None},
        {"baseline_cost":None,"candidate_cost":10},
        {"baseline_cost":20,"candidate_cost":None},
    ])
    assert result["common_solved"] == 0
    assert result["mean_cost_delta_common_only"] is None
    assert result["outcomes"] == {"both_unsolved":1,"newly_solved":1,"lost_solution":1}


def test_new_lost_and_common_costs_remain_separate(summary):
    result = summary.paired_stats([{"baseline_cost":a,"candidate_cost":b}
                                  for a,b in [(10,8),(10,13),(10,10),(None,1),(2,None)]])
    assert result["mean_cost_delta_common_only"] == pytest.approx(1/3)
    assert result["baseline_solved"] == result["candidate_solved"] == 4
    assert result["outcomes"]["newly_solved"] == result["outcomes"]["lost_solution"] == 1


def test_group_weighting_and_interval_vs_exact(summary):
    rows = [{"violation":v,"prediction":hi+v,"fingerprint_group":g,"lower":lo,"upper":hi}
            for v,g,lo,hi in [(0,"a",0,0),(0,"a",0,0),(3,"b",1,2)]]
    result = summary.metric_stats(rows)
    assert result["mean_interval_violation"] == 1
    assert result["equal_group_mean_interval_violation"] == 1.5
    assert result["exact"] == {"0":{"count":2,"mae":0}}


def test_incomplete_controller_rejected_before_scores(summary,tmp_path):
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps({"status":"evaluating","jobs":[{}]*42,"completed":[]}))
    with pytest.raises(ValueError,match="all 42"):
        summary.verified_reports(path)


def test_changed_artifact_or_reordered_completion_rejected(summary,tmp_path):
    jobs=[]; completed=[]
    for i in range(42):
        path=tmp_path/f"{i}.json";path.write_text('{"status":"completed"}')
        jobs.append({"name":str(i),"output":str(path)})
        completed.append({**jobs[-1],"sha256":summary.sha(path)})
    state={"status":"completed","jobs":jobs,"completed":completed,"inputs":{}}
    manifest=tmp_path/"manifest.json";manifest.write_text(json.dumps(state))
    assert len(summary.verified_reports(manifest)[1]) == 42
    completed[0],completed[1]=completed[1],completed[0]
    manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError,match="order"):summary.verified_reports(manifest)
    completed[0],completed[1]=completed[1],completed[0]
    manifest.write_text(json.dumps(state))
    Path(jobs[0]["output"]).write_text('{"status":"completed","changed":true}')
    with pytest.raises(ValueError,match="output changed"):summary.verified_reports(manifest)


def test_missing_summary_conditions_rejected(summary):
    with pytest.raises(ValueError,match="coverage"):summary.summarize({})


@pytest.mark.parametrize("prediction,violation",[(float("nan"),0),(2,0)])
def test_metric_errors_recomputed_not_blindly_trusted(summary,prediction,violation):
    with pytest.raises(ValueError):
        summary.metric_stats([{"prediction":prediction,"violation":violation,
                               "lower":0,"upper":0,"fingerprint_group":"a"}])


def test_current_model_pins_verified(summary,tmp_path):
    reports={};pins={}
    for model in ("parent","value","control"):
        path=tmp_path/model;path.mkdir();(path/"checkpoint.pt").write_bytes(model.encode())
        pins[model]=summary.sha(path/"checkpoint.pt")
        for panel in ("gordian-original","short-u-original","short-u-determinant"):
            reports[f"metric-{model}-{panel}"]={"run":str(path),"checkpoint_sha256":pins[model]}
    summary.verify_model_pins({"final_checkpoints":pins},reports)
    (tmp_path/"value"/"checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="final pin"):
        summary.verify_model_pins({"final_checkpoints":pins},reports)


def test_all_summary_views_and_comparisons_render(summary):
    reports={}
    for model in ("parent","value","control"):
        for panel in ("gordian-original","short-u-original","short-u-determinant"):
            rows=[{"prediction":1,"violation":0,"lower":1,"upper":1,"fingerprint_group":"a"}]
            reports[f"metric-{model}-{panel}"]={"checkpoint_sha256":model,"bounds":panel,
                "constant_baselines":{},"validation":{"predictions":rows},"size_shift":{"predictions":rows}}
    for i in range(4):
        reports[f"cost-{i}-vs-parent"]={"paired":[{"baseline_cost":None,"candidate_cost":None,"weight":w}
                                                   for w in (10,1000)],"summary":{"10":{},"1000":{}}}
    for i in range(12):
        reports[f"mcts-{i}-vs-parent"]={"comparison_kind":"matched_search_models","cc_weight":10,
            "paired":[{"baseline_cost":None,"candidate_cost":None}],"compute":{}}
    for w in (10,1000):
        reports[f"portfolio-vs-parent-{w}"]={"cc_weight":w,"mcts_seed":0,"budget_per_root":512,
            "rows":[{f"mcts_L{w}":None,f"rollout_L{w}":None,"neural_state_evaluations":12,
                     "partial_episode_censored":False}]}
    result=summary.summarize(reports)
    rendered=summary.markdown(result)
    assert rendered.count("| metric-") == 18
    assert rendered.count("| cost-") == 8
    assert rendered.count("| mcts-") == 12
    assert rendered.count("| portfolio-") == 2
    assert "unavailable" in rendered
    assert "No automatic promotion" in rendered
