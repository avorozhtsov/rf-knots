import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def summary(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/summarize_diverse_metric_sequence.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("diverse_summary", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def row(group="3:2", lo=1, hi=1, pred=1):
    return {"panel_index": 0, "left_fingerprint": [3, 2], "right_fingerprint": [1, 0],
            "fingerprint_group": group, "lower": lo, "upper": hi, "prediction": pred,
            "violation": max(lo-pred, pred-hi, 0)}


def test_constant_controls_use_same_group_weighting(summary):
    rows = [row("a", 0, 0, 0), row("a", 0, 0, 0), row("b", 2, 2, 2)]
    m = summary.metric_with_constants(rows)
    assert m["equal_group_mean_interval_violation"] == 0
    assert m["constants"]["0"]["mean_interval_violation"] == pytest.approx(2/3)
    assert m["constants"]["0"]["equal_group_mean_interval_violation"] == 1
    assert m["constants"]["2"]["equal_group_mean_interval_violation"] == 1


def test_intervals_do_not_become_exact_two(summary):
    m = summary.metric_with_constants([row(lo=1, hi=2, pred=1.5)])
    assert m["exact"] == {} and m["mean_interval_violation"] == 0


@pytest.mark.parametrize("key,value", [("panel_index", 1), ("lower", 2), ("fingerprint_group", "other"),
                                      ("prediction", float("nan")), ("violation", 4)])
def test_changed_metric_labels_or_scores_rejected(summary, key, value):
    records = [{"interval": {"lower": 1, "upper": 1}, "left_fingerprint": [3, 2], "right_fingerprint": [1, 0]}]
    r = row(); r[key] = value
    with pytest.raises(ValueError): summary.check_metric_rows([r], records, "original")


@pytest.fixture
def final_fixture(summary, monkeypatch, tmp_path):
    s = summary.sequence
    runs = {n: tmp_path / n for n in ("parent", "candidate", "control")}
    panels = {n: tmp_path / n / "panel.json" for n in ("supplement", "short-u")}
    monkeypatch.setattr(s, "RUNS", runs); monkeypatch.setattr(s, "PANELS", panels)
    def write(p, value):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))
    final = {}
    for n, p in runs.items():
        p.mkdir(); (p/"checkpoint.pt").write_bytes(n.encode()); final[n] = summary.sha(p/"checkpoint.pt")
    audit_pins = {}
    record = {"interval": {"lower": 1, "upper": 1}, "left_fingerprint": [3, 2], "right_fingerprint": [1, 0]}
    for p in panels.values():
        write(p, {"complete": True, "validation": [record], "size_shift": [record]})
        for name in ("metric-independence-audit.json", "rollout-independence-audit.json"):
            a = p.parent/name; write(a, {"status": "passed", "inputs": {str(p): summary.sha(p)}})
            audit_pins[str(a)] = summary.sha(a)
    out = tmp_path/"sequence"; jobs = s.evaluation_jobs(out); completed = []
    lookup = {j["name"]: j["output"] for j in jobs}
    for job in jobs:
        name = job["name"]
        if name.startswith("metric-"):
            model = name.split("-")[1]; panel = "supplement" if "supplement" in name else "short-u"
            report = {"status": "completed", "run": str(runs[model]), "checkpoint_sha256": final[model],
                "panel_sha256": summary.sha(panels[panel]), "initial": False, "normalization": "none",
                "bounds": name.split("-")[-1], "validation": {"predictions": [row()]}, "size_shift": {"predictions": [row()]}}
        elif "-vs-" not in name:
            model = name.removeprefix("cost-"); source = "parent" if model == "length" else model
            report = {"status": "completed", "run": str(runs[source]), "checkpoint_sha256": final[source],
                "panel_sha256": summary.sha(panels["short-u"]), "mode": "length" if model == "length" else "learned",
                "config": {"expansions": 128, "cc_weights": [10, 1000], "cap_moves": 64,
                           "layout_tiebreak": True, "all_queries": True, "witness_caps": True},
                "rows": [{"instance": "short-u-0", "cc_weight": w, "L10": None, "L1000": None} for w in (10, 1000)]}
        else:
            candidate, baseline = name.removeprefix("cost-").split("-vs-")
            report = {"status": "passed", "panel_sha256": summary.sha(panels["short-u"]),
                "reports": [{"path": lookup[f"cost-{n}"], "sha256": summary.sha(Path(lookup[f"cost-{n}"]))}
                            for n in (baseline, candidate)], "summary": {"10": {}, "1000": {}},
                "paired": [{"instance": "short-u-0", "weight": w, "baseline_cost": None, "candidate_cost": None} for w in (10, 1000)]}
        p = Path(job["output"]); write(p, report)
        completed.append({"name": name, "output": str(p), "sha256": summary.sha(p)})
    state = {"status": "completed", "jobs": jobs, "completed": completed, "inputs": {},
             "final_checkpoints": final, "audit_pins": audit_pins}
    manifest = out/"manifest.json"; write(manifest, state)
    return manifest, state


def test_full_matrix_summary_and_disclosures(summary, final_fixture):
    path, _ = final_fixture
    state, reports = summary.verified_reports(path)
    result = summary.summarize(reports); text = summary.markdown(result)
    assert len(result["metric"]) == 9 and len(result["search_comparisons"]) == 4
    assert "FAILED adequacy" in text and "DIFFERENT family" in text
    assert "No automatic promotion" in text and "unavailable" in text
    assert "Same-weight constant controls" in text


@pytest.mark.parametrize("change", ["partial", "order", "commands", "output", "checkpoint", "audits"])
def test_nonfinal_or_changed_evidence_rejected(summary, final_fixture, change):
    path, state = final_fixture
    if change == "partial": state["status"] = "evaluating"
    if change == "order": state["completed"].reverse()
    if change == "commands": state["jobs"][0]["command"].append("--initial")
    if change == "output": Path(state["completed"][0]["output"]).write_text("{}")
    if change == "checkpoint": (summary.sequence.RUNS["candidate"]/"checkpoint.pt").write_bytes(b"changed")
    if change == "audits": state["audit_pins"].pop(next(iter(state["audit_pins"])))
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError): summary.verified_reports(path)


@pytest.mark.parametrize("change", ["model", "missing_cost", "cost", "comparison_source", "query"])
def test_semantic_contract_checked_beyond_file_hashes(summary, final_fixture, change):
    path, _ = final_fixture; state, reports = summary.verified_reports(path)
    if change == "model": reports["metric-candidate-short-u-original"]["checkpoint_sha256"] = "wrong"
    if change == "missing_cost": reports["cost-candidate"]["rows"].pop()
    if change == "cost": reports["cost-candidate-vs-parent"]["paired"][0]["candidate_cost"] = 0
    if change == "comparison_source": reports["cost-candidate-vs-parent"]["reports"].reverse()
    if change == "query": reports["cost-candidate-vs-parent"]["paired"][0]["instance"] = "short-u-1"
    with pytest.raises(ValueError): summary.verify_report_contracts(state, reports)
