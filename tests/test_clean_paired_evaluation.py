from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path=Path(__file__).parents[1]/"scripts/run_clean_paired_evaluation.py"
    spec=importlib.util.spec_from_file_location("paired_evaluation",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_full_frozen_job_matrix_and_unique_outputs(runner,tmp_path):
    jobs=runner.jobs(tmp_path,55,64)
    assert len(jobs)==42
    assert len({j["name"] for j in jobs})==len(jobs)
    assert len({j["output"] for j in jobs})==len(jobs)
    searches=[j for j in jobs if j["command"][2].endswith("evaluate_strand_graph_search.py")]
    assert len(searches)==10
    for j in searches:
        argv=j["command"]
        assert argv[argv.index("--instances")+1]=="55"
        assert argv[argv.index("--seeds")+1]=="2"
        assert argv[argv.index("--simulations")+1]=="512"
        assert "--site-proposals" not in argv and "--factorized-proposals" not in argv
    assert sum("--learned-value" in j["command"] for j in searches)==4
    rollout=next(j for j in jobs if j["name"]=="parent-rollouts")
    assert rollout["command"][rollout["command"].index("--roots")+1]=="64"
    assert rollout["command"][rollout["command"].index("--seed")+1]=="20261016"
    assert not any("train_rollout_value.py" in " ".join(j["command"]) for j in jobs)


def manifests():
    a={"status":"completed","completed_steps":2048,"relative_groups":132,"long_relative_groups":23,
       "config":{"seed":20261013,"steps":2048,"out":"value","value_weight":1},
       "sources":{"source":"hash"},"inputs":{"data":"hash"},"architecture":{"width":48},
       "initial_checkpoint_sha256":"initial"}
    b=deepcopy(a);b["config"].update(out="control",value_weight=0)
    return [a,b]


def test_matched_training_gate(runner):runner.check_training_manifests(manifests())


def test_resume_requires_exact_completed_prefix(runner,tmp_path):
    planned=runner.jobs(tmp_path,55,64)
    runner.check_completed_prefix(planned[:3],planned)
    with pytest.raises(ValueError):runner.check_completed_prefix(planned[1:3],planned)
    with pytest.raises(ValueError):runner.check_completed_prefix(planned+[planned[0]],planned)
    changed=deepcopy(planned[:1]);changed[0]["output"]="another-output"
    with pytest.raises(ValueError):runner.check_completed_prefix(changed,planned)


def test_previously_scored_checkpoints_cannot_be_replaced(runner):
    state={};runner.pin_checkpoints(state,{"parent":"a","value":"b","control":"c"})
    runner.pin_checkpoints(state,{"parent":"a","value":"b","control":"c"})
    with pytest.raises(ValueError):runner.pin_checkpoints(state,{"parent":"a","value":"changed","control":"c"})


@pytest.mark.parametrize("mutation",["partial","steps","seed","initial","sources","inputs","architecture","gate","config"])
def test_training_gate_rejects_changed_or_incomplete_runs(runner,mutation):
    rows=manifests();b=rows[1]
    if mutation=="partial":b["status"]="training"
    if mutation=="steps":b["completed_steps"]=2047
    if mutation=="seed":b["config"]["seed"]+=1
    if mutation=="initial":b["initial_checkpoint_sha256"]="other"
    if mutation in ("sources","inputs","architecture"):b[mutation]={"different":1}
    if mutation=="gate":b["long_relative_groups"]=7
    if mutation=="config":b["config"]["lr"]=0.1
    with pytest.raises(ValueError):runner.check_training_manifests(rows)


def test_live_matching_does_not_adopt_unrelated_training_or_evaluation(runner,tmp_path):
    job=runner.jobs(tmp_path,55,64)[0]
    argv=["/path/to/Python",*job["command"][1:]]
    assert runner.matching_children(job,[(5,argv,"command")])==[5]
    assert runner.matching_children(job,[(5,[*argv,"--initial"],"command")])==[]
    assert runner.matching_children(job,[(5,["bash",*argv[1:]],"command")])==[]
    train=["Python","-u","scripts/train_rollout_value.py","--data",str(runner.DATA),"--out",str(runner.RUNS["value"])]
    assert runner.training_live([(8,train,"command")],12)
    train[-1]="unrelated-run"
    assert not runner.training_live([(8,train,"command")],12)
    assert not runner.training_live([(12,["bash"],"unrelated command")],12)
