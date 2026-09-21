from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from rf_knots.actions import ActionSpec, CROSSING_CHANGE, REDUCE, DESTABILIZE
from rf_knots.evidence import UnknotWitness
from rf_knots.mcts_loop import Instance


@pytest.fixture
def comparison():
    path=Path(__file__).parents[1]/"scripts/compare_matched_mcts.py"
    spec=importlib.util.spec_from_file_location("comparison",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    spec=ActionSpec(8,3)
    w=UnknotWitness.from_actions((1,1,1),2,spec,[spec.encode(CROSSING_CHANGE,0),
        spec.encode(REDUCE,0),spec.encode(DESTABILIZE)])
    r={"split":"train","witness":w.to_dict(),"pair_prefix_length":3,
       "left_fingerprint":[3,-2],"right_fingerprint":[1,0],
       "interval":{"left":w.start.to_dict(),"right":w.steps[-1].after.to_dict(),
                   "lower":1,"upper":1,"witness_semantic_length":3}}
    instance=Instance("trefoil",(1,1,1),2,1,cap_moves=8,cc_weight=10,uid="short-u-0")
    row={"instance":"short-u-0","seed":0,"solved":True,"expansions":3,"simulations":3,
         "cc":1,"moves":3,"L10":13,"L1000":1003,"witness":w.to_dict()}
    report={"status":"completed","panel_sha256":"panel","partition":"test:validation",
        "code_sha256":{"source":"digest"},"config":{"instances":1,"seeds":1,"simulations":8,"cc_weight":10},
        "selected":[{"panel_index":0,"fingerprint":[3,2],"resource_caps":instance.to_dict()}],
        "search":{"rows":[row],"trees":1,"solved":1,"seconds":1,"evaluator_requests":3,"action_scores":8}}
    return report,{"validation":[r]}


def test_paired_witnesses_and_censored_failures(comparison):
    a,p=fixture();b=deepcopy(a)
    assert comparison.compare(a,b,p,"panel")["summary"]["equal_cost"]==1
    b["search"]["solved"]=0
    row=b["search"]["rows"][0];row["solved"]=False
    for key in ("cc","moves","L10","L1000","witness"):
        row[key]=None
    result=comparison.compare(a,b,p,"panel")
    assert result["summary"]["lost_solution"]==1
    assert result["summary"]["mean_cost_delta_common_only"] is None


@pytest.mark.parametrize("mutation",["missing","duplicate","cost","failure_cost","budget","root","source"])
def test_corrupt_or_unmatched_evidence_fails_closed(comparison,mutation):
    a,p=fixture();b=deepcopy(a);row=b["search"]["rows"][0]
    if mutation=="missing":b["search"]["rows"]=[]
    if mutation=="duplicate":b["search"]["rows"].append(deepcopy(row))
    if mutation=="cost":row["L10"]=12
    if mutation=="failure_cost":row["solved"]=False
    if mutation=="budget":row["simulations"]=9
    if mutation=="root":b["selected"][0]["resource_caps"]["word"]=[1]
    if mutation=="source":b["code_sha256"]={"source":"changed"}
    with pytest.raises(ValueError):comparison.compare(a,b,p,"panel")


def proposal_fixture():
    a,p=fixture()
    a.update(checkpoint_sha256="frozen",initial=False,uniform=False,
             initial_checkpoint_sha256=None,initial_state_source=None)
    a["config"]["site_proposals"]=False
    b=deepcopy(a);b["config"]["site_proposals"]=True
    b["search"].update(proposal_kind="SiteProposalProvider",total_neural_state_evaluations=5,
                       proposal_compute={"state_evaluations":2,"action_scores":11})
    return a,b,p


def test_proposal_change_is_explicit_and_all_compute_is_charged(comparison):
    a,b,p=proposal_fixture()
    with pytest.raises(ValueError):comparison.compare(a,b,p,"panel")
    result=comparison.compare(a,b,p,"panel",allow_proposal_change=True)
    assert result["comparison_kind"]=="same_model_site_proposals"
    assert result["compute"]["candidate"]["total_action_scores"]==19
    assert result["compute"]["candidate"]["total_neural_state_evaluations"]==5
    assert result["compute"]["baseline"]["total_action_scores"]==8
    assert result["compute"]["baseline"]["total_neural_state_evaluations"] is None


@pytest.mark.parametrize("mutation",["checkpoint","initial","legacy_initial","uniform","missing","budget","source","no_change"])
def test_proposal_comparison_cannot_confound_weights_or_other_settings(comparison,mutation):
    a,b,p=proposal_fixture()
    if mutation=="checkpoint":b["checkpoint_sha256"]="different"
    if mutation=="initial":b["initial"]=True
    if mutation=="legacy_initial":a["initial"]=b["initial"]=True
    if mutation=="uniform":a["uniform"]=b["uniform"]=True
    if mutation=="missing":del a["checkpoint_sha256"]
    if mutation=="budget":b["config"]["simulations"]+=1
    if mutation=="source":b["code_sha256"]={"source":"different"}
    if mutation=="no_change":b["config"]["site_proposals"]=False
    with pytest.raises(ValueError):comparison.compare(a,b,p,"panel",allow_proposal_change=True)


def test_value_intervention_is_explicit_and_model_change_is_disclosed(comparison):
    a,b,p=proposal_fixture();b["config"]["site_proposals"]=False
    a["config"]["learned_value"]=False;b["config"]["learned_value"]=True
    a["search"]["leaf_value"]="constant-0.5";b["search"]["leaf_value"]="learned"
    with pytest.raises(ValueError):comparison.compare(a,b,p,"panel")
    assert comparison.compare(a,b,p,"panel",allow_value_change=True)["comparison_kind"]=="same_model_leaf_value"
    b["checkpoint_sha256"]="new-model"
    assert comparison.compare(a,b,p,"panel",allow_value_change=True)["comparison_kind"]=="changed_model_and_leaf_value"
    b["config"]["simulations"]+=1
    with pytest.raises(ValueError):comparison.compare(a,b,p,"panel",allow_value_change=True)


@pytest.mark.parametrize("mutation",["wrong_leaf","missing_checkpoint","initial","uniform","both_flags"])
def test_value_comparison_fails_closed(comparison,mutation):
    a,b,p=proposal_fixture();b["config"]["site_proposals"]=False
    a["config"]["learned_value"]=False;b["config"]["learned_value"]=True
    a["search"]["leaf_value"]="constant-0.5";b["search"]["leaf_value"]="learned"
    if mutation=="wrong_leaf":b["search"]["leaf_value"]="constant-0.5"
    if mutation=="missing_checkpoint":del b["checkpoint_sha256"]
    if mutation=="initial":b["initial"]=True
    if mutation=="uniform":a["uniform"]=True
    with pytest.raises(ValueError):comparison.compare(a,b,p,"panel",allow_value_change=True,
                                                     allow_proposal_change=mutation=="both_flags")
