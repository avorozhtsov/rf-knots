import importlib.util
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.rollout_value import rollout_candidates, clipped_relative_loss


@pytest.fixture
def trainer(monkeypatch):
    scripts=Path(__file__).parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("rollout_trainer",scripts/"train_rollout_value.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("micro_batch",[1,2,3,5])
def test_chunked_relative_gradient_matches_whole_group_sum(trainer,micro_batch):
    torch.set_num_threads(1);torch.manual_seed(293)
    net=M.SitePolicyValueNet(width=16,depth=2,embedding_dim=8,encoder_kind="strand_graph",budget_dim=5)
    requests=[]
    for length in (3,5,7,9,11):
        p=M.Instance("t",(1,)*length,2,1,cap_moves=20)
        s=M.SearchState(p.word,p.strands,0,0)
        requests.append((p,s,rollout_candidates(p,s)))
    indices=torch.tensor([0,1,2,3,4])
    coeff=torch.tensor([1.0,-0.25,-0.25,-0.25,-0.25])
    full,_=trainer.forward(net,requests)
    selected=full[torch.arange(5),indices]
    old=selected.detach()+torch.tensor([0.01,-0.02,0.03,-0.04,0.05])
    loss=clipped_relative_loss(selected,old,coeff)
    (0.1*loss).backward()
    gradients={k:p.grad.clone() for k,p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    report=trainer.relative_policy_backward(net,(requests,indices,old,coeff),micro_batch=micro_batch)
    assert report["loss"]==pytest.approx(float(loss.detach()),abs=1e-5)
    assert report["mean_abs_log_ratio"]==pytest.approx(0.03,abs=3e-6)
    assert report["clip_fraction"]==0
    for key,p in net.named_parameters():
        # Float32 padding shapes and reduction order change roundoff, not the
        # summed objective; do not demand bitwise identity across batch shapes.
        if key in gradients:torch.testing.assert_close(p.grad,gradients[key],atol=1e-5,rtol=1e-4)


def test_microbatch_preserves_exact_sum_in_double_precision(trainer,monkeypatch):
    torch.manual_seed(174)
    net=torch.nn.Linear(3,5,dtype=torch.float64)
    features=torch.randn(7,3,dtype=torch.float64)
    requests=list(features)
    def forward(model,requests):
        logits=model(torch.stack(requests))
        return logits.log_softmax(-1),None
    monkeypatch.setattr(trainer,"forward",forward)
    indices=torch.tensor([0,1,2,3,4,0,1])
    old=forward(net,requests)[0][torch.arange(7),indices].detach()+0.01
    coefficients=torch.tensor([1]+[-1/6]*6,dtype=torch.float64)
    selected=forward(net,requests)[0][torch.arange(7),indices]
    loss=clipped_relative_loss(selected,old,coefficients)
    (0.1*loss).backward()
    grads=[p.grad.clone() for p in net.parameters()]
    net.zero_grad(set_to_none=True)
    report=trainer.relative_policy_backward(net,(requests,indices,old,coefficients),micro_batch=2)
    assert report["loss"]==pytest.approx(float(loss.detach()),abs=1e-14)
    for p,expected in zip(net.parameters(),grads,strict=True):
        torch.testing.assert_close(p.grad,expected,atol=1e-14,rtol=1e-12)


@pytest.mark.parametrize("corruption",["missing_seed","duplicate","missing_root","extra_root"])
def test_rollout_coverage_gate_rejects_corruption(trainer,corruption):
    episodes=[{"group":g,"seed":s} for g in range(2) for s in range(8)]
    trainer.validate_episode_coverage(episodes,2,8)
    if corruption=="missing_seed":episodes.pop()
    if corruption=="duplicate":episodes[-1]=episodes[-2]
    if corruption=="missing_root":episodes=episodes[:8]
    if corruption=="extra_root":episodes.append({"group":2,"seed":0})
    with pytest.raises(ValueError):trainer.validate_episode_coverage(episodes,2,8)
