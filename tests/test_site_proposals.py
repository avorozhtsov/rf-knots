from dataclasses import replace

import numpy as np
import torch

from rf_knots import mcts_loop as M, reference
from rf_knots.actions import CROSSING_CHANGE, INSERT
from rf_knots.lazy_actions import LazyActionCatalogue
from rf_knots.site_proposals import SiteProposalProvider, proposal_candidates


def problem():
    return M.Instance("trefoil", (1,1,1), 2, 1, cap_moves=8)


def test_candidate_pool_is_linear_and_never_materializes_words(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager successor materialization")
    monkeypatch.setattr(reference, "apply", forbidden)
    monkeypatch.setattr(reference, "successors", forbidden)
    p=M.Instance("long-chain",tuple(range(1,4097)),4097,1,cap_moves=5000)
    s=M.SearchState(p.word,p.strands,0,0)
    cat=LazyActionCatalogue(p.spec,s.word,s.strands,True)
    assert cat.insertions>30_000_000
    actions=proposal_candidates(p,s,8)
    assert len(actions)==len(cat.ordinary)+8
    assert len(set(actions))==len(actions)
    assert set(cat.ordinary).issubset(actions)
    assert all(reference.is_legal(p.spec,s.word,s.strands,a,True) for a in actions)
    assert sum(p.spec.decode(a)[0]==INSERT for a in actions)==8


class LastCrossingEvaluator:
    def __init__(self,*args,**kwargs):
        self.calls=self.forwards=0

    def evaluate(self,requests):
        self.calls+=len(requests);self.forwards+=bool(requests)
        out=[]
        for p,s,actions in requests:
            target=p.spec.encode(CROSSING_CHANGE,len(s.word)-1)
            weights=np.array([1000.0 if a==target else 1.0 for a in actions])
            out.append(M.Evaluation(weights/weights.sum(),0.5))
        return out


def fake_provider(monkeypatch):
    monkeypatch.setattr("rf_knots.site_proposals.NetworkEvaluator",LastCrossingEvaluator)
    return SiteProposalProvider(None)


def test_far_site_ranked_first_and_batch_cache_is_bounded(monkeypatch):
    provider=fake_provider(monkeypatch)
    p=M.Instance("far",(1,)*201,2,1,cap_moves=250)
    s=M.SearchState(p.word,2,0,0)
    provider.prepare_batch([(p,s,4),(p,s,4)])
    chosen=provider(p,s,4)
    assert chosen[0]==p.spec.encode(CROSSING_CHANGE,200)
    assert len(chosen)==4
    assert provider.statistics()["state_evaluations"]==1
    assert provider.statistics()["logical_requests"]==2
    q=replace(p,cc_weight=10)
    provider.prepare_batch([(p,s,4),(q,s,4)])
    assert len(provider.prepared)==2
    assert provider.statistics()["state_evaluations"]==3
    provider.prepare_batch([])
    assert not provider.prepared


def test_exhausted_cc_and_moves_are_not_proposed():
    p=problem()
    s=M.SearchState(p.word,2,1,0)
    assert all(p.spec.decode(a)[0]!=CROSSING_CHANGE for a in proposal_candidates(p,s,4))
    assert proposal_candidates(p,M.SearchState(p.word,2,0,8),4)==()
    assert proposal_candidates(p,M.SearchState((),1,0,0),4)==()


def test_batched_search_uses_provider_and_replays(monkeypatch):
    provider=fake_provider(monkeypatch)
    batches=[]
    original=provider.prepare_batch
    def record(requests):
        batches.append(len(requests));original(requests)
    provider.prepare_batch=record
    def forbidden(*args,**kwargs):
        raise AssertionError("eager successor enumeration")
    monkeypatch.setattr(reference,"successors",forbidden)
    results=M.run_searches([problem()],M.UniformEvaluator(),simulations=256,seeds=(0,1),
                          widening_batch=4,proposal_provider=provider)
    assert batches[0]==2
    assert all(r.solved for r in results)
    for r in results:r.witness.verify()
    assert provider.statistics()["state_evaluations"]>0
    assert provider.statistics()["action_scores"]>0


def test_real_site_head_scores_without_changing_weights():
    torch.set_num_threads(1);torch.manual_seed(713)
    net=M.SitePolicyValueNet(width=16,depth=2,embedding_dim=8,encoder_kind="strand_graph",budget_dim=5)
    before={k:v.clone() for k,v in net.state_dict().items()}
    provider=SiteProposalProvider(net)
    p=problem();s=M.SearchState(p.word,p.strands,0,0)
    actions=proposal_candidates(p,s,4)
    expected=M.NetworkEvaluator(net).evaluate([(p,s,actions)])[0].priors
    order=sorted(range(len(actions)),key=lambda j:(-expected[j],actions[j]))
    assert provider(p,s,4)==tuple(actions[j] for j in order[:4])
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())


def test_exact_catalogue_remains_exhaustive_with_proposals(monkeypatch):
    provider=fake_provider(monkeypatch)
    p=problem();tree=M.Tree(p,seed=0,widening_batch=2,proposal_provider=provider)
    for _ in range(100):
        pending=tree.select_leaf()
        if pending is None:break
        tree.prepare_pending(pending)
        n=len(pending.actions)
        assert len(set(pending.actions))==n
        tree.apply(pending,M.Evaluation(np.full(n,1/max(n,1)),0.5))
        tree.root.dead[:]=True
    expected={a for a,_,_ in reference.successors(p.spec,p.word,p.strands,True)}
    assert set(tree.root.actions)==expected
