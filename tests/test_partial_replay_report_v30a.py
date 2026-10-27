import json
from types import SimpleNamespace
import recover_random_window_277_280_v30a as recovery

def test_partial_replay_does_not_require_final_report(tmp_path,monkeypatch):
    d=tmp_path/'batch-000280';d.mkdir();actorfile=tmp_path/'actor.pt';actorfile.write_text('pin')
    spec=tmp_path/'spec.json';spec.write_text('pin')
    import hashlib
    digest=hashlib.sha256(b'pin').hexdigest()
    actor=dict(file=str(actorfile),sha256=digest,training_spec=str(spec),training_spec_sha256=digest)
    rows=[dict(witness={},query_group=[1,0],instances=[dict(word=[i+1],strands=3)]*2) for i in range(4)]
    for name,data in [('supply.json',dict(status='completed',rows=rows)),('actor.json',actor),
        ('contract.json',dict(inputs={}))]:(d/name).write_text(json.dumps(data))
    for name in recovery.condition_names()[:9]:
        _,root,weight,seed=name.removesuffix('.json').split('-')
        (d/name).write_text(json.dumps(dict(root=int(root),weight=int(weight),seed=int(seed),actor=actor,
            result=dict(cpu_seconds=1.,episodes=[],witness=None))))
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    monkeypatch.setattr(recovery.p,'UnknotWitness',SimpleNamespace(from_dict=lambda _:object()))
    monkeypatch.setattr(recovery.p,'admit',lambda *args:((1,0),True))
    monkeypatch.setattr(recovery.p,'instance',lambda r:SimpleNamespace(**r,cap_moves=10))
    inputs={};items,views,report=recovery.replay_batch(280,set(),set(),inputs,partial=True)
    assert report['conditions']==9 and report['search_cpu_seconds']==9
    assert not (d/'collection.json').exists()
    assert str(d/'collection.json') not in inputs
    assert not items and not views
