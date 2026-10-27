"""Preserve legacy evidence, quarantine protected-root collisions from sampling.

v28's failed audit is retained. No panel, old row, checkpoint or pinned v28
source is edited. Only legacy rows may be quarantined; a NEW leaked label fails.
"""
import argparse
import shutil
from pathlib import Path
from build_mcts_training_supervision import atomic, sha
from learning_pipeline_v28 import (read, pins_ok, disk_guard, protected_registry,
    instance, admit, norm, invariants, owner, UnknotWitness)
from audit_mcts_root_positives_v1 import key


def prepare(source, out):
    disk_guard()
    if out.exists(): raise ValueError('immutable eligibility snapshot exists')
    m=read(source/'manifest.json');pins_ok(m['inputs'])
    parent=Path(m['parent_bank']); pm=read(parent/'manifest.json')
    pa=read(Path(m['parent_audit']))
    assert pa['status']=='passed_data_only' and pa['manifest_sha256']==sha(parent/'manifest.json')
    assert m['paths'][:len(pm['paths'])]==pm['paths']
    old_ids={read(parent/e['file'])['path_id'] for e in pm['paths']}
    groups,states,_,protection_pins=protected_registry()
    quarantined=[]
    for e in m['paths']:
        p=source/e['file'];assert sha(p)==e['sha256']
        r=read(p);w=UnknotWitness.from_dict(r['witness']);w.verify()
        fp=invariants(w.start.word,w.start.strands)
        assert owner(fp)=='train'
        collision=(fp[0],abs(fp[1])) in groups or norm(w.start.word,w.start.strands) in states
        if collision:
            assert r['path_id'] in old_ids and r['path_id'] not in m['new_path_ids'], 'NEW leaked label: stop'
            quarantined.append(r['path_id'])
    out.mkdir(parents=True)
    for e in m['paths']:shutil.copyfile(source/e['file'],out/e['file'])
    m.update(schema='rf-additive-positive-v28a-eligible',quarantined_path_ids=sorted(quarantined),
        quarantine_reason='legacy root intersects protected panel; retained for evidence, never sampled',
        failed_v28_bank=str(source),failed_v28_manifest_sha256=sha(source/'manifest.json'))
    m['inputs'].update({str(source/'manifest.json'):sha(source/'manifest.json'),
        **protection_pins, **{str(p):sha(p) for p in [Path(__file__),
            Path('scripts/positive_sampler_v28a.py'),Path('scripts/train_continuous_random_view_r2_v28a.py'),
            Path('scripts/smoke_connected_learning_cpu_v28a.py'),Path('tests/test_connected_learning_v28a.py'),
            Path('scripts/collect_connected_learning_v28a.py')]}})
    atomic(out/'manifest.json',m)


def audit(bank,out):
    if out.exists():raise ValueError('immutable audit exists')
    m=read(bank/'manifest.json');pins_ok(m['inputs'])
    parent=Path(m['parent_bank']);pm=read(parent/'manifest.json');pa=read(Path(m['parent_audit']))
    assert pa['status']=='passed_data_only' and pa['manifest_sha256']==sha(parent/'manifest.json')
    assert m['paths'][:len(pm['paths'])]==pm['paths']
    for e in pm['paths']:assert (bank/e['file']).read_bytes()==(parent/e['file']).read_bytes()
    quarantine=set(m['quarantined_path_ids']);assert not quarantine.intersection(m['new_path_ids'])
    groups,states,_,_=protected_registry();seen=set();views=set();gs=set();newviews=set();actual_q=set()
    for e in m['paths']:
        p=bank/e['file'];assert sha(p)==e['sha256'];r=read(p);w=UnknotWitness.from_dict(r['witness'])
        w.verify();ident=key(w);assert ident==r['path_id'] and ident not in seen;seen.add(ident)
        inst=instance(r['instance']);fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]))
        assert owner(fp)=='train' and g==tuple(r['query_group'])
        assert (inst.word,inst.strands)==(w.start.word,w.start.strands)
        assert w.moves<=inst.cap_moves and w.crossing_changes<=inst.cap_crossing_changes
        assert r['root_action']==w.steps[0].action.to_dict()
        assert r['origin']=='MCTS_best_success_not_reference'
        assert r['gradient_scope']=='exact_zero_spent_TRAIN_root_only'
        assert r['achieved']==dict(cc=w.crossing_changes,moves=w.moves,L10=10*w.crossing_changes+w.moves,
            L1000=1000*w.crossing_changes+w.moves,semantics='verified_upper_bound_not_optimal_cost')
        trace=[(w.start.word,w.start.strands),*[(x.after.word,x.after.strands) for x in w.steps]]
        assert len(trace)==len(set(trace))
        collision=g in groups or norm(w.start.word,w.start.strands) in states
        if ident in quarantine:
            assert collision;actual_q.add(ident)
            continue # evidence retained, explicitly NOT training-ready
        assert not collision
        admit(w,inst,groups,states)
        v=norm(w.start.word,w.start.strands);views.add(v);gs.add(g)
        if ident not in m['new_path_ids']:continue
        newviews.add(v)
        for o in r['observations']:
            assert o['source_kind']=='v28'
            sp=Path(o['shard']);assert sha(sp)==o['shard_sha256']
            saved=read(sp);assert saved['actor']==o['actor'] and saved['seed']==o['seed']
            assert sha(Path(o['actor']['file']))==o['actor']['sha256']
            assert key(UnknotWitness.from_dict(saved['result']['witness']))==ident
            roster=Path(o['roster']);assert sha(roster)==o['roster_sha256']
            root=read(roster)['rows'][o['root_index']]
            assert root['instances'][0 if o['weight']==10 else 1]==r['instance']
    assert actual_q==quarantine
    out.mkdir(parents=True)
    atomic(out/'report.json',dict(status='passed_data_only',manifest_sha256=sha(bank/'manifest.json'),
        full_paths_replayed=len(seen),retained_paths=len(pm['paths']),new_paths=len(m['new_path_ids']),
        eligible_paths=len(seen)-len(quarantine),quarantined_paths=len(quarantine),
        normalized_views=len(views),TRAIN_query_groups=len(gs),new_normalized_views=len(newviews),
        optimizer_connected=False,auditor_sha256=sha(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    prepare(a.source,a.out/'bank');audit(a.out/'bank',a.out/'audit')
    atomic(a.out/'READY.json',dict(bank=str(a.out/'bank'),audit=str(a.out/'audit/report.json'),
        manifest_sha256=sha(a.out/'bank/manifest.json'),audit_sha256=sha(a.out/'audit/report.json'),
        optimizer_connected=False))
