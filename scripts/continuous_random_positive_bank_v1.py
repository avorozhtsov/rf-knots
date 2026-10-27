"""Additive immutable random-success bank and independent full audit mode."""
import argparse,shutil,time
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from export_mcts_root_positives_v1 import semantic_id,external_groups
from audit_mcts_root_positives_v1 import key
from build_continuous_train_roster_v1 import exclusions
from screen_legacy_unselected_root_f_v1 import cycle_free
from rf_knots.evidence import UnknotWitness
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm

def good(w,inst,excluded,hn):
    w.verify();fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]))
    assert owner(fp)=='train' and g not in excluded and norm(w.start.word,w.start.strands) not in hn
    assert (w.start.word,w.start.strands)==(inst.word,inst.strands)
    assert w.moves<=inst.cap_moves and w.crossing_changes<=inst.cap_crossing_changes
    return g,cycle_free(inst,[s.action.to_flat(inst.spec) for s in w.steps])[0]

def build(d,old,old_audit):
    start=time.process_time();o=read(old/'manifest.json');a=read(old_audit);assert a['status']=='passed_data_only' and a['manifest_sha256']==sha(old/'manifest.json')
    m=read(d/'search/manifest.json');r=read(d/'full-audit/report.json');assert m['status']=='completed' and m['next_task']==32 and not r['partial'] and r['manifest_sha256']==sha(d/'search/manifest.json')
    out=d/'bank'
    if out.exists():raise ValueError('immutable bank')
    out.mkdir();entries=[];ids=set()
    for e in o['paths']:
        assert sha(old/e['file'])==e['sha256'];shutil.copyfile(old/e['file'],out/e['file']);entries.append(e);ids.add(read(old/e['file'])['path_id'])
    old_ids=ids.copy();_,held=exclusions();hn={norm(*s) for s in held};excluded=external_groups();roots=read(d/'search/roster.json')['rows'];items={};rejected=[];solves=0
    for e in m['shards']:
        sp=d/'search'/e['file'];assert sha(sp)==e['sha256'];s=read(sp)
        if not s['row'] or not s['row']['witness']:continue
        solves+=1;w=UnknotWitness.from_dict(s['row']['witness']);root=roots[s['root']];data=root['instances'][0 if s['weight']==10 else 1];inst=M.Instance(**{**data,'word':tuple(data['word'])})
        g,ok=good(w,inst,excluded,hn)
        if not ok:rejected.append(s['position']);continue
        ident=semantic_id(w);assert ident not in old_ids
        if ident not in items:items[ident]=dict(path_id=ident,query_group=list(g),gradient_scope='exact_zero_spent_TRAIN_root_only',origin='MCTS_best_success_not_reference',
            sampling_source='normalized_random_short_horizon',instance=data,witness=w.to_dict(),root_action=w.steps[0].action.to_dict(),
            achieved=dict(cc=w.crossing_changes,moves=w.moves,L10=10*w.crossing_changes+w.moves,L1000=1000*w.crossing_changes+w.moves,semantics='verified_upper_bound_not_optimal_cost'),observations=[])
        items[ident]['observations'].append(dict(position=s['position'],root=s['root'],seed=s['seed'],weight=s['weight'],shard=str(sp),shard_sha256=sha(sp),actor=m['actor']))
    for ident,row in items.items():
        f=out/f'path-{len(entries):06d}.json';atomic(f,row);entries.append(dict(file=f.name,sha256=sha(f)))
    pins={str(f):sha(f) for f in [old/'manifest.json',old_audit,d/'search/manifest.json',d/'full-audit/report.json',d/'supply.json',d/'supply-audit/report.json',Path(__file__)]}
    atomic(out/'manifest.json',dict(status='exported_not_training',schema='rf-cumulative-random-positive-v1',inputs=pins,paths=entries,
        parent_bank=str(old),parent_audit=str(old_audit),new_path_ids=sorted(items),source_solved=solves,cycle_rejected=rejected,search_cpu_seconds=m['cpu_seconds'],build_cpu_seconds=time.process_time()-start))

def audit(d):
    start=time.process_time();bank=d/'bank';m=read(bank/'manifest.json')
    for f,h in m['inputs'].items():assert sha(Path(f))==h,f
    old=Path(m['parent_bank']);om=read(old/'manifest.json');oa=read(Path(m['parent_audit']));assert oa['status']=='passed_data_only' and oa['manifest_sha256']==sha(old/'manifest.json')
    assert m['paths'][:len(om['paths'])]==om['paths']
    for e in om['paths']:assert (bank/e['file']).read_bytes()==(old/e['file']).read_bytes()
    _,held=exclusions();hn={norm(*s) for s in held};excluded=external_groups();seen=set();views=set();groups=set();newviews=set();observations=set();newobs=set()
    for e in m['paths']:
        p=bank/e['file'];assert sha(p)==e['sha256'];r=read(p);w=UnknotWitness.from_dict(r['witness']);ident=key(w);assert ident==r['path_id'] and ident not in seen;seen.add(ident)
        data=r['instance'];inst=M.Instance(**{**data,'word':tuple(data['word'])});g,ok=good(w,inst,excluded,hn);assert ok and g==tuple(r['query_group'])
        k=norm(inst.word,inst.strands);views.add(k);groups.add(g)
        assert r['origin']=='MCTS_best_success_not_reference' and r['sampling_source']=='normalized_random_short_horizon' and r['gradient_scope']=='exact_zero_spent_TRAIN_root_only'
        assert r['root_action']==w.steps[0].action.to_dict() and r['achieved']==dict(cc=w.crossing_changes,moves=w.moves,L10=10*w.crossing_changes+w.moves,L1000=1000*w.crossing_changes+w.moves,semantics='verified_upper_bound_not_optimal_cost')
        isnew=ident in m['new_path_ids']
        if isnew:newviews.add(k)
        for o in r['observations']:
            obskey=(o['shard'],o['position']);assert obskey not in observations;observations.add(obskey)
            sp=Path(o['shard']);assert sha(sp)==o['shard_sha256'] and sha(Path(o['actor']['file']))==o['actor']['sha256'];s=read(sp)
            assert s['position']==o['position'] and s['seed']==o['seed'] and s['weight']==o['weight'] and s['root']==o['root'] and key(UnknotWitness.from_dict(s['row']['witness']))==ident
            if isnew:assert sp.parent==d/'search';newobs.add(o['position'])
    assert len(newobs)+len(m['cycle_rejected'])==m['source_solved']
    cpu=m['search_cpu_seconds'];gate=len(newviews)>=2 and cpu/len(newviews)<=300
    (d/'bank-audit').mkdir();atomic(d/'bank-audit/report.json',dict(status='passed_data_only',manifest_sha256=sha(bank/'manifest.json'),full_paths_replayed=len(seen),
        normalized_random_views=len(views),TRAIN_query_groups=len(groups),new_normalized_views=len(newviews),new_paths=len(m['new_path_ids']),retained_paths=len(om['paths']),
        yield_gate=gate,search_cpu_seconds=cpu,CPU_per_new_view=cpu/len(newviews) if newviews else None,audit_cpu_seconds=time.process_time()-start,optimizer_connected=False,auditor_sha256=sha(Path(__file__))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path);p.add_argument('--audit',type=Path);p.add_argument('--old-bank',type=Path);p.add_argument('--old-audit',type=Path);a=p.parse_args()
    if a.build:build(a.build,a.old_bank,a.old_audit)
    else:audit(a.audit)
if __name__=='__main__':main()
