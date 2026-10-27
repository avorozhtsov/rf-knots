"""Fixed32 random pilot, cycle-free best witness export only."""
import argparse,shutil
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from export_mcts_root_positives_v1 import semantic_id,external_groups
from build_continuous_train_roster_v1 import exclusions
from screen_legacy_unselected_root_f_v1 import cycle_free
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm

DATA=Path('artifacts/short-horizon-random-mcts-pilot-v4')
AUDIT=Path('artifacts/short-horizon-random-mcts-pilot-audit-v4-full/report.json')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or shutil.disk_usage(a.out.parent).free<12*1024**3:raise ValueError('immutable/storage')
    m=read(DATA/'manifest.json');ar=read(AUDIT);assert m['next_task']==32 and m['status']=='completed'
    assert not ar['partial'] and ar['manifest_sha256']==sha(DATA/'manifest.json') and ar['representations_with_two_signs']==4
    pins={str(f):sha(f) for f in [DATA/'manifest.json',DATA/'roster.json',AUDIT,Path(__file__),
        Path('scripts/audit_random_root_positives_v1.py'),Path('research/433-random-short-horizon-positive-export-v1.md')]}
    for f,h in m['inputs'].items():assert sha(Path(f))==h; pins[f]=h
    roots=read(DATA/'roster.json')['rows'];excluded=external_groups();_,held=exclusions();heldnorm={norm(*s) for s in held}
    items={};rejected=[];success=0
    for pos,e in enumerate(m['shards']):
        sp=DATA/e['file'];assert sha(sp)==e['sha256'];s=read(sp)
        if not s['row'] or not s['row']['witness']:continue
        success+=1;r=roots[s['root']];w=UnknotWitness.from_dict(s['row']['witness']);w.verify()
        d=r['instances'][0 if s['weight']==10 else 1];inst=M.Instance(**{**d,'word':tuple(d['word'])})
        fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]));assert owner(fp)=='train' and g==tuple(r['query_group']) and g not in excluded
        assert (w.start.word,w.start.strands)==(inst.word,inst.strands) and norm(inst.word,inst.strands) not in heldnorm
        assert w.moves<=inst.cap_moves and w.crossing_changes<=inst.cap_crossing_changes
        actions=[t.action.to_flat(inst.spec) for t in w.steps]
        if not cycle_free(inst,actions)[0]:rejected.append(pos);continue
        ident=semantic_id(w)
        if ident not in items:items[ident]=dict(path_id=ident,query_group=list(g),gradient_scope='exact_zero_spent_TRAIN_root_only',
            origin='MCTS_best_success_not_reference',sampling_source='normalized_random_short_horizon',instance=d,witness=w.to_dict(),
            root_action=w.steps[0].action.to_dict(),achieved=dict(cc=w.crossing_changes,moves=w.moves,L10=10*w.crossing_changes+w.moves,
            L1000=1000*w.crossing_changes+w.moves,semantics='verified_upper_bound_not_optimal_cost'),observations=[])
        items[ident]['observations'].append(dict(position=pos,root=s['root'],seed=s['seed'],weight=s['weight'],shard=str(sp),
            shard_sha256=sha(sp),actor=m['actor']))
    views={norm(tuple(r['instance']['word']),r['instance']['strands']) for r in items.values()};assert len(views)>=2
    a.out.mkdir();entries=[]
    for i,r in enumerate(items.values()):
        f=a.out/f'path-{i:04d}.json';atomic(f,r);entries.append(dict(file=f.name,sha256=sha(f)))
    atomic(a.out/'manifest.json',dict(status='exported_not_training',schema='rf-random-root-positive-v1',inputs=pins,paths=entries,
        summary=dict(solved_conditions=success,cycle_rejected_positions=rejected,semantic_paths=len(items),normalized_views=len(views),
        query_groups=len({tuple(r['query_group']) for r in items.values()}),new_F_roots=0,optimizer_steps=0)))
    print(read(a.out/'manifest.json')['summary'])
if __name__=='__main__':main()
