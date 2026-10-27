"""Independent full random-root data/provenance audit, no optimizer."""
import argparse
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from audit_mcts_root_positives_v1 import key
from export_mcts_root_positives_v1 import external_groups
from build_continuous_train_roster_v1 import exclusions
from screen_legacy_unselected_root_f_v1 import cycle_free
from rf_knots.evidence import UnknotWitness
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('immutable audit')
    m=read(a.bank/'manifest.json');assert m['status']=='exported_not_training'
    for f,h in m['inputs'].items():assert sha(Path(f))==h,f
    _,held=exclusions();hn={norm(*s) for s in held};excluded=external_groups();seen=set();views=set();groups=set();obs=set()
    for e in m['paths']:
        f=a.bank/e['file'];assert sha(f)==e['sha256'];r=read(f);w=UnknotWitness.from_dict(r['witness']);w.verify()
        assert key(w)==r['path_id'] and r['path_id'] not in seen;seen.add(r['path_id'])
        assert r['origin']=='MCTS_best_success_not_reference' and r['sampling_source']=='normalized_random_short_horizon'
        assert r['gradient_scope']=='exact_zero_spent_TRAIN_root_only'
        fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]));assert owner(fp)=='train' and g==tuple(r['query_group']) and g not in excluded;groups.add(g)
        d=r['instance'];inst=M.Instance(**{**d,'word':tuple(d['word'])});assert (w.start.word,w.start.strands)==(inst.word,inst.strands)
        n=norm(inst.word,inst.strands);assert n not in hn;views.add(n)
        assert w.moves<=inst.cap_moves and w.crossing_changes<=inst.cap_crossing_changes
        assert cycle_free(inst,[s.action.to_flat(inst.spec) for s in w.steps])[0] and r['root_action']==w.steps[0].action.to_dict()
        assert r['achieved']==dict(cc=w.crossing_changes,moves=w.moves,L10=10*w.crossing_changes+w.moves,L1000=1000*w.crossing_changes+w.moves,semantics='verified_upper_bound_not_optimal_cost')
        for o in r['observations']:
            assert o['position'] not in obs and 0<=o['position']<32;obs.add(o['position'])
            assert sha(Path(o['shard']))==o['shard_sha256'] and sha(Path(o['actor']['file']))==o['actor']['sha256']
            s=read(Path(o['shard']));assert s['position']==o['position'] and s['seed']==o['seed'] and s['weight']==o['weight'] and s['root']==o['root']
            assert key(UnknotWitness.from_dict(s['row']['witness']))==r['path_id']
    z=m['summary'];assert len(seen)==z['semantic_paths'] and len(views)==z['normalized_views'] and len(groups)==z['query_groups']
    assert len(obs)+len(z['cycle_rejected_positions'])==z['solved_conditions'] and len(views)>=2
    a.out.mkdir();atomic(a.out/'report.json',dict(status='passed_data_only',manifest_sha256=sha(a.bank/'manifest.json'),
        full_paths_replayed=len(seen),normalized_random_views=len(views),TRAIN_query_groups=len(groups),source_observations=len(obs),
        auditor_sha256=sha(Path(__file__)),optimizer_steps=0,new_F_roots=0))
    print(read(a.out/'report.json'))
if __name__=='__main__':main()
