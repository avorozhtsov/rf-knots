"""Independent replay/ownership/provenance audit of root-only positives."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from build_continuous_train_roster_v1 import exclusions
from screen_legacy_unselected_root_f_v2 import external_exclusions
from screen_legacy_unselected_root_f_v1 import VIEWED,cycle_free
from run_continuous_train_mcts_v1 import read
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.evidence import UnknotWitness


def key(w):
    acts=[(s.action.kind,s.action.position,s.action.generator,s.action.sign) for s in w.steps]
    return hashlib.sha256(json.dumps([list(w.start.word),w.start.strands,acts],separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('immutable audit exists')
    m=read(a.bank/'manifest.json');assert m['status']=='exported_not_training' and m['cohort_full_batches']==356 and m['conditions']==11392
    for f,h in m['inputs'].items():assert sha(Path(f))==h,f
    assert sha(a.bank/'rejected.json')==m['rejected_sha256']
    heldgroups=external_exclusions()
    for f in VIEWED:heldgroups.update(tuple(r['query_group']) for r in read(f)['rows'])
    _,held=exclusions();seen=set();groups=set();stages=Counter();observations=set()
    for e in m['paths']:
        f=a.bank/e['file'];assert sha(f)==e['sha256'];r=read(f);w=UnknotWitness.from_dict(r['witness']);w.verify()
        ident=key(w);assert ident==r['path_id'] and ident not in seen;seen.add(ident)
        assert r['origin']=='MCTS_best_success_not_reference' and r['gradient_scope']=='exact_zero_spent_TRAIN_root_only'
        fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]));assert owner(fp)=='train' and g==tuple(r['query_group']) and g not in heldgroups
        d=r['instance'];inst=M.Instance(**{**d,'word':tuple(d['word'])});assert (inst.word,inst.strands)==(w.start.word,w.start.strands) and (inst.word,inst.strands) not in held
        assert w.crossing_changes<=inst.cap_crossing_changes and w.moves<=inst.cap_moves
        actions=[s.action.to_flat(inst.spec) for s in w.steps];assert actions and cycle_free(inst,actions)[0]
        assert r['root_action']==w.steps[0].action.to_dict()
        t=r['achieved'];assert t==dict(cc=w.crossing_changes,moves=w.moves,L10=10*w.crossing_changes+w.moves,
            L1000=1000*w.crossing_changes+w.moves,semantics='verified_upper_bound_not_optimal_cost')
        assert r['observations']
        for o in r['observations']:
            assert o['position'] not in observations and 0<=o['position']<11392;observations.add(o['position'])
            assert sha(Path(o['shard']))==o['shard_sha256'] and sha(Path(o['batch_report']))==o['batch_report_sha256']
            report=read(Path(o['batch_report']));assert not report['partial'] and report['conditions']==32 and report['actor']==o['actor']
            assert sha(Path(o['actor']['file']))==o['actor']['sha256']
            saved=read(Path(o['shard']));assert saved['position']==o['position'] and saved['seed']==o['seed'] and saved['weight']==o['weight']
            original=UnknotWitness.from_dict(saved['row']['witness']);assert key(original)==ident
        if g not in groups:groups.add(g);stages[r['stage']]+=1
    assert len(seen)==m['summary']['unique_semantic_paths'] and len(groups)==m['summary']['distinct_query_groups']
    assert len(observations)+len(read(a.bank/'rejected.json'))==m['summary']['solved_conditions']
    a.out.mkdir();atomic(a.out/'report.json',dict(status='passed_data_only',manifest_sha256=sha(a.bank/'manifest.json'),
        full_paths_replayed=len(seen),source_success_observations=len(observations),distinct_TRAIN_query_groups=len(groups),
        stage_groups=dict(stages),gradient_scope='root_only_no_optimizer_in_this_contract',new_F_roots=0,optimizer_steps=0,
        auditor_sha256=sha(Path(__file__))))
    print(read(a.out/'report.json'),flush=True)


if __name__=='__main__':main()
