"""Immutable fixed-cohort single-success MCTS observations; no gradients."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from build_mcts_training_supervision import atomic,sha
from build_continuous_train_roster_v1 import exclusions
from screen_legacy_unselected_root_f_v1 import cycle_free,VIEWED
from screen_legacy_unselected_root_f_v2 import external_exclusions
from run_continuous_train_mcts_v1 import read
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.evidence import UnknotWitness

DATA=Path('artifacts/continuous-train-mcts-v2')
ROSTER=Path('artifacts/continuous-train-roster-v1/roster.json')
PROTOCOL=Path('research/426-fixed356-root-only-mcts-positive-export-v1.md')


def external_groups():
    groups=external_exclusions()
    for p in VIEWED:groups.update(tuple(r['query_group']) for r in read(p)['rows'])
    return groups


def semantic_id(w):
    actions=[(s.action.kind,s.action.position,s.action.generator,s.action.sign) for s in w.steps]
    return hashlib.sha256(json.dumps([list(w.start.word),w.start.strands,actions],separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or shutil.disk_usage(a.out.parent).free<12*1024**3:raise ValueError('output/storage guard')
    source=read(DATA/'manifest.json');assert source['contract_sha256']=='15c058bfda1b5502a4ec62bc5afb36a3309f9ee34a43f015bbb7c8608b555dbe'
    assert source['next_task']>=356*32
    reports={e['batch']:e for e in source['audits'] if e['batch']<356};assert set(reports)==set(range(356))
    pins={str(f):sha(f) for f in [PROTOCOL,Path(__file__),Path('scripts/audit_mcts_root_positives_v1.py'),
        Path('tests/test_mcts_root_positives_v1.py'),ROSTER]}
    for f,h in source['contract']['inputs'].items():assert sha(Path(f))==h; pins[f]=h
    for f,h in read(ROSTER)['inputs'].items():assert sha(Path(f))==h;pins[f]=h
    excluded=external_groups();_,held=exclusions();roots=read(ROSTER)['rows'];items={};rejected=[];successes=0
    for bi in range(356):
        e=reports[bi];path=Path(e['report']);assert sha(path)==e['sha256'];r=read(path)
        bp=DATA/f'batch-{bi:06d}/manifest.json';batch=read(bp)
        assert r['conditions']==32 and not r['partial'] and sha(bp)==r['batch_manifest_sha256']
        assert batch['shards']==source['shards'][bi*32:(bi+1)*32] and batch['actor']==r['actor']
        pins[str(path)]=sha(path);pins[str(bp)]=sha(bp)
        actor=batch['actor'];assert sha(Path(actor['file']))==actor['sha256']
        for offset,entry in enumerate(batch['shards']):
            sp=DATA/entry['file'];assert sha(sp)==entry['sha256'];saved=read(sp);row=saved['row']
            if not row or not row['witness']:continue
            successes+=1;root=roots[saved['root_index']];key=tuple(root['query_group'])
            wi=0 if saved['weight']==10 else 1;d=root['instances'][wi];inst=M.Instance(**{**d,'word':tuple(d['word'])})
            w=UnknotWitness.from_dict(row['witness']);w.verify();fp=invariants(w.start.word,w.start.strands)
            assert (w.start.word,w.start.strands)==(inst.word,inst.strands) and owner(fp)=='train' and (fp[0],abs(fp[1]))==key
            assert key not in excluded and (inst.word,inst.strands) not in held
            assert w.moves<=inst.cap_moves and w.crossing_changes<=inst.cap_crossing_changes
            assert row['L10']==10*w.crossing_changes+w.moves and row['L1000']==1000*w.crossing_changes+w.moves
            actions=[s.action.to_flat(inst.spec) for s in w.steps]
            if not cycle_free(inst,actions)[0]:
                rejected.append(dict(position=bi*32+offset,reason='cycle-containing completed trajectory; not negative target'));continue
            assert actions
            ident=semantic_id(w)
            obs=dict(position=bi*32+offset,batch=bi,seed=saved['seed'],weight=saved['weight'],
                shard=str(sp),shard_sha256=sha(sp),batch_report=str(path),batch_report_sha256=sha(path),actor=actor)
            if ident not in items:
                items[ident]=dict(path_id=ident,query_group=root['query_group'],stage=root['stage'],
                    gradient_scope='exact_zero_spent_TRAIN_root_only',origin='MCTS_best_success_not_reference',
                    instance=d,witness=w.to_dict(),root_action=w.steps[0].action.to_dict(),
                    achieved=dict(cc=w.crossing_changes,moves=w.moves,L10=row['L10'],L1000=row['L1000'],semantics='verified_upper_bound_not_optimal_cost'),
                    observations=[])
            items[ident]['observations'].append(obs)
    a.out.mkdir();entries=[];stages=Counter();seen=set()
    for i,(ident,row) in enumerate(sorted(items.items())):
        f=a.out/f'path-{i:05d}.json';atomic(f,row);entries.append(dict(file=f.name,sha256=sha(f)))
        key=tuple(row['query_group'])
        if key not in seen:seen.add(key);stages[row['stage']]+=1
    atomic(a.out/'rejected.json',rejected)
    atomic(a.out/'manifest.json',dict(status='exported_not_training',schema='rf-root-positive-MCTS-v1',inputs=pins,
        paths=entries,rejected_sha256=sha(a.out/'rejected.json'),cohort_full_batches=356,conditions=356*32,
        summary=dict(solved_conditions=successes,cycle_rejected_conditions=len(rejected),unique_semantic_paths=len(entries),
            distinct_query_groups=len(seen),stage_groups=dict(stages),new_F_roots=0,optimizer_steps=0),
        target_scope='root-only positive action and achieved-cost observations; no negative or intermediate labels'))
    print(read(a.out/'manifest.json')['summary'],flush=True)


if __name__=='__main__':main()
