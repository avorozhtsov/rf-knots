"""Independent full replay and descriptive relative-success candidates."""
import argparse
from collections import defaultdict
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from audit_f_short_positive_r2_yield_v1 import replay_episode
from screen_legacy_unselected_root_f_v1 import cycle_free
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_replay_corpus import invariants,owner

DATA=Path('artifacts/random-representation-mcts-pilot-v1')


def rank_paths(paths):
    if len(paths)<2:return []
    costs=[p['cost'] for p in paths];out=[]
    for p in paths:
        better=sum(c>p['cost'] for c in costs)/(len(costs)-1)
        worse=sum(c<p['cost'] for c in costs)/(len(costs)-1)
        sign=1 if better>=.8 else -1 if worse>=.8 else 0
        if sign:out.append({**p,'sign':sign})
    signs={p['sign'] for p in out}
    if signs!={-1,1}:return []
    for p in out:p['weight']=.5/sum(q['sign']==p['sign'] for q in out)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    if a.out.exists():raise ValueError('immutable audit exists')
    m=read(DATA/'manifest.json');assert a.partial or m['next_task']==32
    for f,h in m['inputs'].items():assert sha(Path(f))==h,f
    assert sha(DATA/'roster.json')==m['roster_sha256'];assert sha(Path(m['actor']['file']))==m['actor']['sha256']
    roots=read(DATA/'roster.json')['rows'];instances={};paths=defaultdict(dict);episodes=0
    for i,r in enumerate(roots):
        w=UnknotWitness.from_dict(r['witness']);w.verify();fp=invariants(w.start.word,w.start.strands)
        assert owner(fp)=='train' and (fp[0],abs(fp[1]))==tuple(r['query_group'])
        for wi,d in enumerate(r['instances']):
            instance=M.Instance(**{**d,'word':tuple(d['word'])});instances[i,wi]=instance
            assert (instance.word,instance.strands)==(w.start.word,w.start.strands)
    censored=0;solutions=0
    for i,e in enumerate(m['shards']):
        assert sha(DATA/e['file'])==e['sha256'];s=read(DATA/e['file']);ri=i//8;wi=(i//4)%2;inst=instances[ri,wi]
        assert (s['position'],s['root'],s['weight'],s['seed'])==(i,ri,inst.cc_weight,20281016+i%4)
        if s['cpu_censored']:
            assert s['row'] is None;censored+=1;continue
        row=s['row'];assert row['simulations']==8 and row['search_seed']==s['seed']
        if row['witness']:
            w=UnknotWitness.from_dict(row['witness']);w.verify();assert (w.start.word,w.start.strands)==(inst.word,inst.strands)
            assert w.crossing_changes<=inst.cap_crossing_changes and w.moves<=inst.cap_moves
        for episode in row['episodes']:
            w=replay_episode(inst,episode);episodes+=1
            if w:
                actions=tuple(t.action.to_flat(inst.spec) for t in w.steps)
                if cycle_free(inst,actions)[0]:
                    solutions+=1;paths[ri,wi][actions]=dict(actions=list(actions),cc=w.crossing_changes,moves=w.moves,
                        cost=inst.cc_weight*w.crossing_changes+w.moves)
    candidates=[]
    for (ri,wi),ps in paths.items():
        ranked=rank_paths(list(ps.values()))
        if ranked:candidates.append(dict(root=ri,weight=instances[ri,wi].cc_weight,paths=ranked))
    useful=len({p['root'] for p in candidates});a.out.mkdir()
    atomic(a.out/'report.json',dict(partial=m['next_task']<32,conditions=m['next_task'],episodes_replayed=episodes,
        cycle_free_solution_episodes=solutions,censored=censored,relative_path_candidates=candidates,
        representations_with_two_signs=useful,admitted_old_style_F_roots=0,optimizer_steps=0,
        cpu_seconds=m['cpu_seconds'],manifest_sha256=sha(DATA/'manifest.json'),auditor_sha256=sha(Path(__file__)),
        decision='partial_no_scale' if m['next_task']<32 else 'optimizer_smoke_contract_only' if useful>=2 else
                 'separate_replication' if useful==1 else 'stop_configuration_review'))
    print(dict(conditions=m['next_task'],episodes=episodes,useful=useful),flush=True)


if __name__=='__main__':main()
