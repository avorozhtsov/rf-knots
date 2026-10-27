"""Independent full baseline/panel audit and frozen evaluator/source lock."""
import json
from pathlib import Path
import torch
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
from rf_knots.embedding_replay_corpus import invariants,owner

ROOT=Path('artifacts/paired-learning-48h100k-v1')
def sid(w,n):
    import hashlib
    word,strands=norm(tuple(w),n)
    return hashlib.sha256(json.dumps([list(word),strands],separators=(',',':')).encode()).hexdigest()

def main():
    assert not (ROOT/'study.json').exists(),'immutable study lock already exists'
    baseline=read(ROOT/'baseline.json');panel=read(ROOT/'panel.json');excluded=read(ROOT/'exclusions.json')
    assert sha(Path(baseline['checkpoint']))==baseline['checkpoint_sha256']
    saved=torch.load(baseline['checkpoint'],map_location='cpu',weights_only=False)
    assert saved['step']==baseline['step'] and len(saved['optimizer']['state'])==71
    def finite(x):
        if torch.is_tensor(x):assert torch.isfinite(x).all()
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,(list,tuple)):
            for v in x:finite(v)
    finite(saved)
    for k in ['rng','torch_rng','mps_rng']:assert k in saved
    for f,h in excluded['inputs'].items():assert sha(Path(f))==h,f
    assert panel['exclusions_sha256']==sha(ROOT/'exclusions.json')
    assert panel['protocol_sha256']==sha(Path('artifacts/paired-learning-48h100k-v1-protocol.json'))
    groups=set(map(tuple,excluded['excluded_query_groups']));states=set(excluded['excluded_normalized_state_sha256']);seen=set();bands=[0]*4
    for row in panel['rows']:
        w=UnknotWitness.from_dict(row['witness']);w.verify();fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]))
        assert owner(fp)=='validation' and g==tuple(row['query_group']) and g not in groups and g not in seen
        seen.add(g);length=len(w.start.word);band=0 if 20<=length<=64 else 1 if 65<=length<=96 else 2 if 97<=length<=128 else 3 if 129<=length<=192 else None
        assert band==row['band'];bands[band]+=1
        assert all(sid(s.word,s.strands) not in states for s in [w.start,*[x.after for x in w.steps]] if len(s.word)>=20)
        for weight,data in zip((10,1000),row['instances'],strict=True):
            assert tuple(data['word'])==w.start.word and data['strands']==w.start.strands and data['cc_weight']==weight
            assert w.moves<=data['cap_moves'] and w.crossing_changes<=data['cap_crossing_changes']
    assert bands==[8]*4 and len(seen)==32
    files=[Path('artifacts/paired-learning-48h100k-v1-protocol.json'),Path(__file__),
        Path('scripts/freeze_learning_comparison_v1.py'),Path('scripts/paired_learning_comparison_v1.py'),
        Path('scripts/analyze_learning_comparison_v1.py'),Path('tests/test_paired_learning_comparison_v1.py'),
        Path('scripts/run_continuous_train_mcts_v1.py'),Path('scripts/bootstrap_cost_value_teacher_v1.py'),
        Path('scripts/pretrained_raster_policy.py'),Path('scripts/audit_cost_mcts_continuations.py'),
        Path('scripts/build_mcts_training_supervision.py'),
        *Path('src/rf_knots').glob('*.py'),*Path('/Users/artemvorozhtsov/projects/pgx-mcts-bench/src/pgx_mcts_bench').glob('*.py'),
        *[ROOT/f for f in ['baseline.pt','baseline.json','baseline-training-manifest.json','baseline-training-spec.json','baseline-search-manifest.json','exclusions.json','panel.json']]]
    atomic(ROOT/'audit.json',dict(status='passed_no_network_evaluation',reference_full_replays=32,bands=bands,
        baseline_step=baseline['step'],baseline_SHA=baseline['checkpoint_sha256'],excluded_groups=len(groups),excluded_normalized_states=len(states)))
    files.append(ROOT/'audit.json')
    atomic(ROOT/'study.json',dict(status='FROZEN_WAITING_48H_AND_100K',inputs={str(p):sha(p) for p in files},
        gates=dict(checkpointed_active_seconds=172800,audited_new_MCTS_episodes=100000)))
    print(read(ROOT/'audit.json'))

if __name__=='__main__':main()
