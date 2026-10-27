"""Independent complete replay/inverse/legal normalized uniqueness audit."""
import argparse
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from build_normalized_random_roster_v3 import known_states
from rf_knots import reference,mcts_loop as M
from rf_knots.actions import COMMUTE,BRAID
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.evidence import UnknotWitness


def main():
    p=argparse.ArgumentParser();p.add_argument('--roster',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('immutable audit exists')
    r=read(a.roster);assert r['status']=='completed'
    for f,h in r['inputs'].items():assert sha(Path(f))==h,f
    known,held=known_states();seen=set();groups=set();stages=set()
    for row in r['rows']:
        w=UnknotWitness.from_dict(row['witness']);w.verify();key=norm(w.start.word,w.start.strands)
        assert key not in known and key not in seen and (w.start.word,w.start.strands) not in held
        assert key==(tuple(row['normalized_word']),row['normalized_strands']);seen.add(key)
        fp=invariants(w.start.word,w.start.strands);assert owner(fp)=='train' and (fp[0],abs(fp[1]))==tuple(row['query_group'])
        base=r['bases'][row['base_ordinal']];state=(tuple(base['word']),base['strands']);visited={norm(*state)}
        assert len(row['walk'])==8 and tuple(base['query_group'])==tuple(row['query_group'])
        for action in row['walk']:
            assert w.spec.decode(action)[0] in [COMMUTE,BRAID] and reference.is_legal(w.spec,*state,action,False)
            after=reference.apply(w.spec,*state,action);assert reference.apply(w.spec,*after,action)==state
            assert norm(*after) not in visited;visited.add(norm(*after));state=after
        assert state==(w.start.word,w.start.strands)
        assert [s.action.to_flat(w.spec) for s in w.steps[:8]]==list(reversed(row['walk']))
        for weight,d in zip([10,1000],row['instances'],strict=True):
            inst=M.Instance(**{**d,'word':tuple(d['word'])});assert inst.cc_weight==weight and (inst.word,inst.strands)==state
            assert w.crossing_changes<=inst.cap_crossing_changes and w.moves<=inst.cap_moves
        assert w.moves<=192 and len(w.start.word)<=192 and w.start.strands<=20
        groups.add(tuple(row['query_group']));stages.add(row['stage'])
    assert len(seen)==r['distinct_normalized_representations']
    a.out.mkdir();atomic(a.out/'report.json',dict(status='passed',roster_sha256=sha(a.roster),full_witnesses_replayed=len(seen),
        distinct_normalized_views=len(seen),base_query_groups=len(groups),stages=sorted(stages),new_query_groups=0,F_labels=0,
        decision=r['decision'],auditor_sha256=sha(Path(__file__))))
    print(dict(views=len(seen),basegroups=len(groups),decision=r['decision']),flush=True)


if __name__=='__main__':main()
