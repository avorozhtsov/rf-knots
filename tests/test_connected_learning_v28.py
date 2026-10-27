"""Safety and data-consumption contracts; no live model/worker writes."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import advance, BatchedPolicyContinuation
from rf_knots.site_proposals import proposal_candidates
from verified_reduction_search_v28 import automatic_action, reduce_trace, ReductionEvaluator
from pretrained_raster_policy import PriorOnlyResult
import positive_sampler_v28 as sampler


def inst(word=(1,-1,1), moves=8):
    return M.Instance('test',tuple(word),2,1,cap_moves=moves,growth_len=2,growth_strands=1)


def test_reduction_is_elementary_and_preserves_CC():
    i=inst();s=M.SearchState(i.word,i.strands,0,0);after,actions=reduce_trace(i,s)
    assert after.key==M.SOLVED and after.used_cc==0 and after.used_moves==2 and len(actions)==2


def test_cap_is_not_bypassed():
    i=inst(moves=1);s=M.SearchState(i.word,i.strands,0,0);after,actions=reduce_trace(i,s)
    assert after.key!=M.SOLVED and after.used_moves==1 and len(actions)==1


def test_no_automatic_crossing_change():
    i=inst((1,1,1));s=M.SearchState(i.word,i.strands,0,0)
    assert automatic_action(i,s,proposal_candidates(i,s,4)) is None


class FailNeural:
    def evaluate(self,requests):raise AssertionError('unnecessary neural forward')


def test_auto_continuation_uses_no_neural_and_keeps_full_trace():
    i=inst();s=M.SearchState(i.word,i.strands,0,0);ev=ReductionEvaluator(FailNeural())
    result=BatchedPolicyContinuation(ev,horizon=i.cap_moves,insertion_candidates=4).run([(i,s)],seeds=[1])[0]
    assert result.stop_reason=='solved' and len(result.actions)==2
    assert ev.automatic_requests==2 and ev.neural_requests==0
    assert result.end.used_moves==2 and result.end.used_cc==0


def test_fallback_is_used_on_nonmechanical_state():
    class Uniform:
        def evaluate(self,requests):return [PriorOnlyResult(np.ones(len(r[2]))/len(r[2])) for r in requests]
    i=inst((1,1,1));s=M.SearchState(i.word,i.strands,0,0);actions=proposal_candidates(i,s,4)
    ev=ReductionEvaluator(Uniform());r=ev.evaluate([(i,s,actions)])
    assert ev.neural_requests==1 and ev.automatic_requests==0 and np.isclose(r[0].priors.sum(),1)


def test_sampler_exposes_all_semantic_paths_without_observation_weight():
    i=inst();base=dict(instance=i.to_dict(),observations=[{'shard':'one'}])
    rows=[{**base,'path_id':str(n)} for n in range(3)]
    groups=[((3,2),[((i.word,i.strands),rows)])]
    with patch.object(sampler,'SOURCES',{'retained':groups,'random':groups}):
        seen={'retained':set(),'random':set()}
        for step in range(351232,351244):
            r,_=sampler.select(step);seen[sampler.LAST['source']].add(r['path_id'])
        assert seen=={'retained':{'0','1','2'},'random':{'0','1','2'}}


def test_sampler_rejects_semantic_duplicates(tmp_path):
    from build_mcts_training_supervision import sha
    i=inst();r=dict(instance=i.to_dict(),path_id='same',query_group=[3,2])
    (tmp_path/'row.json').write_text(json.dumps(r))
    e=dict(file='row.json',sha256=sha(tmp_path/'row.json'))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(paths=[e,e])))
    with pytest.raises(ValueError,match='semantic duplicate'):sampler.load_bank(tmp_path)


def test_live_parent_cannot_be_handed_off_implicitly(tmp_path):
    from learning_pipeline_v28 import prepare_spec
    def fake_read(p):
        if str(p).endswith('current-training.json'):return {'spec':'spec','out':'live'}
        if str(p)=='spec':return {}
        return {'status':'training','stop_reason':'storage_gate'}
    with patch('learning_pipeline_v28.read',side_effect=fake_read):
        with pytest.raises(ValueError,match='graceful parent handoff'):
            prepare_spec(tmp_path/'new.json',Path('a'),Path('b'),Path('c'))


def test_heldout_root_is_never_admitted():
    from rf_knots.evidence import UnknotWitness
    from learning_pipeline_v28 import admit
    from rf_knots.embedding_normalization import fast_normalize_global_view as norm
    i=inst();s=M.SearchState(i.word,i.strands,0,0);_,actions=reduce_trace(i,s)
    w=UnknotWitness.from_actions(i.word,i.strands,i.spec,actions)
    with patch('learning_pipeline_v28.owner',return_value='train'):
        with pytest.raises(ValueError,match='heldout'):
            admit(w,i,set(),{norm(i.word,i.strands)})
