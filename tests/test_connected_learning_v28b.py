"""Prospective supply/intake regressions; never writes live artifacts."""
from pathlib import Path
from unittest.mock import patch
import pytest
import collect_connected_learning_v28b as collector
import connected_learning_supervisor_v28b as supervisor
from rf_knots.conjugate_witnesses import conjugate_witness


def test_fixed_shallow_size_and_proofs_before_any_scores():
    for index in range(512,768):
        w=conjugate_witness(collector.pipeline.SEED,index,max_strands=collector.MAX_STRANDS,
            min_depth=collector.MIN_DEPTH,max_depth=collector.MAX_DEPTH)
        w.verify()
        assert w.start.strands<=6 and len(w.start.word)<=41 and w.moves<=25


def test_closed_attempt_indices_cannot_be_retried(tmp_path):
    with pytest.raises(AssertionError,match='cannot be retried'):
        collector.generate(tmp_path/'supply',1,[])


def test_all_failed_and_unsolved_rosters_remain_excluded():
    rosters=collector.prior_rosters()
    assert Path('artifacts/random-TRAIN-stream-v28/batch-000000/supply.json') in rosters
    assert Path('artifacts/random-TRAIN-stream-v28a/batch-000001/supply.json') in rosters


def test_no_early_or_tiny_checkpoint_but_semantic_path_trigger_works():
    assert not supervisor.refresh_due(3599,100,100)
    assert not supervisor.refresh_due(3600,3,15)
    assert supervisor.refresh_due(3600,4,0)
    assert supervisor.refresh_due(3600,0,16)


def test_degraded_source_cannot_be_called_verified_active():
    assert supervisor.health_status(True,True,False)=='DEGRADED_DATAFLOW'
    assert supervisor.health_status(True,False,True)=='DEGRADED_DATAFLOW'
    assert supervisor.health_status(True,False,False)=='VERIFIED_ACTIVE'


def test_commissioning_does_not_force_source_or_reset_phase():
    with patch.object(supervisor.eligible,'initialize'),patch.object(supervisor.sampler,'select') as select:
        def draw(step):
            supervisor.sampler.LAST={'source':'retained' if step%2==0 else 'random'}
            return {'path_id':'new' if step==104 else 'old'},None
        select.side_effect=draw
        assert supervisor.first_new_path(100,'main','random',{'retained':{'new'},'random':set()})==(5,'new')


def test_no_authorized_new_path_means_fail_not_retry():
    with patch.object(supervisor.eligible,'initialize'),patch.object(supervisor.sampler,'select') as select:
        supervisor.sampler.LAST={'source':'random'};select.return_value=({'path_id':'old'},None)
        with pytest.raises(ValueError,match='no NEW path'):
            supervisor.first_new_path(100,'main','random',{'retained':set(),'random':set()})
