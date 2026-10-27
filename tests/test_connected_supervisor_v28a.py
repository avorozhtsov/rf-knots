import pytest
import connected_learning_supervisor_v28a as supervisor
import train_connected_learning_live_v28a as trainer


def test_commissioning_executes_all_slots_until_first_new_random(monkeypatch):
    monkeypatch.setattr(supervisor.eligible,'initialize',lambda *a:None)
    def select(step):
        supervisor.sampler.LAST={'source':'random' if step%2 else 'retained'}
        return {'path_id':'NEW' if step==105 else 'old'},None
    monkeypatch.setattr(supervisor.sampler,'select',select)
    assert supervisor.first_new_random(100,'main','random',{'NEW'})==(6,'NEW')


def test_no_forced_sampling_or_phase_reset(monkeypatch):
    monkeypatch.setattr(supervisor.eligible,'initialize',lambda *a:None)
    def select(step):
        supervisor.sampler.LAST={'source':'random'}
        return {'path_id':'old'},None
    monkeypatch.setattr(supervisor.sampler,'select',select)
    with pytest.raises(ValueError,match='no NEW random draw'):
        supervisor.first_new_random(100,'main','random',{'NEW'})


def test_actual_consumption_not_ready_counts(monkeypatch):
    monkeypatch.setattr(trainer,'COUNTS',dict(retained=0,random=0,new_retained=0,new_random=0,distinct_new_path_ids=[]))
    monkeypatch.setattr(trainer,'SETTINGS',dict(new_random_ids=['new'],new_retained_ids=[],
        positive_bank_sha256='main-sha',random_positive_bank_sha256='random-sha'))
    monkeypatch.setattr(trainer.connected,'joint_step',lambda *a,**k:{'real_AdamW_steps_this_update':1})
    monkeypatch.setattr(trainer.sampler,'LAST',dict(source='random',path_id='new'))
    m=trainer.joint_step()
    assert m['connected_consumption']['new_random']==1
    assert m['connected_consumption']['distinct_new_path_ids']==['new']
    assert m['consumed_random_bank_sha256']=='random-sha'


def test_nonfinite_full_state_fails():
    import torch
    with pytest.raises(AssertionError):supervisor.finite({'moment':torch.tensor(float('nan'))})
