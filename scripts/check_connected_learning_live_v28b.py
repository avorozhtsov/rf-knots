"""Read-only end-to-end acceptance check. READY/PID alone cannot pass."""
import json
import time
from pathlib import Path
from run_continuous_train_mcts_v1 import read


def consumption_gate(counts,updates):
    assert counts['retained']+counts['random']==updates,'actual optimizer/source counter mismatch'
    assert abs(counts['retained']-counts['random'])<=1,'positive SOURCE ratio is not50:50'
    assert counts['new_random']>0,'no NEW random label reached AdamW'
    assert counts['distinct_new_path_ids'],'READY is not evidence of optimization'


def check():
    root=Path('artifacts/connected-learning-live-v28b');state=read(root/'state.json')
    assert not (root/'failure.json').exists(),'supervisor failed'
    ref=read(Path('artifacts/continuous-r2-current-training.json'))
    spec=read(Path(ref['spec']));out=Path(ref['out']);m=read(out/'manifest.json');p=read(out/'progress.json')
    assert m['status']=='training','learner is not training'
    assert time.time()-p['timestamp']<180,'stale LIVE progress'
    assert p['live_optimizer_steps']>ref['registered_step'],'launch is not advancing native learning'
    metrics=p['metrics'];counts=metrics['connected_consumption']
    consumption_gate(counts,p['live_optimizer_steps']-spec['parent_step'])
    assert metrics['consumed_positive_bank_sha256']==spec['positive_bank_sha256']
    assert metrics['consumed_random_bank_sha256']==spec['random_positive_bank_sha256']
    assert not state.get('supply_stopped'),'generation stopped: not a healthy continuous dataflow'
    assert not state.get('main_intake_stopped'),'main intake stopped'
    assert not (Path('artifacts/paired-learning-48h100k-v3')/'preparation-failure.json').exists(),'holdout preparation failed and blocks new supply'
    pending=state.get('pending_new_random_views',0)
    # Every round resets its local active-time counter, but never the model/RNG
    # or global sampler phase. Conservatively alarm on2durable+live active hours.
    active=m['new_active_seconds']+p.get('unsaved_active_seconds',0)
    assert not (pending>0 and active>7200),'READY backlog unused for>2activehours'
    return dict(status='VERIFIED_OPTIMIZER_CONSUMPTION',run=ref['out'],
        LIVE=p['live_optimizer_steps'],durable=m['optimizer_steps'],consumption=counts,
        new_generation='active_or_between_batches' if state.get('future_AB_panel_audited') else 'waiting_for_prospective_panel_audit',
        pending_new_random_views=pending)


if __name__=='__main__':print(json.dumps(check()))
