"""Read-only proof of real fresh-label updates and measured gradient budgets."""
import time
from pathlib import Path
from build_mcts_training_supervision import sha
from run_continuous_train_mcts_v1 import read
from balanced_new_mcts_f_v29 import mcts_weight,numerical_gate


def check():
    root=Path('artifacts/connected-learning-live-v29');state=read(root/'state.json')
    assert state['status'] not in ['STOPPED_LEARNER','DEGRADED_DATAFLOW'],'supervisor is not running healthy training'
    assert not (root/'failure.json').exists()
    assert not state.get('supply_stopped') and not state.get('main_intake_stopped'),'degraded dataflow'
    r=read(Path('artifacts/continuous-r2-current-training.json'));s=read(Path(r['spec']))
    assert sha(Path(r['spec']))==r['spec_sha256'] and s['version']=='connected-v29'
    initial=read(Path(read(root/'prepared.json')['spec']))
    assert s['balance_origin_step']==initial['balance_origin_step'],'warmup reset at bank handoff'
    out=Path(r['out']);m=read(out/'manifest.json');p=read(out/'progress.json')
    assert m['status']=='training' and time.time()-p['timestamp']<180
    live=p['live_optimizer_steps'];assert live>r['registered_step'],'awaiting real post-commissioning progress'
    metrics=p['metrics'];numerical_gate(metrics)
    assert metrics['balance_origin_step']==s['balance_origin_step']
    assert metrics['balance_update']==live-s['balance_origin_step']
    assert abs(metrics['actual_MCTS_norm_share']-mcts_weight(live-1,s['balance_origin_step']))<1e-6
    c=metrics['connected_consumption']
    assert c['retained']+c['random']==live-s['parent_step'] and abs(c['retained']-c['random'])<=1
    assert c['retained']==c['new_retained']>0 and c['random']==c['new_random']>0
    assert set(c['distinct_new_path_ids'])<=set(s['fresh_retained_ids'])|set(s['fresh_random_ids'])
    for source,prefix in [('retained','positive'),('random','random_positive')]:
        key='consumed_'+('positive' if source=='retained' else 'random')+'_bank_sha256'
        assert metrics[key]==s[prefix+'_bank_sha256']
    return dict(status='VERIFIED_FRESH_MCTS_AND_F_GRADIENT_BUDGET',LIVE=live,durable=m['optimizer_steps'],
        balance_origin_step=s['balance_origin_step'],actual_MCTS_share=metrics['actual_MCTS_norm_share'],
        target_MCTS_share=metrics['target_MCTS_norm_share'],F_norm=metrics['F_effective_full_norm'],
        MCTS_norm=metrics['MCTS_effective_full_norm'],extra_auxiliary=metrics['aux_loss'],
        extra_rehearsal=metrics['rehearsal_encoder_grad_norm'],
        counts={k:v for k,v in c.items() if k!='distinct_new_path_ids'},
        distinct_fresh_paths=len(c['distinct_new_path_ids']))


if __name__=='__main__':print(check())
