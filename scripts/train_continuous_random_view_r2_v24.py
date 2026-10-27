"""Generic immutable-snapshot learner; group/view-balanced random positives."""
from pathlib import Path
import train_continuous_hourly_r2_v17 as guard
import root_positive_joint_step_v1 as positive
import random_view_positive_sampler_v2 as mixture

def joint_step(*args,**kwargs):
    result=positive.joint_step(*args,**kwargs)
    return {**result,'positive_sampling_source':mixture.LAST_SOURCE,'random_positive_draw':int(mixture.LAST_SOURCE=='random'),
        'positive_exact_view_word':list(mixture.LAST_VIEW[0]),'positive_exact_view_strands':mixture.LAST_VIEW[1],
        'positive_observation_shard':mixture.LAST_SHARD}

def setup(path):
    s,order,frozen=guard.setup(path)
    for prefix in ['positive','random_positive']:
        bank=Path(s[prefix+'_bank']);audit=Path(s[prefix+'_audit']);r=guard.core.base.read(audit)
        assert guard.core.base.sha(bank/'manifest.json')==s[prefix+'_bank_sha256'] and guard.core.base.sha(audit)==s[prefix+'_audit_sha256']
        assert r['status']=='passed_data_only' and r['manifest_sha256']==s[prefix+'_bank_sha256']
        for f in [bank/'manifest.json',audit]:frozen[str(f)]=guard.core.base.sha(f)
    positive.initialize(Path(s['positive_bank']));mixture.initialize(Path(s['random_positive_bank']))
    mixture.OLD_SELECT=positive.select;positive.select=mixture.select;positive.BASE_STEP=guard.core.base.train_step;guard.core.base.train_step=joint_step
    guard.core.base.CONFIG.update(max_aux_encoder_ratio=.05,positive_encoder_cap=.05,random_positive_source_draw_fraction=.5,
        positive_scope='exact_zero_spent_TRAIN_root_only',positive_value_semantics='achieved_upper_bound_observation',
        positive_value_loss_weight=.1,positive_group_sampling='source50_groupuniform_viewuniform_bestpath')
    for f in [Path(__file__),Path(positive.__file__),Path(mixture.__file__),Path('scripts/random_positive_sampler_v1.py'),Path('tests/test_random_view_positive_sampler_v2.py')]:
        frozen[str(f)]=guard.core.base.sha(f)
    return s,order,frozen
if __name__=='__main__':
    guard.core.PROTOCOL=Path('research/437-audited-random-stream-refresh-optimizer-v24.md')
    guard.core.setup=setup;guard.core.numerical_gate=guard.gate;guard.core.main()
