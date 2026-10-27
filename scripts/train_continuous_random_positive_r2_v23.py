"""Separate hourly native-MPS continuation with real random-root draws."""
from pathlib import Path
import train_continuous_hourly_r2_v17 as guard
import root_positive_joint_step_v1 as positive
import random_positive_sampler_v1 as mixture

def joint_step(*args,**kwargs):
    result=positive.joint_step(*args,**kwargs)
    return {**result,'positive_sampling_source':mixture.LAST_SOURCE,'random_positive_draw':int(mixture.LAST_SOURCE=='random')}

def setup(path):
    s,order,frozen=guard.setup(path)
    for prefix in ['positive','random_positive']:
        bank=Path(s[prefix+'_bank']);audit=Path(s[prefix+'_audit']);report=guard.core.base.read(audit)
        assert guard.core.base.sha(bank/'manifest.json')==s[prefix+'_bank_sha256']
        assert guard.core.base.sha(audit)==s[prefix+'_audit_sha256'] and report['status']=='passed_data_only'
        assert report['manifest_sha256']==s[prefix+'_bank_sha256']
        for f in [bank/'manifest.json',audit]:frozen[str(f)]=guard.core.base.sha(f)
    positive.initialize(Path(s['positive_bank']));mixture.initialize(Path(s['random_positive_bank']))
    mixture.OLD_SELECT=positive.select;positive.select=mixture.select
    positive.BASE_STEP=guard.core.base.train_step;guard.core.base.train_step=joint_step
    guard.core.base.CONFIG.update(max_aux_encoder_ratio=.05,positive_encoder_cap=.05,random_positive_source_draw_fraction=.5,
        positive_scope='exact_zero_spent_TRAIN_root_only',positive_value_semantics='achieved_upper_bound_observation',
        positive_value_loss_weight=.1,positive_group_sampling='source_50_50_then_uniform_group_best_observed_cost')
    for f in [Path(__file__),Path(positive.__file__),Path(mixture.__file__),Path('tests/test_random_positive_sampler_v1.py')]:
        frozen[str(f)]=guard.core.base.sha(f)
    return s,order,frozen

if __name__=='__main__':
    guard.core.PROTOCOL=Path('research/434-random-positive-mixture-optimizer-v23.md')
    guard.core.setup=setup;guard.core.numerical_gate=guard.gate;guard.core.main()
