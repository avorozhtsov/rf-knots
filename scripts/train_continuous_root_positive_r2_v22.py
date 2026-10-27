"""Hourly F-primary R2 plus independently audited root-only MCTS positives."""
from pathlib import Path
import train_continuous_hourly_r2_v17 as guard
import root_positive_joint_step_v1 as positive


def setup(path):
    s,order,frozen=guard.setup(path)
    bank=Path(s['positive_bank']);audit=Path(s['positive_audit'])
    if guard.core.base.sha(bank/'manifest.json')!=s['positive_bank_sha256'] or guard.core.base.sha(audit)!=s['positive_audit_sha256']:
        raise ValueError('positive data audit pin differs')
    report=guard.core.base.read(audit)
    if report['status']!='passed_data_only' or report['manifest_sha256']!=s['positive_bank_sha256']:
        raise ValueError('independent positive audit failed')
    positive.initialize(bank);positive.BASE_STEP=guard.core.base.train_step
    guard.core.base.CONFIG.update(max_aux_encoder_ratio=.05,positive_encoder_cap=.05,
        positive_scope='exact_zero_spent_TRAIN_root_only',positive_value_semantics='achieved_upper_bound_observation',
        positive_value_loss_weight=.1,positive_group_sampling='uniform_group_best_observed_weighted_cost')
    guard.core.base.train_step=positive.joint_step
    for f in [Path(__file__),Path(positive.__file__),bank/'manifest.json',audit,
              Path('research/427-root-positive-policy-value-f-dominant-optimizer-v22.md'),
              Path('tests/test_root_positive_joint_step_v1.py')]:frozen[str(f)]=guard.core.base.sha(f)
    return s,order,frozen


if __name__=='__main__':
    guard.core.PROTOCOL=Path('research/427-root-positive-policy-value-f-dominant-optimizer-v22.md')
    guard.core.setup=setup;guard.core.numerical_gate=guard.gate;guard.core.main()
