"""Separate balanced version; reuse full-state persistence, not old objective."""
from pathlib import Path
import torch
import train_connected_learning_live_v28b as parent
import balanced_new_mcts_f_v29 as balanced


def setup(path):
    s,order,frozen=parent.setup(path)
    out=Path(s['out']);saved=None
    if (out/'manifest.json').exists():
        m=balanced.base.read(out/'manifest.json')
        if m['checkpoints']:
            c=m['checkpoints'][-1];assert balanced.base.sha(out/c['file'])==c['sha256']
            saved=torch.load(out/c['file'],map_location='cpu',weights_only=False)
    if saved is None:
        # Carry durable EMA between v29 rounds, but not legacy objective counters.
        saved=torch.load(s['parent'],map_location='cpu',weights_only=False)
    balanced.initialize(s,saved)
    balanced.base.train_step=balanced.train_step
    balanced.base.CONFIG.update(objective='NEW_MCTS_vs_F_balanced_full_norm_v29',
        extra_auxiliary=False,extra_rehearsal=False,balance_origin_step=s['balance_origin_step'],
        max_aux_encoder_ratio=0.,max_rehearsal_encoder_ratio=0.,positive_encoder_cap=None,
        positive_scope='fresh_verified_MCTS_root_only',
        positive_value_semantics='achieved_upper_bound_inside_MCTS_budget',
        gradient_budget='full_parameter_vector_MCTS20_35_50_F80_65_50')
    for f in [Path(__file__),Path(balanced.__file__),Path('scripts/connected_learning_supervisor_v29.py'),
              Path('scripts/intake_fresh_main_v29.py'),Path('tests/test_balanced_new_mcts_f_v29.py'),
              Path('scripts/smoke_balanced_new_mcts_f_v29.py'),Path('scripts/collect_connected_learning_v29.py'),
              Path('scripts/prepare_balanced_continuation_v29.py'),
              Path('scripts/audit_fresh_main_eligibility_v29.py'),Path('scripts/intake_fresh_main_v29a.py'),
              Path('artifacts/connected-learning-live-v29-protocol.json')]:frozen[str(f)]=balanced.base.sha(f)
    return s,order,frozen


if __name__=='__main__':
    core=parent.parent.connected.old.guard.core
    core.PROTOCOL=Path('artifacts/connected-learning-live-v29-protocol.json')
    core.setup=setup;core.numerical_gate=balanced.numerical_gate;core.main()
