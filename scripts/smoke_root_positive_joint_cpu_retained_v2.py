"""Isolated one-update CPU diagnostic; never changes active learner/checkpoint."""
import argparse
import copy
import math
from pathlib import Path
import torch
import train_additive_root_only_r2_v3 as prior
import root_positive_joint_step_v1 as positive
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('immutable diagnostic output exists')
    bank=Path('artifacts/mcts-root-positive-bank-v1');audit=Path('artifacts/mcts-root-positive-bank-audit-v1/report.json')
    assert read(audit)['status']=='passed_data_only' and read(audit)['manifest_sha256']==sha(bank/'manifest.json')
    parent=Path('artifacts/continuous-r2-v21');m=read(parent/'manifest.json');e=m['checkpoints'][-1]
    assert sha(parent/e['file'])==e['sha256'];saved=torch.load(parent/e['file'],map_location='cpu',weights_only=False)
    torch.set_num_threads(1);base=positive.base;prior.setup_training();positive.BASE_STEP=base.train_step
    base.CONFIG['max_aux_encoder_ratio']=.05;positive.initialize(bank)
    net,objective=base.load_actor();net.encoder.graph_rounds=2;head=base.AuxiliaryCostHead()
    net.load_state_dict(saved['model']);objective.load_state_dict(saved['objective']);head.load_state_dict(saved['auxiliary_head'])
    parameters=[*net.parameters(),*objective.parameters(),*head.parameters()]
    optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=.01);optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng']);before=copy.deepcopy(optimizer.state_dict())
    anchor=Path('artifacts/continuous-additive-f-bank-v59');am=read(anchor/'manifest.json');ae=next(e for e in am['roots'] if read(anchor/e['file'])['groups'] and read(anchor/e['file'])['groups'][0].get('gradient_scope')!='exact_zero_spent_root_only')
    assert sha(anchor/ae['file'])==ae['sha256'];group=read(anchor/ae['file'])['groups'][0]
    metrics=positive.joint_step(net,objective,head,optimizer,group,saved['step'])
    after=optimizer.state_dict();assert len(before['state'])==len(after['state'])==71
    for k,v in before['state'].items():assert float(after['state'][k]['step']) in [float(v['step']),float(v['step'])+1]
    assert metrics['real_AdamW_steps_this_update']==1 and metrics['positive_encoder_grad_norm']>0
    f=metrics['F_encoder_grad_norm'];assert math.isfinite(f) and f>1e-12
    assert metrics['positive_effective_F_ratio']<=.050001 and metrics['aux_encoder_grad_norm']<=.100001*f
    for t in [*net.state_dict().values(),*objective.state_dict().values(),*head.state_dict().values()]:
        if torch.is_tensor(t):assert torch.isfinite(t).all()
    inputs=[Path(__file__),Path(positive.__file__),Path('scripts/train_continuous_root_positive_r2_v22.py'),
        Path('research/427-root-positive-policy-value-f-dominant-optimizer-v22.md'),Path('tests/test_root_positive_joint_step_v1.py'),
        bank/'manifest.json',audit,anchor/'manifest.json',parent/e['file']]
    a.out.mkdir();atomic(a.out/'report.json',dict(status='passed_isolated_CPU_diagnostic',source_step=saved['step'],
        source_checkpoint_sha256=e['sha256'],metrics=metrics,AdamW_states=71,optimizer_updates=1,
        active_training_modified=False,model_published=False,inputs={str(f):sha(f) for f in inputs}))
    print(metrics,flush=True)


if __name__=='__main__':main()
