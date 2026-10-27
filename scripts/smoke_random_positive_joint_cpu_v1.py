"""Two real isolated updates; no mutation of active learner state."""
import argparse,copy,math
from pathlib import Path
import torch
import train_additive_root_only_r2_v3 as prior
import root_positive_joint_step_v1 as pos
import random_positive_sampler_v1 as mix
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('immutable output')
    parent=Path('artifacts/continuous-r2-v22');m=read(parent/'manifest.json');e=m['checkpoints'][-1];cp=parent/e['file'];assert sha(cp)==e['sha256']
    s=torch.load(cp,map_location='cpu',weights_only=False);torch.set_num_threads(1);b=pos.base;prior.setup_training();pos.BASE_STEP=b.train_step
    b.CONFIG['max_aux_encoder_ratio']=.05;pos.initialize(Path('artifacts/mcts-root-positive-bank-v1'));mix.OLD_SELECT=pos.select
    mix.initialize(Path('artifacts/random-root-positive-bank-v1'));pos.select=mix.select
    net,obj=b.load_actor();net.encoder.graph_rounds=2;head=b.AuxiliaryCostHead();net.load_state_dict(s['model']);obj.load_state_dict(s['objective']);head.load_state_dict(s['auxiliary_head'])
    params=[*net.parameters(),*obj.parameters(),*head.parameters()];opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01);opt.load_state_dict(s['optimizer'])
    anchor=Path('artifacts/continuous-additive-f-bank-v59');entry=read(anchor/'manifest.json')['roots'][-1];assert sha(anchor/entry['file'])==entry['sha256'];group=read(anchor/entry['file'])['groups'][0]
    results=[]
    for step in [s['step'],s['step']+1]:
        before=copy.deepcopy(opt.state_dict());x=pos.joint_step(net,obj,head,opt,group,step);after=opt.state_dict();assert len(after['state'])==71
        for k,v in before['state'].items():assert float(after['state'][k]['step']) in (float(v['step']),float(v['step'])+1)
        assert x['real_AdamW_steps_this_update']==1 and x['positive_encoder_grad_norm']>0 and x['positive_effective_F_ratio']<=.050001
        assert x['aux_encoder_grad_norm']<=.100001*x['F_encoder_grad_norm']
        results.append(dict(source=mix.LAST_SOURCE,metrics=x))
    assert {r['source'] for r in results}=={'random','retained'}
    assert all(torch.isfinite(t).all() for t in net.state_dict().values())
    a.out.mkdir();atomic(a.out/'report.json',dict(status='passed_isolated_CPU',source_checkpoint=str(cp),source_sha256=sha(cp),updates=2,AdamW_states=71,results=results,active_training_modified=False))
    print([(r['source'],r['metrics']['positive_query_group'],r['metrics']['positive_effective_F_ratio']) for r in results])
if __name__=='__main__':main()
