"""Two isolated real AdamW updates, including a metadata-selected new main path."""
import argparse
import copy
import random
from pathlib import Path

import torch
from build_mcts_training_supervision import atomic, sha
from run_continuous_train_mcts_v1 import read
import train_additive_root_only_r2_v3 as prior
import root_positive_joint_step_v1 as positive
import positive_sampler_v28 as sampler


def main():
    p=argparse.ArgumentParser();p.add_argument('--main-bank',type=Path,required=True)
    p.add_argument('--random-bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('immutable isolated diagnostic exists')
    from learning_pipeline_v28 import disk_guard, pins_ok
    disk_guard()
    for bank in [a.main_bank,a.random_bank]:
        m=read(bank/'manifest.json');pins_ok(m['inputs'])
    ref=read(Path('artifacts/continuous-r2-current-training.json'));run=Path(ref['out'])
    m=read(run/'manifest.json');entry=m['checkpoints'][-1];cp=run/entry['file']
    assert sha(cp)==entry['sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False)
    torch.set_num_threads(1);base=positive.base;prior.setup_training()
    positive.BASE_STEP=base.train_step;base.CONFIG['max_aux_encoder_ratio']=.05
    sampler.initialize(a.main_bank,a.random_bank);positive.select=sampler.select
    new=set(read(a.main_bank/'manifest.json')['new_path_ids'])
    chosen=None
    for step in range(saved['step'],saved['step']+20000):
        row,_=sampler.select(step)
        if sampler.LAST['source']=='retained' and row['path_id'] in new:
            chosen=step;break
    if chosen is None:raise ValueError('new label unreachable within metadata window')
    net,obj=base.load_actor();net.encoder.graph_rounds=2;head=base.AuxiliaryCostHead()
    for key,model in [('model',net),('objective',obj),('auxiliary_head',head)]:
        model.load_state_dict(saved[key]);assert all(torch.equal(v.cpu(),model.state_dict()[k].cpu()) for k,v in saved[key].items())
    opt=torch.optim.AdamW([*net.parameters(),*obj.parameters(),*head.parameters()],lr=5e-5,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']);assert len(opt.state_dict()['state'])==71
    # Restore CPU RNG for this isolated diagnostic; original MPS RNG stays in cp.
    torch.set_rng_state(saved['torch_rng']);rng=random.Random();rng.setstate(saved['rng'])
    anchor=Path('artifacts/continuous-additive-f-bank-v59');e=read(anchor/'manifest.json')['roots'][-1]
    assert sha(anchor/e['file'])==e['sha256'];group=read(anchor/e['file'])['groups'][0]
    results=[]
    for step in [chosen,chosen+1]:
        before=copy.deepcopy(opt.state_dict());metrics=positive.joint_step(net,obj,head,opt,group,step)
        after=opt.state_dict()
        assert len(after['state'])==71 and metrics['real_AdamW_steps_this_update']==1
        for k,v in before['state'].items():
            assert float(after['state'][k]['step']) in [float(v['step']),float(v['step'])+1]
        assert metrics['positive_effective_F_ratio']<=.050001
        assert metrics['aux_encoder_grad_norm']<=.100001*metrics['F_encoder_grad_norm']
        results.append(dict(sampler_step=step,selection=copy.deepcopy(sampler.LAST),metrics=metrics))
    assert {r['selection']['source'] for r in results}=={'random','retained'}
    assert results[0]['selection']['path_id'] in new
    for state in [net.state_dict(),obj.state_dict(),head.state_dict(),opt.state_dict()['state']]:
        def finite(v):
            if torch.is_tensor(v):assert torch.isfinite(v).all()
            elif isinstance(v,dict):
                for x in v.values():finite(x)
        finite(state)
    assert sha(cp)==entry['sha256']
    a.out.mkdir();atomic(a.out/'report.json',dict(status='passed_isolated_CPU_v28',updates=2,
        source_checkpoint=str(cp),source_sha256=entry['sha256'],AdamW_states=71,
        positive_bank_sha256=sha(a.main_bank/'manifest.json'),
        random_positive_bank_sha256=sha(a.random_bank/'manifest.json'),
        metadata_offset=chosen-saved['step'],results=results,weights_saved=False,
        active_training_modified=False,native_commissioning=False))
    print(dict(status='passed_isolated_CPU_v28',new_main_path=results[0]['selection']['path_id'],
               metadata_offset=chosen-saved['step']))


if __name__=='__main__':main()
