"""Isolated two-update diagnostic: a genuinely new path from either source.

Metadata-selected future slots exercise wiring only, not native continuation.
No weights are saved and no training-age counter or live worker is modified.
"""
import argparse
import copy
import random
from pathlib import Path
import torch
from build_mcts_training_supervision import atomic, sha
from run_continuous_train_mcts_v1 import read
from learning_pipeline_v28 import disk_guard, pins_ok
import train_additive_root_only_r2_v3 as prior
import root_positive_joint_step_v1 as positive
import positive_sampler_v28 as sampler
import positive_sampler_v28a as eligible


def main():
    p=argparse.ArgumentParser()
    for name in ['main-bank','main-audit','random-bank','random-audit','out']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('immutable diagnostic exists')
    disk_guard();new={}
    for source,bank,audit in [('retained',a.main_bank,a.main_audit),('random',a.random_bank,a.random_audit)]:
        m=read(bank/'manifest.json');ar=read(audit);pins_ok(m['inputs'])
        assert ar['status']=='passed_data_only' and ar['manifest_sha256']==sha(bank/'manifest.json')
        new[source]=set(m['new_path_ids'])-set(m.get('quarantined_path_ids',[]))
    ref=read(Path('artifacts/continuous-r2-current-training.json'));run=Path(ref['out'])
    e=read(run/'manifest.json')['checkpoints'][-1];cp=run/e['file'];assert sha(cp)==e['sha256']
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    torch.set_num_threads(1);base=positive.base;prior.setup_training()
    positive.BASE_STEP=base.train_step;base.CONFIG['max_aux_encoder_ratio']=.05
    eligible.initialize(a.main_bank,a.random_bank);positive.select=sampler.select
    chosen=None
    for step in range(saved['step'],saved['step']+20000):
        row,_=sampler.select(step)
        if row['path_id'] in new[sampler.LAST['source']]:chosen=step;break
    assert chosen is not None,'no new MCTS label reachable in metadata window'
    net,obj=base.load_actor();net.encoder.graph_rounds=2;head=base.AuxiliaryCostHead()
    for name,model in [('model',net),('objective',obj),('auxiliary_head',head)]:
        model.load_state_dict(saved[name])
        assert all(torch.equal(v.cpu(),model.state_dict()[k].cpu()) for k,v in saved[name].items())
    opt=torch.optim.AdamW([*net.parameters(),*obj.parameters(),*head.parameters()],lr=5e-5,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']);assert len(opt.state_dict()['state'])==71
    torch.set_rng_state(saved['torch_rng']);random.setstate(saved['rng'])
    assert 'mps_rng' in saved # retained in source checkpoint, not installed on CPU
    anchor=Path('artifacts/continuous-additive-f-bank-v59');entry=read(anchor/'manifest.json')['roots'][-1]
    assert sha(anchor/entry['file'])==entry['sha256'];group=read(anchor/entry['file'])['groups'][0]
    results=[]
    for step in [chosen,chosen+1]:
        before=copy.deepcopy(opt.state_dict());metrics=positive.joint_step(net,obj,head,opt,group,step)
        after=opt.state_dict();assert len(after['state'])==71 and metrics['real_AdamW_steps_this_update']==1
        for k,v in before['state'].items():
            assert float(after['state'][k]['step']) in [float(v['step']),float(v['step'])+1]
        assert metrics['positive_effective_F_ratio']<=.050001
        assert metrics['aux_encoder_grad_norm']<=.100001*metrics['F_encoder_grad_norm']
        results.append(dict(sampler_step=step,selection=copy.deepcopy(sampler.LAST),metrics=metrics))
    assert {r['selection']['source'] for r in results}=={'retained','random'}
    first=results[0]['selection'];assert first['path_id'] in new[first['source']]
    def finite(v):
        if torch.is_tensor(v):assert torch.isfinite(v).all()
        elif isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,(list,tuple)):
            for x in v:finite(x)
    finite([net.state_dict(),obj.state_dict(),head.state_dict(),opt.state_dict()])
    assert sha(cp)==e['sha256'];a.out.mkdir()
    atomic(a.out/'report.json',dict(status='passed_isolated_CPU_v28',updates=2,
        source_checkpoint=str(cp),source_sha256=e['sha256'],AdamW_states=71,
        positive_bank_sha256=sha(a.main_bank/'manifest.json'),positive_audit_sha256=sha(a.main_audit),
        random_positive_bank_sha256=sha(a.random_bank/'manifest.json'),random_positive_audit_sha256=sha(a.random_audit),
        metadata_offset=chosen-saved['step'],results=results,weights_saved=False,
        active_training_modified=False,native_commissioning=False,diagnostic_sha256=sha(Path(__file__))))
    print(dict(status='passed_isolated_CPU_v28',new_path=first['path_id'],source=first['source']))


if __name__=='__main__':main()
