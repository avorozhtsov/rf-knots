"""Bounded isolated replay; never saves or promotes diagnostic model weights."""
import inspect
import json
import math
import shutil
import time
from pathlib import Path
import torch
import train_continuous_random_view_r2_v24 as learner

OUT=Path('artifacts/v27-root-gradient-diagnostic-v1')

def main():
    assert not OUT.exists(), 'immutable diagnostic already exists'
    assert shutil.disk_usage('.').free>=12*1024**3
    core=learner.guard.core
    core.PROTOCOL=Path('research/437-audited-random-stream-refresh-optimizer-v24.md')
    spec,order,pins=learner.setup(Path('artifacts/continuous-r2-start-spec-v27.json'))
    run=Path(spec['out']);m=core.base.read(run/'manifest.json');e=m['checkpoints'][-1]
    assert e['step']==1384680
    assert core.base.sha(run/e['file'])==e['sha256']
    for p,h in m['contract']['inputs'].items():assert core.base.sha(Path(p))==h,p
    saved=torch.load(run/e['file'],map_location='cpu',weights_only=False)
    torch.set_num_threads(1)
    net,obj,head,opt,rng=core.engine.make_objects(saved)
    for k,v in [('model',net.state_dict()),('objective',obj.state_dict()),('auxiliary_head',head.state_dict()),('optimizer',opt.state_dict()),('rng',rng.getstate()),('torch_rng',torch.get_rng_state()),('mps_rng',torch.mps.get_rng_state())]:
        assert core.exact_tree(saved[k],v),k
    OUT.mkdir()
    contract=dict(source_checkpoint=str(run/e['file']),checkpoint_sha256=e['sha256'],source_step=e['step'],maximum_updates=4096,maximum_process_CPU_seconds=1800,device='native_mps',restore='bitwise_full_state_passed',promote=False,pins=pins)
    core.base.atomic(OUT/'contract.json',contract)
    original=learner.positive.base.encoder_norm
    observation={}
    def capture(*args,**kwargs):
        value=original(*args,**kwargs)
        frame=inspect.currentframe().f_back
        if frame.f_code.co_name=='root_only_step':
            observation.update(F_encoder_grad_norm=value,F_loss=float(frame.f_locals['f_loss'].detach()),gradient_finite=math.isfinite(value),gradient_above_floor=value>1e-12)
        return value
    learner.positive.base.encoder_norm=capture
    start=time.process_time();completed=0
    try:
        for step in range(e['step'],e['step']+4096):
            if time.process_time()-start>=1800 or shutil.disk_usage('.').free<12*1024**3:break
            net.train();obj.train();head.train()
            root,group,gi=core.base.selected_group(order,step)
            observation.clear()
            metrics=core.base.train_step(net,obj,head,opt,group,step)
            torch.mps.synchronize();learner.guard.validate(metrics);completed+=1
            if completed%128==0:
                core.base.atomic(OUT/'progress.json',dict(completed=completed,step=step+1,cpu_seconds=time.process_time()-start))
        result=dict(status='bounded_no_failure',completed=completed)
    except Exception as error:
        result=dict(status='reproduced_failure',error=repr(error),attempted_step=step+1,root_index=root['root_index'],group_index=gi,observation=observation,completed=completed)
    result.update(cpu_seconds=time.process_time()-start,weights_saved=False,active_run_unchanged=True)
    core.base.atomic(OUT/'report.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
