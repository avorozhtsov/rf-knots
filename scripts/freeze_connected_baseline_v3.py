"""One immutable new A, before connected training commissioning. No scores."""
import shutil
import time
from pathlib import Path
import torch
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read

OUT=Path('artifacts/paired-learning-48h100k-v3')


def main():
    assert not OUT.exists(),'A namespace already exists: never replace baseline'
    assert shutil.disk_usage('.').free>=12*1024**3
    ref=read(Path('artifacts/continuous-r2-current-training.json'))
    run,spec=Path(ref['out']),Path(ref['spec']);m=read(run/'manifest.json')
    assert sha(spec)==ref['spec_sha256']
    assert m['status']=='checkpointed','freeze durable stopped parent, not unsaved live weights'
    e=m['checkpoints'][-1];cp=run/e['file'];assert sha(cp)==e['sha256']
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    assert saved['step']==e['step'] and len(saved['optimizer']['state'])==71
    assert all(k in saved for k in ['rng','torch_rng','mps_rng'])
    def finite(v):
        if torch.is_tensor(v):assert torch.isfinite(v).all()
        elif isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,(tuple,list)):
            for x in v:finite(x)
    finite(saved)
    data=Path('artifacts/continuous-train-mcts-v2');search=read(data/'manifest.json')
    OUT.mkdir()
    for source,name in [(cp,'baseline.pt'),(run/'manifest.json','baseline-training-manifest.json'),
                        (spec,'baseline-training-spec.json'),(data/'manifest.json','baseline-search-manifest.json')]:
        shutil.copy2(source,OUT/name);assert sha(source)==sha(OUT/name)
    atomic(OUT/'baseline.json',dict(created_unix=time.time(),checkpoint=str(OUT/'baseline.pt'),
        checkpoint_sha256=e['sha256'],step=e['step'],source_checkpoint=str(cp),run=str(run),
        spec=str(spec),spec_sha256=sha(spec),contract_sha256=m['contract_sha256'],
        checkpointed_active_origin_seconds=m['new_active_seconds'],main_search=str(data),
        first_new_task=search['next_task'],first_counted_task=((search['next_task']+31)//32)*32,
        note='New human-authorized A, before any connected native-MPS commissioning; no reset of optimizer/RNG.'))
    atomic(OUT/'status.json',dict(status='A_FROZEN_PANEL_NOT_YET_BUILT',step=e['step'],sha256=e['sha256']))
    print(read(OUT/'baseline.json'))


if __name__=='__main__':main()
