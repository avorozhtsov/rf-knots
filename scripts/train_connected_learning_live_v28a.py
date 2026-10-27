"""Native v28a with durable actual-consumption telemetry, never live bank edits."""
from pathlib import Path
import torch
import train_continuous_random_view_r2_v28a as predecessor
import train_continuous_random_view_r2_v28 as connected
import positive_sampler_v28 as sampler

COUNTS=None
SETTINGS=None


def joint_step(*args,**kwargs):
    result=connected.joint_step(*args,**kwargs)
    source=sampler.LAST['source'];ident=sampler.LAST['path_id']
    COUNTS[source]+=1
    if ident in SETTINGS['new_'+source+'_ids']:
        COUNTS['new_'+source]+=1
        if ident not in COUNTS['distinct_new_path_ids']:COUNTS['distinct_new_path_ids'].append(ident)
    return {**result,'connected_consumption':dict(COUNTS),
        'consumed_positive_bank_sha256':SETTINGS['positive_bank_sha256'],
        'consumed_random_bank_sha256':SETTINGS['random_positive_bank_sha256']}


def setup(path):
    global SETTINGS,COUNTS
    s,order,frozen=predecessor.setup(path);SETTINGS=s
    COUNTS=dict(retained=0,random=0,new_retained=0,new_random=0,distinct_new_path_ids=[])
    out=Path(s['out'])
    if (out/'manifest.json').exists():
        m=connected.old.guard.core.base.read(out/'manifest.json')
        if m['checkpoints']:
            c=m['checkpoints'][-1]
            assert connected.old.guard.core.base.sha(out/c['file'])==c['sha256']
            saved=torch.load(out/c['file'],map_location='cpu',weights_only=False)
            COUNTS=saved['metrics']['connected_consumption']
    connected.old.guard.core.base.train_step=joint_step
    for f in [Path(__file__),Path('scripts/connected_learning_supervisor_v28a.py'),
              Path('artifacts/connected-learning-live-v28a-protocol.json')]:
        frozen[str(f)]=connected.old.guard.core.base.sha(f)
    return s,order,frozen


if __name__=='__main__':
    core=connected.old.guard.core
    core.PROTOCOL=Path('artifacts/connected-learning-live-v28a-protocol.json')
    core.setup=setup;core.numerical_gate=connected.old.guard.gate
    core.main()
