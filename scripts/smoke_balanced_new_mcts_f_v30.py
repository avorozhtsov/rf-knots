"""Six isolated CPU updates test both sources at all preregistered warmup stages."""
import argparse
import copy
import random
from pathlib import Path
import torch
import balanced_new_mcts_f_v29 as balanced
import train_additive_root_only_r2_v3 as prior
import connected_learning_supervisor_v30 as supervisor


def main():
    p=argparse.ArgumentParser()
    for k in ['main-ready','random-ready','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();supervisor.disk_guard();assert not a.out.exists()
    s=supervisor.configure(a.main_ready,a.random_ready)
    assert supervisor.sha(Path(s['parent']))==s['parent_sha256']
    saved=torch.load(s['parent'],map_location='cpu',weights_only=False)
    torch.set_num_threads(1);prior.setup_training();base=balanced.base
    net,obj=base.load_actor();net.encoder.graph_rounds=2;head=base.AuxiliaryCostHead()
    for key,model in [('model',net),('objective',obj),('auxiliary_head',head)]:
        model.load_state_dict(saved[key]);assert all(torch.equal(v,model.state_dict()[k]) for k,v in saved[key].items())
    opt=torch.optim.AdamW([*net.parameters(),*obj.parameters(),*head.parameters()],lr=5e-5,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']);assert len(opt.state_dict()['state'])==71
    torch.set_rng_state(saved['torch_rng']);random.setstate(saved['rng']);assert 'mps_rng' in saved
    balanced.initialize(s)
    bank=Path(s['bank']);root=base.read(bank/base.read(bank/'manifest.json')['roots'][-1]['file'])
    group=root['groups'][0];results=[]
    for offset in [0,1,2000,2001,4000,4001]:
        step=s['balance_origin_step']+offset
        before=copy.deepcopy(opt.state_dict())
        metrics=balanced.train_step(net,obj,head,opt,group,step);balanced.numerical_gate(metrics)
        for k,v in before['state'].items():assert 0<=float(opt.state_dict()['state'][k]['step'])-float(v['step'])<=1
        results.append(dict(offset=offset,source=metrics['positive_sampling_source'],
            actual_share=metrics['actual_MCTS_norm_share'],target_share=metrics['target_MCTS_norm_share'],
            F_norm=metrics['F_effective_full_norm'],MCTS_norm=metrics['MCTS_effective_full_norm'],
            path_id=metrics['positive_admitted_path_id']))
    assert {r['source'] for r in results}=={'random','retained'}
    supervisor.finite([net.state_dict(),obj.state_dict(),head.state_dict(),opt.state_dict()])
    assert supervisor.sha(Path(s['parent']))==s['parent_sha256']
    a.out.mkdir()
    supervisor.atomic(a.out/'report.json',dict(status='passed_balanced_CPU_v29',updates=6,
        AdamW_states=71,source_sha256=s['parent_sha256'],positive_bank_sha256=s['positive_bank_sha256'],
        random_positive_bank_sha256=s['random_positive_bank_sha256'],results=results,
        inputs={str(f):supervisor.sha(f) for f in [Path(__file__),Path(balanced.__file__),
            Path(supervisor.__file__),Path('scripts/train_connected_learning_live_v30.py'),
            Path('tests/test_balanced_new_mcts_f_v29.py'),Path('artifacts/connected-learning-live-v30-protocol.json')]},
        weights_saved=False,active_clock_modified=False,metadata_offsets_not_native_training=True))
    print(dict(status='passed_balanced_CPU_v29',results=results))


if __name__=='__main__':main()
