"""Independent read/restore/count/sampler verification; never optimizer/search/evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

def snapshot():
    import connected_learning_supervisor_v30 as supervisor
    import positive_sampler_v28a as eligible
    import positive_sampler_v28 as sampler
    import balanced_new_mcts_f_v29 as balanced
    from paired_learning_comparison_v3 import gate
    spec, parent, manifest, entry, saved = supervisor.validate_parent()
    assert entry['step'] == 2056741
    assert spec['balance_origin_step'] == 1645639
    assert saved['metrics']['balance_origin_step'] == 1645639
    assert saved['metrics']['balance_ema_f'] > 0
    baseline=Path('artifacts/paired-learning-48h100k-v3/baseline.pt')
    assert supervisor.sha(baseline)=='a2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c'
    banks={}
    for prefix in ['positive','random_positive']:
        bank=Path(spec[prefix+'_bank']);m=supervisor.read(bank/'manifest.json')
        assert supervisor.sha(bank/'manifest.json')==spec[prefix+'_bank_sha256']
        for row in m['paths']:
            assert supervisor.sha(bank/row['file'])==row['sha256']
        banks[prefix]=dict(sha=spec[prefix+'_bank_sha256'], rows=len(m['paths']))
    eligible.initialize(Path(spec['positive_bank']),Path(spec['random_positive_bank']))
    balanced.initialize(spec,saved)
    draws=[]
    for step in range(saved['step'],saved['step']+64):
        row,_=sampler.select(step)
        draws.append([step,sampler.LAST['source'],row['path_id']])
    return dict(status='passed_read_only_exact_semantics', A_sha256=supervisor.sha(baseline),
        saved_step=entry['step'],saved_sha256=entry['sha256'],optimizer_states=len(saved['optimizer']['state']),
        balance_origin_step=spec['balance_origin_step'], balance_ema_f=saved['metrics']['balance_ema_f'],
        source_counts=saved['metrics']['connected_consumption'], banks=banks,
        sampler_next64=draws,gate=gate())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--compare',type=Path)
    args=parser.parse_args();result=snapshot()
    if args.compare:
        expected=json.loads(args.compare.read_text())
        assert result==expected,'A/restore/gates/data/sampling semantics changed'
    if args.output:
        with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_counts','sampler_next64']},sort_keys=True))

if __name__=='__main__':main()
