"""Frozen paired root-cluster analysis; no checkpoint or task selection."""
from pathlib import Path
import numpy as np
from build_mcts_training_supervision import atomic
from run_continuous_train_mcts_v1 import read

ROOT=Path('artifacts/paired-learning-48h100k-v1')
def bootstrap(delta):
    d=np.asarray(delta,dtype=float);assert d.shape==(32,)
    rng=np.random.default_rng(20291019448)
    boot=d[rng.integers(0,32,size=(10000,32))].mean(axis=1)
    return dict(delta=float(d.mean()),CI95=list(map(float,np.quantile(boot,[.025,.975]))))

def main():
    from paired_learning_comparison_v1 import gate
    assert gate()['ready'] and (ROOT/'candidate.json').exists()
    out=ROOT/'evaluation';assert (out/'COMPLETED.json').exists() and not (out/'analysis.json').exists()
    result={}
    for weight in [10,1000]:
        deltas=[];wins=ties=losses=0;joint=[];resources=[]
        for root in range(32):
            diffs=[]
            for seed in [20291019,20291020]:
                arms=[read(out/f'{root:03d}-{weight}-{seed}-{name}.json')['result'] for name in ['baseline','candidate']]
                a,b=arms;diffs.append(int(b['solved'])-int(a['solved']))
                ka=(not a['solved'],a['CC'] if a['solved'] else 0,a['moves'] if a['solved'] else 0)
                kb=(not b['solved'],b['CC'] if b['solved'] else 0,b['moves'] if b['solved'] else 0)
                wins+=kb<ka;ties+=kb==ka;losses+=kb>ka
                if a['solved'] and b['solved']:joint.append({k:b[k]-a[k] for k in ['CC','moves','L10','L1000']})
                resources.extend(dict(arm=name,cpu=r['cpu_seconds'],wall=r['wall_seconds'],simulations=r['actual_simulations'],episodes=len(r['episodes']),neural=r['neural_evaluations'],stop=r['stop_reason']) for name,r in zip(['baseline','candidate'],arms))
            deltas.append(sum(diffs)/2)
        result[str(weight)]=dict(success_difference=bootstrap(deltas),paired_lexicographic=dict(wins=wins,ties=ties,losses=losses),
            success_difference_by_length_band=[float(np.mean(deltas[i*8:(i+1)*8])) for i in range(4)],
            joint_success_pairs=len(joint),conditional_mean_differences={k:float(np.mean([r[k] for r in joint])) if joint else None for k in ['CC','moves','L10','L1000']},resources=resources)
    atomic(out/'analysis.json',dict(status='completed_paired_analysis',primary=result['1000']['success_difference'],
        by_objective=result,conclusion='heldout_panel_success_improvement' if result['1000']['success_difference']['CI95'][0]>0 else 'inconclusive_null_or_worse',
        limitation='One synthetic-family heldout panel; no production promotion or superiority to nonneural control.'))
    print(read(out/'analysis.json')['conclusion'])

if __name__=='__main__':main()
