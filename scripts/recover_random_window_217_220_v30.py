"""Explicit user-authorized storage-boundary recovery; never rerun217..219."""
from pathlib import Path
import fcntl
import shutil
import learning_pipeline_v30 as p
import collect_connected_learning_v30 as collector
import eligible_positive_bank_v28a as eligible

ROOT=Path('artifacts/random-TRAIN-stream-v30')
WINDOW=ROOT/'window-000217-000220'
RECEIPT=ROOT/'storage-recovery-217-220.json'
SEEDS=(20261023461,20261023462)

def replay_batch(batch,groups,states,inputs):
    from audit_cost_mcts_continuations import audit_episode
    d=ROOT/f'batch-{batch:06d}';s=p.read(d/'supply.json');a=p.read(d/'actor.json')
    assert s['status']=='completed' and len(s['rows'])==4
    p.pins_ok(p.read(d/'contract.json')['inputs'])
    assert p.sha(Path(a['file']))==a['sha256']
    assert p.sha(Path(a['training_spec']))==a['training_spec_sha256']
    items=[];cpu=0.;replayed=0
    for ri,row in enumerate(s['rows']):
        proof=p.UnknotWitness.from_dict(row['witness'])
        g,ok=p.admit(proof,p.instance(row['instances'][0]),groups,states)
        assert ok and g==tuple(row['query_group'])
        for wi,weight in enumerate((10,1000)):
            inst=p.instance(row['instances'][wi])
            for seed in SEEDS:
                sp=d/f'search-{ri}-{weight}-{seed}.json';saved=p.read(sp)
                assert (saved['root'],saved['weight'],saved['seed'],saved['actor'])==(ri,weight,seed,a)
                inputs[str(sp)]=p.sha(sp);r=saved['result'];cpu+=r['cpu_seconds']
                for ep in r['episodes']:
                    audit_episode(inst,ep,dict(horizon=inst.cap_moves,insertion_candidates=4));replayed+=1
                if r['witness']:
                    w=p.UnknotWitness.from_dict(r['witness']);g,ok=p.admit(w,inst,groups,states)
                    if ok:
                        obs=dict(source_kind='v28',query_group=list(g),shard=str(sp),shard_sha256=p.sha(sp),
                            actor=a,roster=str(d/'supply.json'),roster_sha256=p.sha(d/'supply.json'),
                            root_index=ri,seed=seed,weight=weight)
                        items.append(p.positive_row(w,row['instances'][wi],obs,random_source=True))
    views={(tuple(r['instance']['word']),r['instance']['strands']) for r in items}
    report=dict(conditions=16,search_cpu_seconds=cpu,solved_normalized_views=len(views),
        reference_labels=0,independently_replayed_episodes=replayed)
    if batch<220:
        assert p.read(d/'collection.json')==report,'existing batch must reproduce exactly, not be replaced'
    else:
        assert not (d/'collection.json').exists();p.atomic(d/'collection.json',report)
    for f in [d/'contract.json',d/'supply.json',d/'actor.json',d/'collection.json']:
        inputs[str(f)]=p.sha(f)
    return items,views,report

def main():
    from run_continuous_train_mcts_v2 import snapshot
    from run_continuous_train_mcts_v1 import policy
    from verified_reduction_search_v28 import search
    assert shutil.disk_usage('.').free>=16*1024**3
    import connected_learning_supervisor_v30 as supervisor
    assert not supervisor.has_process('scripts/connected_learning_supervisor_v30.py')
    assert not supervisor.has_process('scripts/train_connected_learning_live_v30.py')
    state=p.read(supervisor.ROOT/'state.json')
    assert state['status']=='STOPPED_LEARNER' and state['exit_code']==0 and state['stop_reason']=='storage_gate'
    supervisor.validate_parent() # fullSHA/allpins/finite71AdamW/RNG
    with (ROOT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not RECEIPT.exists(),'one reviewed recovery, never retry partially completed work'
        m=p.read(ROOT/'manifest.json')
        assert m['status']=='collecting' and m['next_batch']==220
        assert [r['batch'] for r in m['window_batches']]==[217,218,219]
        assert not (ROOT/'batch-000220').exists() and not (WINDOW/'bank').exists()
        contract=p.read(WINDOW/'contract.json')
        assert contract==dict(first_batch=217,batches=4,minimum_solved_views=8,
            maximum_CPU_per_solved_view=300,all_batches_count=True,
            parent_manifest_sha256=p.sha(Path(m['current_bank'])/'manifest.json'))
        frozen={str(f):p.sha(f) for bi in (217,218,219) for f in (ROOT/f'batch-{bi:06d}').glob('*.json')}
        frozen[str(WINDOW/'contract.json')]=p.sha(WINDOW/'contract.json')
        p.atomic(RECEIPT,dict(status='reviewed_recovery_started',authorization='USER2026_10_26',
            original_manifest=m,existing_files=frozen,only_new_search_batch=220,
            script_sha256=p.sha(Path(__file__)),no_deletion=True,no_A_or_clock_reset=True))
        groups,states,_,_=p.protected_registry();inputs={};items=[];views=set();reports=[]
        for bi in (217,218,219):
            it,vs,r=replay_batch(bi,groups,states,inputs);items+=it;views.update(vs)
            reports.append(dict(batch=bi,solved_views=r['solved_normalized_views'],
                cpu_seconds=r['search_cpu_seconds'],replayed_episodes=r['independently_replayed_episodes']))
        assert reports==m['window_batches']
        parent=collector.parent;parent.pipeline=p;parent.FIRST_BATCH=61
        parent.prior_rosters=collector.rosters;parent.code_pins=collector.pins
        d=ROOT/'batch-000220';s=parent.generate(d,220,sorted(ROOT.glob('batch-*/supply.json')))
        assert s['status']=='completed','supply failure, preserve and review; no retry'
        for row in s['rows']:
            w=p.UnknotWitness.from_dict(row['witness']);g,ok=p.admit(w,p.instance(row['instances'][0]),groups,states)
            assert ok and g==tuple(row['query_group'])
        actor=snapshot();p.atomic(d/'actor.json',actor);net,obj=policy(actor)
        for ri,row in enumerate(s['rows']):
            for wi,weight in enumerate((10,1000)):
                for seed in SEEDS:
                    p.disk_guard();sp=d/f'search-{ri}-{weight}-{seed}.json';assert not sp.exists()
                    result=search(p.instance(row['instances'][wi]),net,obj,seed,cpu_seconds=20,simulations=16)
                    p.atomic(sp,dict(root=ri,weight=weight,seed=seed,actor=actor,result=result))
        it,vs,r=replay_batch(220,groups,states,inputs);items+=it;views.update(vs)
        reports.append(dict(batch=220,solved_views=r['solved_normalized_views'],
            cpu_seconds=r['search_cpu_seconds'],replayed_episodes=r['independently_replayed_episodes']))
        cpu=sum(r['cpu_seconds'] for r in reports)
        report=dict(batches=reports,solved_normalized_views=len(views),search_cpu_seconds=cpu,
            passed=p.window_gate(len(views),cpu),reference_labels=0,conditions=64,all_batches_counted=True)
        assert not (WINDOW/'yield-audit.json').exists();p.atomic(WINDOW/'yield-audit.json',report)
        assert report['passed'],'original window yield failed; no retry/extension'
        p.pins_ok(frozen)
        inputs.update(frozen)
        for f in [Path(__file__),RECEIPT,WINDOW/'yield-audit.json',Path('scripts/verified_reduction_search_v28.py')]:
            inputs[str(f)]=p.sha(f)
        parent.write_bank(WINDOW/'bank',Path(m['current_bank']),Path(m['current_audit']),items,inputs,
            search_cpu_seconds=cpu,source_conditions=64)
        eligible.audit(WINDOW/'bank',WINDOW/'bank-audit');p.pins_ok(frozen)
        m.update(status='ready_for_next_batch',next_batch=221,current_bank=str(WINDOW/'bank'),
            current_audit=str(WINDOW/'bank-audit/report.json'),window_batches=reports)
        m['completed'].append(dict(first_batch=217,last_batch=220,bank_sha256=p.sha(WINDOW/'bank/manifest.json')))
        p.atomic(ROOT/'manifest.json',m)
        p.atomic(ROOT/'READY.json',dict(bank=m['current_bank'],audit=m['current_audit'],
            manifest_sha256=p.sha(WINDOW/'bank/manifest.json'),audit_sha256=p.sha(Path(m['current_audit'])),
            optimizer_connected=False))
        p.atomic(WINDOW/'storage-recovery-completed.json',dict(status='passed',old_files_byte_identical=True,
            original_batches_not_researched=[217,218,219],new_search_conditions=16,recovery_plan_sha256=p.sha(RECEIPT),
            yield_report=report,READY_sha256=p.sha(ROOT/'READY.json')))
        print(dict(status='RECOVERED_FIXED_WINDOW',report=report))

if __name__=='__main__':main()
