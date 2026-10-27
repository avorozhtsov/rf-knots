"""Tool-managed generation -> audit -> immutable intake -> actual AdamW loop.

Single supervisor; child commands and logs remain inspectable. No service/nohup,
checkpoint resets, live bank mutation, outcome-selected checkpoints or retries.
"""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import torch
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from learning_pipeline_v28 import disk_guard,pins_ok
import positive_sampler_v28a as eligible
import positive_sampler_v28 as sampler

ROOT=Path('artifacts/connected-learning-live-v30')
REF=Path('artifacts/continuous-r2-current-training.json')
TRAINER='scripts/train_connected_learning_live_v30.py'
RANDOM=Path('artifacts/random-TRAIN-stream-v30')
INITIAL_MAIN=Path('artifacts/connected-learning-live-v30/initial-main-READY.json')
INITIAL_RANDOM=Path('artifacts/random-TRAIN-stream-v30/READY.json')


def health_status(verified,supply_stopped,intake_stopped):
    if supply_stopped or intake_stopped:return 'DEGRADED_DATAFLOW'
    return 'VERIFIED_ACTIVE' if verified else 'LAUNCHED_AWAITING_PROGRESS'


def refresh_due(seconds,new_views,new_paths):
    return seconds>=3600 and (new_views>=4 or new_paths>=16)


def first_new_path(step,main,random,new_ids):
    eligible.initialize(main,random)
    for n in range(1,65):
        row,_=sampler.select(step+n-1)
        if row['path_id'] in new_ids[sampler.LAST['source']]:return n,row['path_id']
    raise ValueError('no NEW path within64 ordinary updates; no phase reset or forced sampling')


def finite(v):
    if torch.is_tensor(v):assert torch.isfinite(v).all()
    elif isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,(list,tuple)):
        for x in v:finite(x)


def checked_ready(path):
    r=read(path);bank,audit=Path(r['bank']),Path(r['audit'])
    assert sha(bank/'manifest.json')==r['manifest_sha256'] and sha(audit)==r['audit_sha256']
    ar=read(audit);assert ar['status']=='passed_data_only' and ar['manifest_sha256']==r['manifest_sha256']
    pins_ok(read(bank/'manifest.json')['inputs'])
    return r


def ids(bank):
    m=read(Path(bank)/'manifest.json');excluded=set(m.get('quarantined_path_ids',[]))
    return {read(Path(bank)/e['file'])['path_id'] for e in m['paths']}-excluded


def fresh_main_ids(bank):
    return set(read(Path(bank)/'manifest.json')['fresh_main_path_ids'])


def first_new_random(step,main,random,new_ids):
    eligible.initialize(main,random)
    for n in range(1,65):
        row,_=sampler.select(step+n-1)
        if sampler.LAST['source']=='random' and row['path_id'] in new_ids:return n,row['path_id']
    raise ValueError('no NEW random draw within64 ordinary updates; no phase reset or forced sampling')


def validate_parent():
    ref=read(REF);s=read(Path(ref['spec']));parent=Path(ref['out']);m=read(parent/'manifest.json')
    assert sha(Path(ref['spec']))==ref['spec_sha256']
    assert m['status']=='checkpointed' and m['stop_reason'] in ['storage_gate','user_stop_request','smoke_complete']
    pins_ok(m['contract']['inputs'])
    e=m['checkpoints'][-1];cp=parent/e['file'];assert sha(cp)==e['sha256']
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    assert saved['step']==e['step'] and saved['contract_sha256']==m['contract_sha256']
    assert len(saved['optimizer']['state'])==71 and all(k in saved for k in ['rng','torch_rng','mps_rng'])
    finite(saved)
    return s,parent,m,e,saved


def configure(main_ready,random_ready):
    from audit_fresh_main_eligibility_v29 import validate_fresh
    old,parent,m,e,_=validate_parent();s=dict(old)
    for prefix,path in [('positive',main_ready),('random_positive',random_ready)]:
        r=checked_ready(path);bank=Path(r['bank']);before=Path(old[prefix+'_bank'])
        oldrows=read(before/'manifest.json')['paths'];assert read(bank/'manifest.json')['paths'][:len(oldrows)]==oldrows
        for row in oldrows:assert (before/row['file']).read_bytes()==(bank/row['file']).read_bytes()
        s.update({prefix+'_bank':str(bank),prefix+'_bank_sha256':r['manifest_sha256'],
            prefix+'_audit':r['audit'],prefix+'_audit_sha256':r['audit_sha256']})
    baseline=read(Path('artifacts/paired-learning-48h100k-v3/baseline.json'))
    assert baseline['step']==1628687 and baseline['first_counted_task']==39168
    assert sha(Path(baseline['checkpoint']))==baseline['checkpoint_sha256']
    bs=read(Path(baseline['spec']));assert sha(Path(baseline['spec']))==baseline['spec_sha256']
    s['fresh_retained_ids']=sorted(validate_fresh(Path(s['positive_bank'])))
    s['fresh_random_ids']=sorted(ids(s['random_positive_bank'])-ids(bs['random_positive_bank']))
    closed=read(Path('artifacts/random-TRAIN-stream-v29/READY.json'))
    s['v30_introduced_random_ids']=sorted(ids(s['random_positive_bank'])-ids(closed['bank']))
    assert s['v30_introduced_random_ids'],'repair must supply genuinely new random labels'
    assert s['fresh_retained_ids'] and s['fresh_random_ids']
    assert old['balance_origin_step']==1645639,'original gradient phase must survive migration'
    s.update(balance_origin_step=old['balance_origin_step'],
        new_retained_ids=s['fresh_retained_ids'],new_random_ids=s['fresh_random_ids'],
        parent=str(parent/e['file']),parent_step=e['step'],parent_sha256=e['sha256'],
        parent_manifest_sha256=sha(parent/'manifest.json'),version='connected-v30')
    return s


def prepare(round_number,main_ready,random_ready,cpu_report):
    import balanced_new_mcts_f_v29 as balanced
    disk_guard();s=configure(main_ready,random_ready);cpu=read(cpu_report)
    assert cpu['status']=='passed_balanced_CPU_v29' and cpu['updates']==6 and cpu['AdamW_states']==71
    pins_ok(cpu['inputs'])
    assert cpu['source_sha256']==s['parent_sha256']
    assert cpu['positive_bank_sha256']==s['positive_bank_sha256'] and cpu['random_positive_bank_sha256']==s['random_positive_bank_sha256']
    balanced.initialize(s)
    _,_=sampler.select(s['parent_step']+1)
    spec=Path(f'artifacts/continuous-r2-start-spec-connected-v30-{round_number:03d}.json')
    out=Path(f'artifacts/continuous-r2-connected-v30-{round_number:03d}')
    assert not spec.exists() and not out.exists(),'immutable round namespace exists'
    s.update(out=str(out),commissioning_updates=2,commissioning_path_id=sampler.LAST['path_id'],
        cpu_report=str(cpu_report),cpu_report_sha256=sha(cpu_report))
    atomic(spec,s);return spec


def spawn(command,log):
    disk_guard();log.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONPATH':'src:scripts:/Users/artemvorozhtsov/projects/pgx-mcts-bench/src'}
    with log.open('a') as f:
        p=subprocess.Popen([sys.executable,*map(str,command)],stdout=f,stderr=subprocess.STDOUT,env=env)
    return p


def commission(spec):
    s=read(spec);out=Path(s['out']);parent=torch.load(s['parent'],map_location='cpu',weights_only=False)
    p=spawn([TRAINER,'--spec',spec,'--stop-after',s['commissioning_updates']],ROOT/f'{out.name}-commission.log')
    code=p.wait();assert code==0,('native commissioning failed',code)
    m=read(out/'manifest.json');c=m['checkpoints'][-1]
    assert m['status']=='checkpointed' and m['stop_reason']=='smoke_complete'
    assert c['step']==s['parent_step']+s['commissioning_updates'] and sha(out/c['file'])==c['sha256']
    saved=torch.load(out/c['file'],map_location='cpu',weights_only=False);finite(saved)
    assert len(saved['optimizer']['state'])==71 and all(k in saved for k in ['rng','torch_rng','mps_rng'])
    for k,v in parent['optimizer']['state'].items():
        assert 0<=float(saved['optimizer']['state'][k]['step'])-float(v['step'])<=s['commissioning_updates']
    metrics=saved['metrics'];consumed=metrics['connected_consumption']
    assert metrics['positive_admitted_path_id']==s['commissioning_path_id']
    assert metrics['positive_admitted_path_id'] in s['new_'+metrics['positive_sampling_source']+'_ids']
    assert consumed['new_random']+consumed['new_retained']>=1
    assert consumed['retained']+consumed['random']==s['commissioning_updates']
    assert abs(consumed['retained']-consumed['random'])<=1
    from balanced_new_mcts_f_v29 import numerical_gate
    numerical_gate(metrics)
    assert consumed['retained']==consumed['new_retained']==1 and consumed['random']==consumed['new_random']==1
    atomic(out/'commissioning-audit.json',dict(status='passed_native_MPS_balanced_fresh_MCTS_F',
        startup_bitwise_restore='enforced by unchanged core before any update',
        checkpoint=c,updates=s['commissioning_updates'],AdamW_states=71,consumption=consumed,
        parent_sha256=s['parent_sha256'],spec_sha256=sha(spec),metrics=metrics))
    atomic(REF,dict(out=str(out),spec=str(spec),spec_sha256=sha(spec),registered_step=c['step'],
        bank_manifest_sha256=s['bank_sha256']))
    return spawn([TRAINER,'--spec',spec,'--resume'],out/'runtime.log'),c['step']


def cpu_check(main_ready,random_ready,out):
    a,b=checked_ready(main_ready),checked_ready(random_ready)
    command=['scripts/smoke_balanced_new_mcts_f_v30.py','--main-ready',main_ready,'--random-ready',random_ready,'--out',out]
    code=spawn(command,ROOT/(out.name+'.log')).wait();assert code==0,('CPU intake failed',code)
    return out/'report.json'


def has_process(fragment):
    listing=subprocess.check_output(['ps','-axo','command='],text=True)
    return any(fragment in line and not line.startswith(('rg ','grep ')) for line in listing.splitlines())


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-initial',action='store_true')
    p.add_argument('--cpu-report',type=Path);p.add_argument('--start',action='store_true')
    p.add_argument('--resume',action='store_true',help='reviewed clean storage stop, SAME saved native state; no new commissioning')
    a=p.parse_args()
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.prepare_initial:
            assert not has_process('scripts/train_continuous_random_view_r2_v24.py --spec artifacts/continuous-r2-start-spec-v27.json')
            spec=prepare(0,INITIAL_MAIN,INITIAL_RANDOM,a.cpu_report)
            atomic(ROOT/'prepared.json',dict(spec=str(spec),main_ready=str(INITIAL_MAIN),random_ready=str(INITIAL_RANDOM)))
            print(dict(status='prepared',spec=str(spec),updates=read(spec)['commissioning_updates']));return
        assert shutil.disk_usage('.').free>=16*1024**3,'launch headroom16GiB required, unchanged runtime guard12GiB'
        if a.resume:
            assert not (ROOT/'failure.json').exists(),'input/numerical failure requires review/new version, not retry'
            state=read(ROOT/'state.json');spec=Path(read(REF)['spec'])
            assert not state.get('supply_stopped') and not state.get('main_intake_stopped'),'failed dataflow requires new reviewed version'
            assert str(spec)==state['spec'] and state['status']=='STOPPED_LEARNER'
            assert state.get('exit_code')==0 and state.get('stop_reason') in ['storage_gate','user_stop_request']
            assert not (ROOT/'STOP_REQUEST').exists(),'explicit supervisor stop still in force'
            assert not has_process(TRAINER),'native learner already exists'
            s,parent,m,e,_=validate_parent();round_number=state['round']
            for folder in [parent,Path('artifacts/continuous-train-mcts-v2'),RANDOM]:
                marker=folder/'STOP_REQUEST'
                if marker.exists():
                    reason=read(marker).get('reason')
                    assert reason in ['unchanged12GiB resource guard','learner stopped; finish current frozen batch only',
                        'owned supervisor resource stop; preserve partial batch'],'human stop must not be cleared'
                    marker.rename(folder/f'STOP_REQUEST.handled-{time.time_ns()}') # preserve exact request evidence
            prepared={}
            for label,prefix in [('main','positive'),('random','random_positive')]:
                path=ROOT/f'resume-{e["step"]}-{label}-READY.json'
                r=dict(bank=s[prefix+'_bank'],audit=s[prefix+'_audit'],
                    manifest_sha256=s[prefix+'_bank_sha256'],audit_sha256=s[prefix+'_audit_sha256'],optimizer_connected=True)
                if path.exists():assert read(path)==r
                else:atomic(path,r)
                prepared[label+'_ready']=str(path)
            learner=spawn([TRAINER,'--spec',spec,'--resume'],parent/'runtime.log');commission_step=e['step']
        else:
            assert a.start and not (ROOT/'state.json').exists(),'fresh start only; stopped evidence needs review'
            assert not has_process('scripts/train_connected_learning_live_'),'native connected learner already exists'
            # New supervisor owns only reviewed resource-stop markers, never a
            # human STOP_REQUEST. Preserve the exact old request under a new name.
            marker=Path('artifacts/continuous-train-mcts-v2/STOP_REQUEST')
            if marker.exists():
                assert read(marker).get('reason') in ['owned supervisor resource stop; preserve partial batch',
                    'unchanged12GiB resource guard'],'explicit main stop is not cleared'
                marker.rename(marker.parent/f'STOP_REQUEST.handled-{time.time_ns()}')
            prepared=read(ROOT/'prepared.json');spec=Path(prepared['spec']);round_number=0
            state=dict(status='COMMISSIONING',round=0,spec=str(spec),pid=os.getpid())
            atomic(ROOT/'state.json',state)
            learner,commission_step=commission(spec)
        round_started=time.monotonic()
        pending_since=None
        main_ready=Path(prepared['main_ready']);random_ready=Path(prepared['random_ready'])
        active_random=checked_ready(random_ready);active_main=checked_ready(main_ready)
        if a.resume and state.get('latest_main_ready'):
            main_ready=Path(state['latest_main_ready']);checked_ready(main_ready)
        main_intake=None;collector=None;next_main_ready=None
        main_worker=None;supply_stopped=False;intake_stopped=False
        if a.resume and (RANDOM/'manifest.json').exists():
            supply_stopped=read(RANDOM/'manifest.json')['status']!='ready_for_next_batch'
            if (RANDOM/'READY.json').exists():random_ready=RANDOM/'READY.json'
        stream=Path('artifacts/continuous-train-mcts-v2')
        if not has_process('scripts/run_continuous_train_mcts_v2.py'):
            m=read(stream/'manifest.json');pins_ok(m['contract']['inputs'])
            for e in m['shards']:assert sha(stream/e['file'])==e['sha256']
            for e in m['audits']:assert sha(Path(e['report']))==e['sha256']
            for actor in m['batches'].values():assert sha(Path(actor['file']))==actor['sha256']
            assert not (stream/'STOP_REQUEST').exists()
            main_worker=spawn(['scripts/run_continuous_train_mcts_v2.py','--stream-spec',
                'artifacts/continuous-mcts-stream-start-spec-v2.json','--out',stream,
                '--pgx-root','/Users/artemvorozhtsov/projects/pgx-mcts-bench','--resume'],ROOT/'main-MCTS.log')
        while True:
            now=time.monotonic();s=read(spec);out=Path(s['out'])
            if learner.poll() is not None:
                m=read(out/'manifest.json')
                state.update(status='STOPPED_LEARNER',exit_code=learner.returncode,
                    stop_reason=m.get('stop_reason'),learner_manifest_status=m['status'])
                atomic(ROOT/'state.json',state)
                if collector and collector.poll() is None:atomic(RANDOM/'STOP_REQUEST',dict(reason='learner stopped; finish current frozen batch only'))
                if main_intake and main_intake.poll() is None:main_intake.wait()
                if collector and collector.poll() is None:collector.wait()
                # Main has its own resource guard; do not kill an unaffected healthy search.
                if main_worker and main_worker.poll() is None:
                    atomic(stream/'STOP_REQUEST',dict(reason='owned supervisor resource stop; preserve partial batch'))
                    main_worker.wait()
                return
            progress=read(out/'progress.json') if (out/'progress.json').exists() else {}
            consumed=progress.get('metrics',{}).get('connected_consumption',{}) if progress.get('metrics') else {}
            verified=(progress.get('live_optimizer_steps',0)>commission_step and consumed.get('new_random',0)>0
                and consumed.get('retained',0)>0 and consumed.get('random',0)>0 and time.time()-progress.get('timestamp',0)<180)
            state.update(status=health_status(verified,supply_stopped,intake_stopped),round=round_number,
                spec=str(spec),learner_pid=learner.pid,progress=progress,source_fraction=dict(retained=.5,random=.5),
                disk_free_GiB=shutil.disk_usage('.').free/1024**3,supply_stopped=supply_stopped)
            panel_ready=Path('artifacts/paired-learning-48h100k-v3/study.json').exists()
            if collector and collector.poll() is not None:
                if collector.returncode!=0 or read(RANDOM/'manifest.json')['status']!='ready_for_next_batch':
                    supply_stopped=True
                elif (RANDOM/'READY.json').exists():random_ready=RANDOM/'READY.json'
                collector=None
            if main_intake and main_intake.poll() is not None:
                if main_intake.returncode!=0:intake_stopped=True
                else:main_ready=next_main_ready
                main_intake=None
            pending=0
            if random_ready.exists():
                r=checked_ready(random_ready);ar=read(Path(r['audit']));old_ar=read(Path(active_random['audit']))
                pending=max(0,ar['normalized_views']-old_ar['normalized_views'])
            state.update(pending_new_random_views=pending,latest_random_ready=str(random_ready),
                latest_main_ready=str(main_ready),active_random_bank=active_random['bank'],
                collector_pid=collector.pid if collector else None,intake_pid=main_intake.pid if main_intake else None,
                future_AB_panel_audited=panel_ready,main_intake_stopped=intake_stopped)
            state['status']=health_status(verified,supply_stopped,intake_stopped)
            atomic(ROOT/'state.json',state)
            if (ROOT/'STOP_REQUEST').exists():
                atomic(out/'STOP_REQUEST',dict(reason='explicit supervisor stop'));continue
            if shutil.disk_usage('.').free<12*1024**3:
                atomic(out/'STOP_REQUEST',dict(reason='unchanged12GiB resource guard'));continue
            # Until fresh disjoint panel is frozen, do not generate any new TRAIN roster.
            if panel_ready and not supply_stopped and collector is None and pending<32:
                collector=spawn(['scripts/collect_connected_learning_v30.py','--out',RANDOM,
                    '--parent-ready',INITIAL_RANDOM,'--batches','4'],ROOT/'random-collector.log')
            pending_paths=len(fresh_main_ids(checked_ready(main_ready)['bank'])-fresh_main_ids(active_main['bank']))
            if panel_ready and main_intake is None and not intake_stopped and pending_paths<32:
                r=checked_ready(main_ready);cursor=read(Path(r['bank'])/'manifest.json')['main_next_batch']
                completed={e['batch'] for e in read(stream/'manifest.json')['audits']}
                if all(i in completed for i in range(cursor,cursor+4)):
                    dest=ROOT/f'main-intake-{cursor:06d}-{cursor+3:06d}';next_main_ready=dest/'READY.json'
                    if next_main_ready.exists():
                        checked_ready(next_main_ready);main_ready=next_main_ready
                        continue
                    assert not dest.exists(),'partial intake is not safe to retry'
                    main_intake=spawn(['scripts/intake_fresh_main_v29a.py','--out',dest,
                        '--parent-bank',r['bank'],'--parent-audit',r['audit'],
                        '--first-batch',str(cursor),'--batches','4'],ROOT/'main-intake.log')
            pending_paths+=len(ids(checked_ready(random_ready)['bank'])-ids(active_random['bank']))
            state['pending_new_semantic_paths']=pending_paths
            if pending or pending_paths:
                if pending_since is None:pending_since=time.monotonic()
            else:pending_since=None
            state['pending_READY_age_seconds']=0 if pending_since is None else time.monotonic()-pending_since
            if state['pending_READY_age_seconds']>7200:
                state['status']='DEGRADED_DATAFLOW'
                state['degradation_reason']='READY not consumed for two active continuous hours'
            atomic(ROOT/'state.json',state)
            if verified and refresh_due(now-round_started,pending,pending_paths) and main_intake is None:
                # No forced short checkpoints: only a substantial >=hour intake handoff.
                # Freeze CONCRETE READY values before any CPU check; the collector
                # may publish a later READY while this handoff is in progress.
                next_round=round_number+1
                frozen_main=ROOT/f'round-{next_round:03d}-main-READY.json'
                frozen_random=ROOT/f'round-{next_round:03d}-random-READY.json'
                atomic(frozen_main,checked_ready(main_ready));atomic(frozen_random,checked_ready(random_ready))
                atomic(out/'STOP_REQUEST',dict(reason='authorized hourly full-state audited snapshot intake'))
                assert learner.wait()==0
                round_number+=1
                report=cpu_check(frozen_main,frozen_random,ROOT/f'round-{round_number:03d}-CPU')
                spec=prepare(round_number,frozen_main,frozen_random,report)
                learner,commission_step=commission(spec);round_started=time.monotonic()
                active_random=checked_ready(frozen_random);active_main=checked_ready(frozen_main)
                pending_since=None
                prepared['main_ready']=str(main_ready)
            time.sleep(15)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if ROOT.exists():atomic(ROOT/'failure.json',dict(error=repr(e),time=time.time(),no_automatic_retry=True))
        raise
