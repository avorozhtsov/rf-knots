"""One-shot reviewed plan -> compressed verified CAS -> independent checks -> exact unlink.

No old code/spec/report/manifest is rewritten; unique bytes remain recoverable.
Recent MCTS stays raw. Old full checkpoints leave loose storage according to48h/weekly.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from compact_storage_v31 import Store, digest

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'artifacts/compact-storage-v31'
HELD_LOCKS=[]

def atomic(path,data):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,indent=2,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def no_workers():
    ps=subprocess.check_output(['ps','-axo','pid,command'],text=True)
    for line in ps.splitlines():
        if any('/'+name in line or 'scripts/'+name in line for name in [
                'connected_learning_supervisor_v30.py','train_connected_learning_live_v30.py',
                'run_continuous_train_mcts_v2.py','collect_connected_learning_v30.py']):
            raise ValueError('active worker: '+line)
    if not HELD_LOCKS:
        for relative in ['connected-learning-live-v30/writer.lock',
                         'continuous-train-mcts-v2/writer.lock','random-TRAIN-stream-v30/writer.lock']:
            stream=(ROOT/'artifacts'/relative).open('a')
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            HELD_LOCKS.append(stream)

def targets(existing=None):
    result={};now=time.time()
    for manifest in (ROOT/'artifacts').rglob('bank/manifest.json'):
        data=json.loads(manifest.read_text())
        for entry in data.get('paths',[]):
            path=manifest.parent/entry['file']
            if existing is not None and not path.exists():
                entry_in_store=existing.entry(path)
                if entry_in_store and entry_in_store[0]==entry['sha256']:
                    continue
            if path.parent!=manifest.parent or path.is_symlink() or not path.is_file():
                raise ValueError('unsafe/missing bank row '+str(path))
            result[path]=('bank_row',entry['sha256'])
    old=[]
    for path in (ROOT/'artifacts').rglob('step-*.pt'):
        if path.is_file() and not path.is_symlink():old.append(path)
    keep=set();weekly={}
    for path in old:
        stamp=path.stat().st_mtime
        if now-stamp<=172800:keep.add(path);continue
        iso=datetime.datetime.fromtimestamp(stamp,datetime.timezone.utc).isocalendar()
        key=(str(path.parent),iso.year,iso.week)
        if key not in weekly or stamp>weekly[key].stat().st_mtime:weekly[key]=path
    keep.update(weekly.values())
    for path in set(old)-keep:result[path]=('checkpoint',None)
    for path in (ROOT/'artifacts/continuous-train-mcts-v2').glob('attempt-*.json'):
        if now-path.stat().st_mtime>172800:result[path]=('MAIN_search',None)
    for root in (ROOT/'artifacts').glob('random-TRAIN-stream-v*'):
        for path in root.glob('batch-*/search-*.json'):
            if now-path.stat().st_mtime>172800:result[path]=('random_search',None)
    return result,dict(policy='last48h all; older latest in each UTC ISO week per original run',
                       raw_checkpoint_keep=len(keep),archive_checkpoint_count=len(set(old)-keep))

def run_verify(output=None,compare=None,archive=False):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable]
    if archive:
        command+=['scripts/run_compact_v31.py']
        env['RF_KNOTS_COMPACT_VALIDATE_ARCHIVE']='1'
    command+=['scripts/verify_compact_v31.py']
    if output:command+=['--output',str(output)]
    if compare:command+=['--compare',str(compare)]
    subprocess.run(command,env=env,cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--delete-verified-copies',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);no_workers()
    assert shutil.disk_usage(ROOT).free>=16*2**30,'unchanged16GiB maintenance headroom'
    if args.prepare:
        if OUT.exists():raise ValueError('migration output exists; no blind repeat')
        OUT.mkdir();store=Store(ROOT,OUT/'store.sqlite3',writable=True);store.create()
        report=dict(status='packing_no_deletion',created=time.time(),retention=None,packed=0)
        atomic(OUT/'migration.json',report)
        run_verify(output=OUT/'semantics-before.json')
        candidates,retention=targets();report['retention']=retention
        store.connect().execute('BEGIN IMMEDIATE')
        for index,(path,(kind,expected)) in enumerate(sorted(candidates.items()),1):
            info=path.lstat()
            if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_nlink!=1:
                raise ValueError('unique regular source required '+str(path))
            data=path.read_bytes();actual=digest(data)
            if expected and actual!=expected:raise ValueError('source manifest SHA differs')
            store.put(path,data,kind,dict(mode=info.st_mode,mtime_ns=info.st_mtime_ns,
                                        inode=info.st_ino,device=info.st_dev))
            if index%500==0:
                store.connect().execute('COMMIT');store.connect().execute('BEGIN IMMEDIATE')
            if index%5000==0:print(json.dumps(dict(packed=index,total=len(candidates))),flush=True)
        store.connect().execute('COMMIT')
        # Independent decompression/hash pass over EVERY unique object.
        for sha,size in store.connect().execute('SELECT sha,size FROM objects'):
            assert len(store.object(sha))==size
        assert store.connect().execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        store.connect().execute('PRAGMA wal_checkpoint(TRUNCATE)')
        run_verify(compare=OUT/'semantics-before.json',archive=True)
        report.update(status='verified_ready_for_deletion',packed=len(candidates),
                      objects=store.connect().execute('SELECT count(*) FROM objects').fetchone()[0],
                      packed_logical_bytes=store.connect().execute('SELECT sum(size) FROM paths').fetchone()[0])
        atomic(OUT/'migration.json',report)
        print(json.dumps(report),flush=True)
    elif args.delete_verified_copies:
        report=json.loads((OUT/'migration.json').read_text())
        assert report['status']=='verified_ready_for_deletion','not verified, or already executed'
        store=Store(ROOT,OUT/'store.sqlite3',writable=True)
        run_verify(compare=OUT/'semantics-before.json',archive=True)
        # Compare inode, time, size AND original SHA before deleting any copy.
        entries=list(store.connect().execute('SELECT path,sha,size,inode,device,mtime_ns FROM paths'))
        for relative,sha,size,inode,device,mtime in entries:
            path=ROOT/relative;info=path.lstat()
            assert stat.S_ISREG(info.st_mode) and info.st_nlink==1
            assert (info.st_size,info.st_ino,info.st_dev,info.st_mtime_ns)==(size,inode,device,mtime)
            assert digest(path.read_bytes())==sha and digest(store.object(sha))==sha
        no_workers();report.update(status='deleting_verified_copies',removed=0);atomic(OUT/'migration.json',report)
        store.connect().execute('BEGIN IMMEDIATE')
        for index,(relative,sha,*_) in enumerate(entries,1):
            (ROOT/relative).unlink()
            store.connect().execute('UPDATE paths SET removed=1 WHERE path=?',(relative,))
            if index%500==0:
                store.connect().execute('COMMIT');store.connect().execute('BEGIN IMMEDIATE')
            if index%5000==0:print(json.dumps(dict(removed=index,total=len(entries))),flush=True)
        store.connect().execute('COMMIT')
        run_verify(compare=OUT/'semantics-before.json',archive=True)
        store.connect().execute('PRAGMA wal_checkpoint(TRUNCATE)')
        report.update(status='completed',removed=len(entries),completed=time.time(),
                      database_bytes=(OUT/'store.sqlite3').stat().st_size)
        atomic(OUT/'migration.json',report);print(json.dumps(report),flush=True)
    else:parser.error('choose one explicit phase')

if __name__=='__main__':main()
