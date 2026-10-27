"""Bounded retention of immutable artifact payloads; no optimizer or search calls.

Run as plain Python (not from an already patched process). Safe for readers:
commit -> checksum -> unlink. Never removes manifests, audits, A/B or partial work.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path
from compact_storage_v31 import Store,digest
from migrate_compact_storage_v31 import ROOT,OUT,targets,atomic

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    assert json.loads((OUT/'migration.json').read_text())['status']=='completed'
    assert shutil.disk_usage(ROOT).free>=16*2**30,'16GiB maintenance headroom; never override runtime12GiB'
    with (OUT/'maintenance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(ROOT,OUT/'store.sqlite3',writable=True)
        candidates,retention=targets(store)
        report=dict(status='planned',timestamp=time.time(),candidates=len(candidates),retention=retention,
                    bytes=sum(p.stat().st_size for p in candidates))
        if not args.apply:
            print(json.dumps(report));return
        checked=[]
        store.connect().execute('BEGIN IMMEDIATE')
        for path,(kind,expected) in sorted(candidates.items()):
            info=path.lstat()
            assert stat.S_ISREG(info.st_mode) and info.st_nlink==1
            data=path.read_bytes();sha=digest(data)
            if expected:assert sha==expected
            # New search data are eligible only after the fixed48h window.
            if kind!='bank_row':assert time.time()-info.st_mtime>172800
            store.put(path,data,kind,dict(mode=info.st_mode,mtime_ns=info.st_mtime_ns,
                                        inode=info.st_ino,device=info.st_dev))
            checked.append((path,sha,info))
        store.connect().execute('COMMIT')
        for path,sha,info in checked:
            now=path.lstat()
            assert (now.st_ino,now.st_size,now.st_mtime_ns)==(info.st_ino,info.st_size,info.st_mtime_ns)
            assert digest(path.read_bytes())==sha and digest(store.object(sha))==sha
        for path,sha,info in checked:
            path.unlink()
            store.connect().execute('UPDATE paths SET removed=1 WHERE path=?',(store.key(path),))
        report.update(status='completed',removed=len(checked))
        atomic(OUT/'maintenance.json',report);print(json.dumps(report))

if __name__=='__main__':main()
