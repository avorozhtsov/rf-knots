"""Export ONE explicitly named archived payload, SHA checked; never mass rehydrate."""
import argparse
import os
from pathlib import Path
from compact_storage_v31 import Store,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--path',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent.parent
    store=Store(root,root/'artifacts/compact-storage-v31/store.sqlite3')
    path=root/args.path
    key=store.key(path)
    if key is None or not key.startswith('artifacts/') or path.exists():
        raise ValueError('explicit missing archived artifact only; no overwrite')
    entry=store.entry(path)
    if entry is None:raise ValueError('not in archive')
    data=store.read(path);assert digest(data)==entry[0]
    temporary=path.with_name(path.name+'.restore-v31')
    with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path);os.utime(path,ns=(entry[3],entry[3]))
    assert digest(path.read_bytes())==entry[0]
    print(dict(status='restored_one_payload',path=key,sha256=entry[0],bytes=len(data)))

if __name__=='__main__':main()
