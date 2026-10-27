import hashlib
import io
import json
import shutil
from pathlib import Path
import pytest
from compact_storage_v31 import Store, install, uninstall, digest

@pytest.fixture
def store(tmp_path):
    root=tmp_path; (root/'artifacts/bank').mkdir(parents=True)
    s=Store(root, root/'artifacts/store.sqlite3', writable=True); s.create()
    yield s
    uninstall()

def test_dedup_exact_roundtrip(store):
    for name in ['a.json','b.json']:
        store.put(store.root/'artifacts/bank'/name,b'{"x":1}\n','bank_row')
    assert store.connect().execute('SELECT count(*) FROM objects').fetchone()[0]==1
    assert store.connect().execute('SELECT count(*) FROM paths').fetchone()[0]==2
    assert store.read(store.root/'artifacts/bank/a.json')==b'{"x":1}\n'

def test_immutable_overwrite_rejected(store):
    p=store.root/'artifacts/bank/a.json';store.put(p,b'a','bank_row')
    with pytest.raises(ValueError):store.put(p,b'b','bank_row')
    assert store.read(p)==b'a'

def test_outside_root_rejected(store):
    with pytest.raises(ValueError):store.put(store.root/'../outside',b'a','bank_row')

def test_corruption_fails_closed(store):
    p=store.root/'artifacts/bank/a.json';store.put(p,b'a','bank_row')
    store.connect().execute('UPDATE objects SET data=?',(b'bad',))
    with pytest.raises(Exception):store.read(p)

def test_legacy_path_open_json_seek_exists_glob(store):
    p=store.root/'artifacts/bank/a.json';store.put(p,b'{"x":1}\n','bank_row')
    install(store)
    assert p.exists() and p.is_file() and p.read_bytes()==b'{"x":1}\n'
    assert json.loads(p.read_text())=={'x':1}
    with open(p,'rb') as f:
        assert f.seek(2)==2 and f.read(1)==b'x'
    assert p in list(p.parent.iterdir())
    assert p in list(p.parent.glob('*.json'))
    assert p in list((store.root/'artifacts').rglob('*.json'))
    with pytest.raises(PermissionError):open(p,'wb')

def test_physical_file_never_hidden_by_archive(store):
    p=store.root/'artifacts/bank/a.json';store.put(p,b'old','bank_row');p.write_bytes(b'changed')
    install(store)
    assert p.read_bytes()==b'changed'

def test_future_copied_and_atomic_rows_go_to_cas(store):
    source=store.root/'artifacts/source.json';source.write_bytes(b'{"x":1}\n')
    a=store.root/'artifacts/bank/a.json';b=store.root/'artifacts/bank/b.json'
    install(store,capture_banks=True)
    shutil.copyfile(source,a)
    part=b.with_suffix('.partial');part.write_bytes(b'{"x":1}\n');part.replace(b)
    assert a.read_bytes()==b.read_bytes()==source.read_bytes()
    assert not __import__('os').path.exists(a) and not __import__('os').path.exists(b)
    assert store.connect().execute('SELECT count(*) FROM objects').fetchone()[0]==1
    with pytest.raises(ValueError):
        part.write_bytes(b'bad');part.replace(a)

def test_torch_full_checkpoint_roundtrip(store):
    import torch
    p=store.root/'artifacts/step-123.pt';buffer=io.BytesIO()
    torch.save({'model':torch.arange(8),'optimizer':{'state':{i:{'step':torch.tensor(9)} for i in range(71)}},
                'rng':('state',),'torch_rng':torch.get_rng_state()},buffer)
    store.put(p,buffer.getvalue(),'checkpoint')
    install(store)
    restored=torch.load(p,map_location='cpu',weights_only=False)
    assert torch.equal(restored['model'],torch.arange(8)) and len(restored['optimizer']['state'])==71

def test_read_only_adapter_cannot_capture_writes(store):
    p=store.root/'artifacts/bank/a.json';store.put(p,b'a','bank_row')
    readonly=Store(store.root,store.database,writable=False)
    with pytest.raises(PermissionError):readonly.put(p,b'a','bank_row')

def test_opt_in_is_inherited_by_subprocess(tmp_path):
    import os,subprocess,sys
    root=tmp_path;database=root/'artifacts/compact-storage-v31/store.sqlite3'
    s=Store(root,database,writable=True);s.create()
    s.put(root/'artifacts/bank/path-1.json',b'{"x":1}\n','bank_row')
    repo=Path(__file__).resolve().parent.parent
    env=dict(os.environ,RF_KNOTS_COMPACT_V31='1',RF_KNOTS_COMPACT_ROOT=str(root),
             RF_KNOTS_COMPACT_WRITE='0',RF_KNOTS_COMPACT_VALIDATE_ARCHIVE='1',
             PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=str(repo/'scripts/compact_runtime_v31')+os.pathsep+str(repo/'scripts'))
    code='from pathlib import Path;import json;assert json.loads(Path("artifacts/bank/path-1.json").read_text())=={"x":1};print("inherited")'
    result=subprocess.run([sys.executable,'-c',code],env=env,cwd=root,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'inherited' in result.stdout

def test_missing_archive_runtime_fails_closed(tmp_path):
    import os,subprocess,sys
    repo=Path(__file__).resolve().parent.parent
    env=dict(os.environ,RF_KNOTS_COMPACT_V31='1',RF_KNOTS_COMPACT_ROOT=str(tmp_path),
             PYTHONPATH=str(repo/'scripts/compact_runtime_v31')+os.pathsep+str(repo/'scripts'))
    result=subprocess.run([sys.executable,'-c','print("MUST_NOT_RUN")'],env=env,text=True,capture_output=True)
    assert result.returncode==78 and 'MUST_NOT_RUN' not in result.stdout
