"""Only explicit v31 launchers activate archive compatibility in child processes."""
import os
if os.environ.get('RF_KNOTS_COMPACT_V31') == '1':
    try:
        import sys
        import json
        import hashlib
        from pathlib import Path
        root = Path(os.environ['RF_KNOTS_COMPACT_ROOT'])
        sys.path.insert(0, str(root/'scripts'))
        from compact_storage_v31 import Store, install
        database = root/'artifacts/compact-storage-v31/store.sqlite3'
        if not database.is_file():
            raise RuntimeError('v31 database missing')
        pins_file=root/'artifacts/compact-storage-v31/runtime-pins.json'
        if pins_file.is_file():
            for relative,expected in json.loads(pins_file.read_text())['files'].items():
                if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
                    raise RuntimeError('compact runtime pin changed: '+relative)
        elif os.environ.get('RF_KNOTS_COMPACT_VALIDATE_ARCHIVE')!='1':
            raise RuntimeError('compact runtime not published/pinned')
        store = Store(root, database, writable=os.environ.get('RF_KNOTS_COMPACT_WRITE') == '1')
        import shutil
        store._original_copyfile = shutil.copyfile
        prefer = os.environ.get('RF_KNOTS_COMPACT_VALIDATE_ARCHIVE') == '1'
        if prefer and store.writable:
            raise RuntimeError('forced archive reads are validation-only')
        install(store, capture_banks=store.writable, prefer_archive=prefer)
    except BaseException as error:
        import sys
        print('COMPACT_V31_FAIL_CLOSED: '+repr(error), file=sys.stderr, flush=True)
        os._exit(78)
