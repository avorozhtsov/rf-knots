"""Versioned CAS: compressed immutable bytes, indexed legacy paths, opt-in I/O adapter.

No source/spec/model/objective pins are rewritten. No network or optimizer work.
Only paths explicitly present in this database can be virtual read-only files.
"""
import builtins
import functools
import io
import os
import sqlite3
import stat
import threading
import zlib
import hashlib
from pathlib import Path

RAW_OPEN = builtins.open
RAW_PATH_OPEN = Path.open
RAW_EXISTS = Path.exists
RAW_IS_FILE = Path.is_file
RAW_REPLACE = Path.replace
RAW_ITERDIR = Path.iterdir
RAW_GLOB = Path.glob
RAW_RGLOB = Path.rglob
INSTALLED = None


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Store:
    def __init__(self, root, database, writable=False):
        self.root = Path(os.path.abspath(root))
        self.database = Path(os.path.abspath(database))
        self.writable = writable
        self.local = threading.local()

    def connect(self):
        if not hasattr(self.local, 'db'):
            if self.writable:
                db = sqlite3.connect(str(self.database), timeout=60, isolation_level=None)
                db.execute('PRAGMA journal_mode=WAL')
                db.execute('PRAGMA synchronous=FULL')
            else:
                db = sqlite3.connect(self.database.as_uri()+'?mode=ro', uri=True,
                                     timeout=60, isolation_level=None)
            db.execute('PRAGMA foreign_keys=ON')
            self.local.db = db
        return self.local.db

    def create(self):
        if not self.writable:
            raise ValueError('read-only')
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connect().executescript('''
          CREATE TABLE IF NOT EXISTS objects(sha TEXT PRIMARY KEY, size INTEGER NOT NULL,
                                             data BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS paths(path TEXT PRIMARY KEY, sha TEXT NOT NULL REFERENCES objects(sha),
              kind TEXT NOT NULL, size INTEGER NOT NULL, mode INTEGER NOT NULL,
              mtime_ns INTEGER NOT NULL, inode INTEGER NOT NULL, device INTEGER NOT NULL,
              removed INTEGER NOT NULL DEFAULT 0);
          CREATE INDEX IF NOT EXISTS paths_sha ON paths(sha);
          CREATE INDEX IF NOT EXISTS paths_kind ON paths(kind);
        ''')

    def key(self, path):
        if isinstance(path, int):
            return None
        try:
            absolute = Path(os.path.abspath(os.fsdecode(path)))
            return absolute.relative_to(self.root).as_posix()
        except (TypeError, ValueError):
            return None

    def entry(self, path):
        key = self.key(path)
        if key is None or not key.startswith('artifacts/'):
            return None
        return self.connect().execute('SELECT sha,size,mode,mtime_ns,removed FROM paths WHERE path=?',
                                      (key,)).fetchone()

    @functools.lru_cache(maxsize=32)
    def object(self, sha):
        row = self.connect().execute('SELECT size,data FROM objects WHERE sha=?', (sha,)).fetchone()
        if row is None:
            raise OSError('missing CAS object '+sha)
        data = zlib.decompress(row[1])
        if len(data) != row[0] or digest(data) != sha:
            raise OSError('corrupt CAS object '+sha)
        return data

    def read(self, path):
        row = self.entry(path)
        if row is None:
            raise FileNotFoundError(path)
        return self.object(row[0])

    def put(self, path, data, kind, metadata=None):
        if not self.writable:
            raise PermissionError('CAS writer not enabled')
        key = self.key(path)
        if key is None or not key.startswith('artifacts/'):
            raise ValueError('outside artifact namespace')
        sha = digest(data)
        if metadata is None:
            metadata = dict(mode=stat.S_IFREG|0o644, mtime_ns=__import__('time').time_ns(), inode=0, device=0)
        db = self.connect()
        owns_transaction = not db.in_transaction
        if owns_transaction:
            db.execute('BEGIN IMMEDIATE')
        try:
            old = db.execute('SELECT sha FROM paths WHERE path=?', (key,)).fetchone()
            if old and old[0] != sha:
                raise ValueError('immutable archived path overwrite: '+key)
            if db.execute('SELECT 1 FROM objects WHERE sha=?', (sha,)).fetchone() is None:
                db.execute('INSERT INTO objects VALUES(?,?,?)',
                           (sha, len(data), sqlite3.Binary(zlib.compress(data, 6))))
            db.execute('INSERT OR IGNORE INTO paths(path,sha,kind,size,mode,mtime_ns,inode,device) VALUES(?,?,?,?,?,?,?,?)',
                       (key, sha, kind, len(data), metadata['mode'], metadata['mtime_ns'],
                        metadata['inode'], metadata['device']))
            if owns_transaction:
                db.execute('COMMIT')
        except BaseException:
            if owns_transaction:
                db.execute('ROLLBACK')
            raise
        return sha

    def direct_children(self, path):
        key = self.key(path)
        if key is None:
            return []
        prefix = key.rstrip('/')+'/'
        return [self.root / row[0] for row in self.connect().execute(
            'SELECT path FROM paths WHERE path>=? AND path<?', (prefix, prefix+'\uffff'))
            if '/' not in row[0][len(prefix):]]


def bank_row(store, path):
    key = store.key(path)
    if key is None:
        return False
    p = Path(key)
    return key.startswith('artifacts/') and p.parent.name == 'bank' and p.suffix == '.json' and p.name != 'manifest.json'


def install(store, capture_banks=False, prefer_archive=False):
    """Opt-in narrow compatibility layer, inherited by explicitly bootstrapped children."""
    global INSTALLED
    if INSTALLED is not None:
        raise RuntimeError('adapter already installed')
    import shutil
    original_copy = shutil.copyfile
    store._original_copyfile = original_copy

    def opened(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        # Existing physical files remain authoritative. No silent correction of corruption.
        if not isinstance(file, int) and (prefer_archive or not RAW_EXISTS(Path(os.fsdecode(file)))):
            row = store.entry(file)
            if row is not None:
                if any(flag in mode for flag in 'wax+'):
                    raise PermissionError('immutable archived path is read-only: '+str(file))
                stream = io.BytesIO(store.object(row[0]))
                stream.name = os.fspath(file)
                if 'b' in mode:
                    return stream
                return io.TextIOWrapper(stream, encoding=encoding or 'utf-8', errors=errors, newline=newline)
        try:
            return RAW_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)
        except FileNotFoundError:
            # A maintenance process may unlink a verified physical copy between
            # exists() and open(). The committed immutable CAS is the fallback.
            if not isinstance(file, int) and not any(flag in mode for flag in 'wax+'):
                row = store.entry(file)
                if row is not None:
                    stream = io.BytesIO(store.object(row[0])); stream.name = os.fspath(file)
                    return stream if 'b' in mode else io.TextIOWrapper(stream, encoding=encoding or 'utf-8', errors=errors, newline=newline)
            raise

    def path_open(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
        return opened(path, mode, buffering, encoding, errors, newline)

    def exists(path):
        return RAW_EXISTS(path) or store.entry(path) is not None

    def is_file(path):
        return RAW_IS_FILE(path) or store.entry(path) is not None

    def copyfile(src, dst, *, follow_symlinks=True):
        if capture_banks and bank_row(store, dst):
            with opened(src, 'rb') as stream:
                data = stream.read()
            if RAW_EXISTS(Path(dst)):
                with RAW_OPEN(dst, 'rb') as stream:
                    if stream.read() != data:
                        raise ValueError('bank destination differs')
                return dst
            store.put(dst, data, 'bank_row')
            return dst
        return original_copy(src, dst, follow_symlinks=follow_symlinks)

    def replace(path, target):
        if capture_banks and bank_row(store, target):
            with RAW_OPEN(path, 'rb') as stream:
                data = stream.read()
            if RAW_EXISTS(Path(target)):
                with RAW_OPEN(target, 'rb') as stream:
                    if stream.read() != data:
                        raise ValueError('bank destination differs')
            else:
                store.put(target, data, 'bank_row')
            Path(path).unlink()
            return Path(target)
        if store.entry(target) is not None:
            raise PermissionError('cannot overwrite immutable archived destination')
        if store.entry(path) is not None and not RAW_EXISTS(path):
            raise PermissionError('cannot rename immutable archived path')
        return RAW_REPLACE(path, target)

    def iterdir(path):
        physical = list(RAW_ITERDIR(path))
        seen = {p.name for p in physical}
        yield from physical
        for child in store.direct_children(path):
            if child.name not in seen:
                yield Path(path) / child.name

    def glob(path, pattern, *, case_sensitive=None):
        physical = list(RAW_GLOB(path, pattern, case_sensitive=case_sensitive))
        seen = {str(p) for p in physical}
        yield from physical
        tail = Path(pattern).name
        if not any(c in tail for c in '*?[') and not tail.startswith(('path-', 'v28-path-', 'attempt-', 'search-', 'step-')):
            return
        key = store.key(path)
        if key is None:
            return
        prefix = key.rstrip('/')+'/'
        for row in store.connect().execute('SELECT path FROM paths WHERE path>=? AND path<?',
                                           (prefix, prefix+'\uffff')):
            relative = row[0][len(prefix):]
            # Explicit recursive glob or matching number of path components.
            matches = Path(relative).match(pattern) or (pattern.startswith('**/') and Path(relative).match(pattern[3:]))
            if ('**' in pattern or relative.count('/') == pattern.count('/')) and matches:
                result = Path(path) / relative
                if str(result) not in seen:
                    yield result

    builtins.open = opened
    Path.open = path_open
    Path.exists = exists
    Path.is_file = is_file
    Path.replace = replace
    Path.iterdir = iterdir
    Path.glob = glob
    Path.rglob = lambda path, pattern, **kw: glob(path, '**/'+pattern, **kw)
    shutil.copyfile = copyfile
    INSTALLED = store
    return store


def uninstall():
    global INSTALLED
    if INSTALLED is None:
        return
    # Test-only; production processes install once and retain the versioned adapter.
    import shutil
    shutil.copyfile = INSTALLED._original_copyfile
    builtins.open = RAW_OPEN
    Path.open, Path.exists, Path.is_file = RAW_PATH_OPEN, RAW_EXISTS, RAW_IS_FILE
    Path.replace, Path.iterdir = RAW_REPLACE, RAW_ITERDIR
    Path.glob, Path.rglob = RAW_GLOB, RAW_RGLOB
    INSTALLED = None
