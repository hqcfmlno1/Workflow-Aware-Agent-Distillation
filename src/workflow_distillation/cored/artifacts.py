"""Identity-bound, hash-verified local artifacts. Completion markers are written last."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import uuid

import torch

from workflow_distillation.epoch_data import sha256_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def exclusive(root):
    """One writer per stage; POSIX lock is released on normal exit or process death."""
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Stage already running: {root}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Store:
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'identity.json'
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError('Cache identity differs; choose a new output directory')
        else:
            atomic_json(path, identity)
        (self.root / 'parts').mkdir(exist_ok=True)

    def _path(self, key):
        if not key or any(c not in '0123456789abcdefghijklmnopqrstuvwxyz-_' for c in key):
            raise ValueError('Unsafe part key')
        return self.root / 'parts' / f'{key}.json'

    def put(self, key, value):
        path = self._path(key)
        atomic_json(path, value)
        atomic_json(path.with_suffix('.sha.json'), {'sha256': sha256_file(path)})

    def get(self, key):
        path = self._path(key)
        check = path.with_suffix('.sha.json')
        if not check.exists():
            return None  # An interrupted write is recomputed.
        if not path.exists() or sha256_file(path) != json.loads(check.read_text())['sha256']:
            raise ValueError(f'Cache checksum differs: {key}')
        return json.loads(path.read_text())


def save_checkpoint(root, step, state, identity):
    root = Path(root)
    path = root / f'checkpoint-{step:08d}'
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'complete.json').exists():
        raise FileExistsError(f'Checkpoint already committed: {path}')
    tmp = path / 'state.pt.tmp'
    with tmp.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path / 'state.pt')
    atomic_json(path / 'complete.json', dict(step=step, identity=identity,
                sha256=sha256_file(path / 'state.pt')))
    atomic_json(root / 'latest.json', {'checkpoint': path.name})
    return path


def load_checkpoint(path, identity):
    path = Path(path)
    manifest = json.loads((path / 'complete.json').read_text())
    if manifest['identity'] != identity:
        raise ValueError('Checkpoint identity differs')
    if sha256_file(path / 'state.pt') != manifest['sha256']:
        raise ValueError('Checkpoint checksum differs')
    # Our own locally generated tensor/optimizer/RNG checkpoint only, never an untrusted pickle.
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=False)
    if state.get('cursor', manifest['step']) != manifest['step']:
        raise ValueError('Checkpoint cursor differs from manifest')
    return state


def file_identity(path):
    path = Path(path).resolve()
    if path.is_file():
        return {'sha256': sha256_file(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob('*'))
             if p.is_file() and not any(part.startswith('.') for part in p.relative_to(path).parts)}
    if not files:
        raise ValueError('Empty model/artifact directory')
    return {'files': files, 'sha256': digest(files)}


def verify_export(path):
    path = Path(path)
    m = json.loads((path / 'complete.json').read_text())
    for name, expected in m['files'].items():
        candidate = (path / name).resolve()
        if not candidate.is_relative_to(path.resolve()) or sha256_file(candidate) != expected:
            raise ValueError(f'Export checksum differs: {name}')
    if not m['files']:
        raise ValueError('Empty export')
    return m


def finish_export(path, identity):
    path = Path(path)
    files = {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob('*'))
             if p.is_file() and p.name != 'complete.json'}
    atomic_json(path / 'complete.json', dict(identity=identity, files=files))
