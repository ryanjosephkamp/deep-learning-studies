"""
common.py - configuration, seeding, environment logging, and resumable units for the study.

Every unit of work writes its result to a temporary name and renames it into place, and a
unit whose final file exists is skipped, so a stopped run picks up where it left off.
"""

import os
import sys
import json
import time
import random
import hashlib
import platform
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def load_config(smoke=False):
    with open(os.path.join(HERE, 'config.json'), encoding='utf-8') as fh:
        cfg = json.load(fh)
    if smoke:
        s = cfg['smoke']
        cfg['split']['test_identities'] = s['test_identities']
        cfg['crop_size'] = s['crop_size']
        cfg['frame_size'] = s['frame_size']
        cfg['frame_levels'] = s['frame_levels']
        cfg['train'].update(epochs=s['epochs'], batch_size=s['batch_size'], seeds=s['seeds'],
                            warmup_epochs=0, triplet_identities_per_batch=8)
        cfg['attention_probes'] = 16
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_for(*key):
    """A NumPy generator keyed by strings and integers, so any single draw can be rerun alone."""
    digest = hashlib.sha256('/'.join(str(k) for k in key).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_path(final):
    """Temporary path next to `final`; call `commit(tmp, final)` once the file is complete."""
    os.makedirs(os.path.dirname(final), exist_ok=True)
    root, ext = os.path.splitext(final)
    if final.endswith('.csv.gz'):
        root, ext = final[:-7], '.csv.gz'
    return f'{root}.tmp{os.getpid()}{ext}'


def commit(tmp, final):
    os.replace(tmp, final)


def write_json(path, obj):
    tmp = atomic_path(path)
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, default=str)
    commit(tmp, path)


def save_npy(path, arr):
    tmp = atomic_path(path)
    with open(tmp, 'wb') as fh:
        np.save(fh, arr)
    commit(tmp, path)


def save_csv(path, df):
    tmp = atomic_path(path)
    df.to_csv(tmp, index=False, compression='gzip' if path.endswith('.gz') else None)
    commit(tmp, path)


class Units:
    """Counts units, skips finished ones, and prints one progress line per unit."""

    def __init__(self, total):
        self.total, self.k, self.ran, self.skipped = total, 0, 0, 0
        self.t0 = time.time()

    def run(self, name, final, fn):
        self.k += 1
        if os.path.exists(final):
            self.skipped += 1
            print(f'[{self.k}/{self.total}] {name} done before, skipped', flush=True)
            return
        t = time.time()
        fn()
        if not os.path.exists(final):
            raise RuntimeError(f'unit {name} finished without writing {final}')
        self.ran += 1
        print(f'[{self.k}/{self.total}] {name} {time.time() - t:.1f}s', flush=True)


def environment(device):
    """Versions, hardware, and determinism settings for the run log."""
    import torch
    env = {'python': platform.python_version(), 'platform': platform.platform(),
           'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
           'cudnn_benchmark': torch.backends.cudnn.benchmark,
           'cudnn_deterministic': torch.backends.cudnn.deterministic,
           'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
           'tf32_cudnn': torch.backends.cudnn.allow_tf32}
    for mod in ['cv2', 'pandas', 'PIL', 'torchvision']:
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    if torch.cuda.is_available():
        env['gpu'] = torch.cuda.get_device_name(0)
        env['cuda'] = torch.version.cuda
        env['cudnn'] = torch.backends.cudnn.version()
        try:
            env['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            env['driver'] = None
    env['cpu_count'] = os.cpu_count()
    return env
