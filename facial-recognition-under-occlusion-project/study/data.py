"""
data.py - identity labels, the identity-disjoint split, face alignment, and occlusion.

Labels: a text file with one line per CelebA-HQ image, "<image> <identity>" (the image's index
may be written with or without leading zeros and with any extension), as in CelebA-Dialog's
identity file for CelebA-HQ. Lines that don't parse are skipped and counted. By default an image
file's number is taken to be its CelebA-HQ index. Copies of CelebA-HQ that were renumbered (the
Kaggle 256 x 256 copy is one) need an image map: a CSV with columns `file` and `hq_index` giving
each local file's CelebA-HQ index, as written by match_images.py.

Split: test identities are drawn (seeded) from identities with at least `min_images_test`
images; each keeps its lowest-numbered image as the gallery image and the rest as probes.
Every other identity with at least `min_images_train` images is used for training. No
identity is in both.

Alignment: YuNet finds the face and its five landmarks in each 256 x 256 image, and OpenCV's
FaceRecognizerSF.alignCrop maps them onto the standard 112 x 112 ArcFace template. Every model
that takes face crops sees these same crops, and occluders are drawn on the crop, so the
occluder never changes where the face is found.
"""

import os
import re
import hashlib

import numpy as np
import pandas as pd

from common import rng_for

IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def image_set_fingerprint(folder, names=None):
    """SHA-256 over one 'name<TAB>SHA-256 of the file' line per image, in name order."""
    h = hashlib.sha256()
    for n in (list_images(folder) if names is None else names):
        with open(os.path.join(folder, n), 'rb') as fh:
            h.update(f'{n}\t{hashlib.sha256(fh.read()).hexdigest()}\n'.encode())
    return h.hexdigest()


def map_fingerprint(path_or_df):
    """SHA-256 over one 'file,hq_index' line per image, in file order: the image map without its scores."""
    m = pd.read_csv(path_or_df, usecols=['file', 'hq_index']) if isinstance(path_or_df, str) else path_or_df
    m = m.assign(file=m['file'].map(os.path.basename)).sort_values('file')
    return hashlib.sha256(''.join(f'{f},{int(i)}\n' for f, i in zip(m['file'], m['hq_index'])).encode()).hexdigest()


def image_index(name):
    m = re.search(r'(\d+)', os.path.basename(name))
    return int(m.group(1)) if m else None


def read_image_map(path, files):
    """Read a file -> CelebA-HQ index map (CSV with columns file and hq_index); check it is one-to-one."""
    m = pd.read_csv(path, usecols=['file', 'hq_index'])
    m['file'] = m['file'].map(os.path.basename)
    if m['file'].duplicated().any() or m['hq_index'].duplicated().any():
        raise ValueError(f'{path}: the image map must be one-to-one (a file or an index appears twice)')
    known = set(files)
    m = m[m['file'].isin(known)]
    if len(m) < 0.99 * len(files):
        raise ValueError(f'{path}: the image map covers {len(m)} of {len(files)} images')
    return dict(zip(m['file'], m['hq_index'].astype(int)))


def read_labels(path, files, image_map=None, whole_set=True):
    """Map each local image file to an identity. Returns (DataFrame file/identity, skipped lines).

    With `image_map` (file -> CelebA-HQ index), labels are looked up by that index instead of the
    number in the file name, and the index is kept in an `hq_index` column. Pass whole_set=False
    when `files` is only part of the image set, to skip the label-count check.
    """
    by_index = {}
    skipped = 0
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2 or not parts[-1].lstrip('-').isdigit() or image_index(parts[0]) is None:
                skipped += line.strip() != ''
                continue
            by_index[image_index(parts[0])] = int(parts[-1])
    if whole_set and len(by_index) > 1.05 * len(files):
        # CelebA's own identity file numbers its 202,599 images differently from CelebA-HQ.
        raise ValueError(f'the label file has {len(by_index)} entries for {len(files)} images; '
                         'it looks like a label file for a different image set (CelebA, not CelebA-HQ)')
    if image_map is None:
        rows = [(f, by_index[image_index(f)]) for f in files if image_index(f) in by_index]
        df = pd.DataFrame(rows, columns=['file', 'identity'])
    else:
        rows = [(f, by_index[image_map[f]], image_map[f]) for f in files
                if f in image_map and image_map[f] in by_index]
        df = pd.DataFrame(rows, columns=['file', 'identity', 'hq_index'])
    if len(df) < 0.99 * len(files):
        raise ValueError(f'labels cover {len(df)} of {len(files)} images; check the label file format')
    return df, skipped


def make_split(labels, cfg, seed):
    """Assign each image to train, gallery, or probe, with identities disjoint between train and test."""
    counts = labels.groupby('identity')['file'].count()
    eligible = np.sort(counts[counts >= cfg['split']['min_images_test']].index.values)
    rng = rng_for('split', seed)
    n_test = min(cfg['split']['test_identities'], len(eligible) // 2)
    test_ids = set(rng.choice(eligible, size=n_test, replace=False).tolist())
    df = labels.copy()
    df['index'] = df['file'].map(image_index)
    df = df.sort_values(['identity', 'index']).reset_index(drop=True)
    df['role'] = 'unused'
    is_test = df['identity'].isin(test_ids)
    first = ~df.duplicated('identity')
    df.loc[is_test & first, 'role'] = 'gallery'
    df.loc[is_test & ~first, 'role'] = 'probe'
    train_ids = counts[(counts >= cfg['split']['min_images_train'])].index.difference(list(test_ids))
    df.loc[df['identity'].isin(train_ids), 'role'] = 'train'
    return df


class Aligner:
    """YuNet detection plus five-point alignment to the 112 x 112 ArcFace template."""

    def __init__(self, detector_path, sface_path, score_threshold, size):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(detector_path, '', (256, 256), score_threshold, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(sface_path, '')
        self.size = size

    def __call__(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None, None
        face = faces[np.argmax(faces[:, 14])]
        crop = self.rec.alignCrop(img_bgr, face)
        if self.size != crop.shape[0]:
            crop = self.cv2.resize(crop, (self.size, self.size), interpolation=self.cv2.INTER_AREA)
        return crop, face


class ResizeAligner:
    """Smoke-test stand-in: resizes the whole image to the crop size."""

    def __init__(self, size):
        import cv2
        self.cv2, self.size = cv2, size

    def __call__(self, img_bgr):
        return self.cv2.resize(img_bgr, (self.size, self.size), interpolation=self.cv2.INTER_AREA), None


# ---- occlusion -------------------------------------------------------------------------------

def conditions(cfg):
    """The probe conditions, by name. Each is drawn on the aligned crop."""
    c = cfg['conditions']
    out = {}
    for cov in c['square_coverages']:
        out[f'square_{cov:.2f}'] = {'kind': 'square', 'coverage': cov, 'fill': 'black', 'position': 'center'}
    out['eyes_band'] = {'kind': 'band', 'rows': c['eyes_band_rows'], 'fill': 'black'}
    out['lower_band'] = {'kind': 'band', 'rows': c['lower_band_rows'], 'fill': 'black'}
    cov = c['random_square_coverage']
    out[f'random_square_{cov:.2f}'] = {'kind': 'square', 'coverage': cov, 'fill': 'black', 'position': 'random'}
    for fill, cov in c['fill_variants']:
        out[f'{fill}_square_{cov:.2f}'] = {'kind': 'square', 'coverage': cov, 'fill': fill, 'position': 'center'}
    return out


def same_image_conditions(cfg):
    return {f'square_{cov:.2f}': {'kind': 'square', 'coverage': cov, 'fill': 'black', 'position': 'center'}
            for cov in cfg['conditions']['same_image_coverages']}


def square_box(h, w, coverage):
    """Side and centered top-left corner of the project's square (frfpv1.data_utils.apply_occlusion)."""
    side = int(np.sqrt(coverage * h * w))
    return side, max(0, (h - side) // 2), max(0, (w - side) // 2)


def occlude(images, cond, keys):
    """Draw condition `cond` on a batch of uint8 images (B, H, W, 3); `keys` seed random parts per image."""
    out = images.copy()
    b, h, w, _ = out.shape
    for i in range(b):
        if cond['kind'] == 'band':
            r0, r1 = int(cond['rows'][0] * h), int(cond['rows'][1] * h)
            region = (slice(r0, r1), slice(0, w))
        else:
            side, top, left = square_box(h, w, cond['coverage'])
            if side == 0:
                continue
            if cond.get('position') == 'random':
                g = rng_for('position', keys[i])
                top, left = int(g.integers(0, h - side + 1)), int(g.integers(0, w - side + 1))
            region = (slice(top, min(h, top + side)), slice(left, min(w, left + side)))
        if cond['fill'] == 'black':
            out[i][region] = 0
        elif cond['fill'] == 'gray':
            out[i][region] = 128
        else:
            patch = out[i][region]
            out[i][region] = rng_for('noise', keys[i]).integers(0, 256, size=patch.shape, dtype=np.uint8)
    return out


def occlusion_mask(cond, h, w, key):
    """Boolean (H, W) mask of the region a condition covers, for the attention check."""
    ones = np.ones((1, h, w, 3), dtype=np.uint8)
    return occlude(ones, dict(cond, fill='black'), [key])[0, :, :, 0] == 0


def torch_center_square(x, coverage, apply):
    """Centered black squares on a batch (B, C, H, W) with per-sample coverage; `apply` masks samples."""
    import torch
    b, _, h, w = x.shape
    side = torch.floor(torch.sqrt(coverage * h * w)).long()
    top = torch.clamp((h - side) // 2, min=0)
    left = torch.clamp((w - side) // 2, min=0)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    inside = ((ys >= top.view(b, 1, 1)) & (ys < (top + side).view(b, 1, 1)) &
              (xs >= left.view(b, 1, 1)) & (xs < (left + side).view(b, 1, 1)) & apply.view(b, 1, 1))
    return x.masked_fill(inside.unsqueeze(1), 0)


# ---- smoke data ------------------------------------------------------------------------------

def make_smoke_images(folder, n_ids, per_id, size, seed=0):
    """Synthetic 'identities' for the smoke test: a smooth random base image per identity,
    with a small shift and noise per image. Writes JPEGs and a label file; returns its path."""
    import cv2
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    lines = []
    k = 0
    for ident in range(n_ids):
        base = cv2.resize(rng.integers(0, 256, (8, 8, 3)).astype(np.uint8), (size, size),
                          interpolation=cv2.INTER_CUBIC)
        for _ in range(per_id):
            dx, dy = rng.integers(-2, 3, size=2)
            img = np.roll(base, (int(dy), int(dx)), axis=(0, 1)).astype(np.int16)
            img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
            name = f'{k:05d}.jpg'
            cv2.imwrite(os.path.join(folder, name), img)
            lines.append(f'{name} {ident + 1}')
            k += 1
    label_path = os.path.join(folder, 'labels.txt')
    with open(label_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return label_path
