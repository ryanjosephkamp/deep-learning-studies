"""
match_images.py - find each local image's CelebA-HQ index by matching pixels against a copy of
CelebA-HQ whose file names are that index (CelebA-Dialog's "Image (HQ)" folder, 0.jpg to
29999.jpg, the copy that shares the identity labels' numbering).

Copies of CelebA-HQ don't all keep its numbering: the Kaggle 256 x 256 copy numbers its files
00000.jpg to 29999.jpg in an order unrelated to the CelebA-HQ index that the identity labels use.
Both copies are the same 30,000 pictures, so a small thumbnail of each one identifies it.

Method: every image is decoded at reduced size, shrunk to 24 x 24 color, and normalized (mean
removed, unit length), so the match is a normalized correlation. Each local image takes its most
correlated reference image; where two local images want the same reference image (CelebA-HQ
holds a few near-duplicate pictures), a greedy assignment over each image's five best candidates
keeps the map one-to-one.

    python study/match_images.py --images DIR --ref REF_DIR --out image_map.csv

Writes image_map.csv (file, hq_index, corr, second, margin) and prints how clear the matches are.
A correct match has corr near 1 and a clear margin over the second-best reference image. It also
prints two fingerprints to compare with the ones in study/README.md: one of the local image files
(their names and bytes) and one of the map's (file, hq_index) pairs.
"""

import os
import re
import sys
import time
import argparse
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import image_set_fingerprint, map_fingerprint  # noqa: E402

SIDE = 24
TOPK = 5


def _number(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else None


def _thumb(path):
    import cv2
    from PIL import Image
    cv2.setNumThreads(1)
    try:
        im = Image.open(path)
        im.draft('RGB', (4 * SIDE, 4 * SIDE))  # JPEG decodes at 1/2 to 1/8 size: fast for 1024 px
        img = np.asarray(im.convert('RGB'))
    except OSError:
        return None
    t = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / max(float(np.linalg.norm(t)), 1e-6)


def thumbs(folder, names, workers):
    paths = [os.path.join(folder, n) for n in names]
    with Pool(workers) as pool:
        out = pool.map(_thumb, paths, chunksize=64)
    bad = [n for n, t in zip(names, out) if t is None]
    if bad:
        sys.exit(f'{len(bad)} images in {folder} could not be read, e.g. {bad[:3]}')
    return np.stack(out)


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--images', required=True, help='the local copy to map (e.g. celeba_hq_256)')
    p.add_argument('--ref', required=True, help='reference copy whose file numbers are the CelebA-HQ index')
    p.add_argument('--out', required=True, help='CSV to write')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()

    names = list_images(a.images)
    ref_names = list_images(a.ref)
    numbers = [_number(n) for n in ref_names]
    if None in numbers or len(set(numbers)) != len(numbers):
        sys.exit(f'{a.ref}: every reference file name must carry a distinct number (its CelebA-HQ index)')
    ref_index = np.array(numbers)
    print(f'{len(names)} images to map, {len(ref_names)} reference images', flush=True)
    q = thumbs(a.images, names, a.workers)
    print(f'  local thumbnails {time.time() - t0:.0f}s', flush=True)
    r = thumbs(a.ref, ref_names, a.workers)
    print(f'  reference thumbnails {time.time() - t0:.0f}s', flush=True)

    # Top-k reference candidates for every local image, in chunks to bound memory.
    k = min(TOPK, len(ref_names))
    cand = np.zeros((len(q), k), dtype=np.int64)
    corr = np.zeros((len(q), k), dtype=np.float32)
    for i in range(0, len(q), 2048):
        s = q[i:i + 2048] @ r.T
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, top, 1), axis=1)
        cand[i:i + 2048] = np.take_along_axis(top, order, 1)
        corr[i:i + 2048] = np.take_along_axis(s, cand[i:i + 2048], 1)

    # Greedy one-to-one assignment, most confident pairs first.
    pairs = sorted(((float(corr[i, j]), i, int(cand[i, j])) for i in range(len(q)) for j in range(k)),
                   reverse=True)
    assigned = np.full(len(q), -1)
    taken = set()
    for c, i, j in pairs:
        if assigned[i] < 0 and j not in taken:
            assigned[i] = j
            taken.add(j)
    unmatched = int((assigned < 0).sum())

    best = corr[:, 0]
    second = corr[:, 1] if k > 1 else np.zeros(len(q))
    got = np.array([corr[i, list(cand[i]).index(assigned[i])] if assigned[i] >= 0 else np.nan
                    for i in range(len(q))])
    not_first = int(sum(assigned[i] >= 0 and assigned[i] != cand[i, 0] for i in range(len(q))))

    import pandas as pd
    df = pd.DataFrame({'file': names,
                       'hq_index': [int(ref_index[j]) if j >= 0 else -1 for j in assigned],
                       'corr': np.round(got, 5), 'second': np.round(second, 5),
                       'margin': np.round(best - second, 5)})
    df = df[df.hq_index >= 0]
    tmp = a.out + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, a.out)

    print(f'mapped {len(df)} of {len(names)} images ({unmatched} unmatched) in {time.time() - t0:.0f}s')
    print(f'  correlation of the match: min {np.nanmin(got):.4f}, 1st percentile {np.nanpercentile(got, 1):.4f}, '
          f'median {np.nanmedian(got):.4f}')
    print(f'  best minus second-best: median {np.median(best - second):.4f}; '
          f'{int(((best - second) < 0.02).sum())} images below 0.02 (near-duplicate pictures)')
    print(f'  {not_first} images took a candidate other than their best, to keep the map one-to-one')
    print(f'  images matched with correlation below 0.9: {int((got < 0.9).sum())}')
    print(f'  image set fingerprint ({len(names)} files in {a.images}): {image_set_fingerprint(a.images, names)}')
    print(f'  map fingerprint (file,hq_index pairs): {map_fingerprint(df[["file", "hq_index"]])}')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
