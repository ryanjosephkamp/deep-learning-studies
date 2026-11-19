#!/usr/bin/env python3
"""
check_near_copies.py - a sensitivity check: how much of five models' clean identity rank-1 comes from
the probes most similar to their own gallery image (CPU, seconds).

    python study/check_near_copies.py --results DIR    (from the project folder)

CelebA-HQ holds a few near-duplicate pictures, and the split does not remove them, so a probe can be
almost the same picture as its identity's gallery image, which brings identification close to
matching an image to itself; that matters most for the weakest scores. A probe is flagged here when ArcFace R50's cosine similarity
between it and its own gallery image, both clean, is at least THRESHOLD; whether a flagged pair is a
near-duplicate picture was not checked by eye. For ArcFace R50, SFace, raw crop pixels, the original
model, and raw frame pixels, this writes clean identity rank-1 with and without the flagged probes.

Reads DIR/eval/<model>/identity__square_0.00.csv.gz (identity__frame_square_0.00.csv.gz for the
frame models); writes ../results/study/near_copies.csv. The per-probe files are not in this
repository (they carry CelebA file names), so the output is committed and the numbers in the paper
come from it.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

THRESHOLD = 0.9
MODELS = {  # model folder: clean identity file
    'arcface_r50': 'identity__square_0.00.csv.gz', 'sface': 'identity__square_0.00.csv.gz',
    'pixels_crop': 'identity__square_0.00.csv.gz', 'original_model': 'identity__frame_square_0.00.csv.gz',
    'pixels_frame': 'identity__frame_square_0.00.csv.gz',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--results', required=True, type=Path)
    args = ap.parse_args()
    ev = args.results / 'eval'
    arc = pd.read_csv(ev / 'arcface_r50' / MODELS['arcface_r50']).set_index('probe')
    near = arc['genuine'] >= THRESHOLD
    rows = []
    for model, name in MODELS.items():
        d = pd.read_csv(ev / model / name).set_index('probe')
        hit = d['rank'] == 1
        nc = near.reindex(d.index)
        assert nc.notna().all(), f'{model}: probes differ from ArcFace R50'
        nc = nc.astype(bool)
        rows.append({'model': model, 'probes': len(d), 'threshold': THRESHOLD, 'near_copies': int(nc.sum()),
                     'hits': int(hit.sum()), 'hits_on_near_copies': int(hit[nc].sum()),
                     'rank1': hit.mean(), 'rank1_without_near_copies': hit[~nc].mean()})
    out = Path(__file__).resolve().parent.parent / 'results/study/near_copies.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:15s} {r['hits']:5d} hits, {r['hits_on_near_copies']} on the {r['near_copies']} flagged probes; "
              f"rank-1 {100 * r['rank1']:.2f}% -> {100 * r['rank1_without_near_copies']:.2f}%")


if __name__ == '__main__':
    main()
