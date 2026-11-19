"""
check_labels.py - test whether identity labels fit the images, before anything is trained on them.

Embeds a random sample of the aligned crops that run_study.py's align stage wrote
(<results>/prepare/aligned_*.npz) with a pretrained recognizer (ArcFace by default), then scores
each way of reading the labels:

  file numbers   each image's file number taken as its CelebA-HQ index (run_study.py's default)
  <map>.csv      each image's CelebA-HQ index taken from an image map (--image-map, repeatable)

For each it reports, on the sample: the mean cosine similarity of pairs that share a label
(genuine) and of pairs that don't (impostor); rank-1, the share of images whose most similar other
image shares its label (among images with at least one same-label partner in the sample); and
the share of the k most similar pairs that share a label (k = the number of same-label pairs, at
most 300). Labels that fit give genuine pairs well above impostors and a high rank-1; labels that
don't fit give genuine and impostor pairs the same mean and rank-1 near zero. A labeling passes at
rank-1 >= 0.5, the floor run_study.py's label check uses.

    python study/check_labels.py --results DIR --labels FILE --weights DIR [--image-map MAP.csv ...]

Runs on CPU; 5,000 crops take one to two minutes on 30 threads. Exit code 0 if the last
labeling listed passes, 3 if not.
"""

import os
import sys
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--results', required=True, help='a run_study.py output folder whose align stage finished')
    p.add_argument('--labels', required=True)
    p.add_argument('--image-map', action='append', default=[], help='CSV (file, hq_index); may be repeated')
    p.add_argument('--weights', help='folder with the recognizer ONNX file')
    p.add_argument('--model', default='arcface_r50', help='a key from config.json "pretrained"')
    p.add_argument('--sample', type=int, default=5000)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--mock', action='store_true', help='smoke-test stand-in for the recognizer')
    a = p.parse_args()
    if not a.mock and not a.weights:
        p.error('--weights is required')
    return a


def score(emb, ids, max_top=300):
    import numpy as np
    s = emb @ emb.T
    np.fill_diagonal(s, -np.inf)
    same = ids[:, None] == ids[None, :]
    np.fill_diagonal(same, False)
    iu = np.triu_indices(len(ids), 1)
    sv, gv = s[iu], same[iu]
    has_partner = same.any(1)
    rank1 = float((ids[s.argmax(1)] == ids)[has_partner].mean()) if has_partner.any() else float('nan')
    top = np.argsort(-sv)[:max(1, min(max_top, int(gv.sum())))]
    return {'images': len(ids), 'genuine_pairs': int(gv.sum()),
            'genuine_cos': float(sv[gv].mean()) if gv.any() else float('nan'),
            'impostor_cos': float(sv[~gv].mean()),
            'rank1': rank1, 'with_partner': int(has_partner.sum()),
            'top_pairs_same_label': float(gv[top].mean())}


def main():
    a = parse_args()
    threads = a.threads or os.cpu_count()
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[var] = str(threads)
    import numpy as np
    import common
    import data
    import nets

    print(f'label file SHA-256: {common.sha256_of(a.labels)}', flush=True)
    for m in a.image_map:
        print(f'{os.path.basename(m)} fingerprint (file,hq_index pairs): {data.map_fingerprint(m)}', flush=True)
    paths = sorted(glob.glob(os.path.join(a.results, 'prepare', 'aligned_*.npz')))
    if not paths:
        sys.exit(f'no aligned crops in {a.results}/prepare; run the align stage first')
    parts = [np.load(p, allow_pickle=True) for p in paths]
    files = np.concatenate([p['files'] for p in parts])
    crops = np.concatenate([p['crops'] for p in parts])
    ok = np.concatenate([p['ok'] for p in parts])
    files, crops = files[ok], crops[ok]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(files), size=min(a.sample, len(files)), replace=False))
    files, crops = files[pick], crops[pick]

    cfg = common.load_config(smoke=a.mock)
    if a.mock:
        emb_fn = nets.MockEmbedder(128, crops.shape[1], seed=0)
    else:
        spec = next(m for m in cfg['pretrained'] if m['key'] == a.model)
        emb_fn = nets.OnnxEmbedder(os.path.join(a.weights, spec['file']), spec, threads)
    print(f'embedding {len(files)} aligned crops with {"a mock model" if a.mock else a.model}...', flush=True)
    emb = emb_fn(crops)

    all_files = sorted(set(np.concatenate([p['files'] for p in parts]).tolist()))
    readings = [('file numbers', None)] + [(os.path.basename(m), m) for m in a.image_map]
    print(f'\n{"labeling":28s} {"genuine":>8s} {"impostor":>9s} {"rank-1":>7s} {"top-k":>8s}  pairs  verdict')
    passed = False
    for name, map_path in readings:
        image_map = data.read_image_map(map_path, all_files) if map_path else None
        labels, _ = data.read_labels(a.labels, all_files, image_map, whole_set=False)
        lab = dict(zip(labels['file'], labels['identity']))
        keep = np.array([f in lab for f in files])
        ids = np.array([lab[f] for f in files[keep]])
        r = score(emb[keep], ids)
        passed = r['rank1'] >= 0.5
        print(f'{name:28s} {r["genuine_cos"]:8.3f} {r["impostor_cos"]:9.3f} {r["rank1"]:7.3f} '
              f'{r["top_pairs_same_label"]:8.3f}  {r["genuine_pairs"]:5d}  {"fits" if passed else "DOES NOT FIT"}',
              flush=True)
    print('\ngenuine/impostor: mean cosine of same-label / different-label pairs in the sample; rank-1 over '
          'images with a same-label partner in the sample; top-k: share of the k most similar pairs that '
          'share a label, k = same-label pairs (at most 300). A labeling fits at rank-1 >= 0.5.')
    sys.exit(0 if passed else 3)


if __name__ == '__main__':
    main()
