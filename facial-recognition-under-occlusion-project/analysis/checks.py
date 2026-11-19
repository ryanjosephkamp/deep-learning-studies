#!/usr/bin/env python3
"""
checks.py - CPU checks of the original runs on their saved image samples.

The original runs saved each sample of CelebA-HQ images under
experiments/<run>/sample_photos/ (and processed_photos/ for the background-removed runs). Those
images aren't in the repository; point --experiments at the folder that holds the run folders.

    python analysis/checks.py --experiments DIR --check pixels
    python analysis/checks.py --experiments DIR --check custom --weights MODEL_DIR
    python analysis/checks.py --experiments DIR --check detector --sface FILE

pixels    Nearest neighbor on raw pixels (no learning) under the same probe and gallery as the
          study: the occluded copy of each image against all clean images. Writes
          analysis/out/check_pixel_baseline.csv.
custom    Reruns the saved custom model on the saved samples and compares with the result files.
          --weights is the folder holding novel_model/model.pth and experiments/<run>/novel_model/.
          Writes analysis/out/check_custom_rerun.csv.
detector  Re-creates the path DeepFace 0.0.96 takes for SFace with its default detector
          ("opencv": a Haar cascade on the image with a 50% black border, scale 1.1, 10
          neighbors; with enforce_detection=False a missed face falls back to the whole image)
          and compares it with two paths that crop probe and gallery the same way. --sface is
          face_recognition_sface_2021dec.onnx. Eye alignment is left out, so the numbers
          approximate DeepFace's rather than reproduce them. Writes
          analysis/out/check_detector_sface.csv and check_detector_face_coverage.csv.

CPU only, 2 threads. The pixel check takes about a minute, the detector check about two, and
the custom reruns about 15 minutes for all runs listed in CUSTOM_RUNS.
"""

import os
import sys
import time
import argparse

os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'analysis', 'out')
RESULTS = os.path.join(ROOT, 'results', 'original')
LEVELS = np.round(np.linspace(0, 1, 21), 2)

PIXEL_RUNS = [
    ('exper__size_50__20251201_192436', 'sample_photos'),
    ('exper__size_100__20251130_235656', 'sample_photos'),
    ('exper__size_200__20251201_203249', 'sample_photos'),
    ('exper__size_500__20251201_004627', 'sample_photos'),
    ('exper__NO_BG__size_100__20251203_012434', 'processed_photos'),
]
# (label, run folder, image subfolder, weights relative to --weights, result file to compare)
CUSTOM_RUNS = [
    ('N100', 'exper__size_100__20251130_235656', 'sample_photos', 'novel_model/model.pth',
     'exper__size_100__20251130_235656'),
    ('N50', 'exper__size_50__20251201_192436', 'sample_photos', 'novel_model/model.pth',
     'exper__size_50__20251201_192436'),
    ('N200', 'exper__size_200__20251201_203249', 'sample_photos', 'novel_model/model.pth',
     'exper__size_200__20251201_203249'),
    ('NOBG100_finetuned', 'exper__NO_BG__size_100__20251203_012434', 'processed_photos',
     'experiments/exper__NO_BG__size_100__20251203_012434/novel_model/model_nobg_finetuned.pth',
     'exper__NO_BG__size_100__20251203_012434'),
    ('NOBG100_master', 'exper__NO_BG__size_100__20251203_012434', 'processed_photos',
     'novel_model/model.pth', None),
]


def occlude(arr, ratio):
    """The project's centered black square (frfpv1.data_utils.apply_occlusion) on a uint8 array."""
    arr = arr.copy()
    h, w = arr.shape[:2]
    side = int(np.sqrt(ratio * h * w))
    top, left = max(0, (h - side) // 2), max(0, (w - side) // 2)
    arr[top:min(h, top + side), left:min(w, left + side)] = 0
    return arr


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')))


def nearest_is_self(probes, gallery):
    """Share of probes whose nearest gallery row (L2) is their own index."""
    d = (probes ** 2).sum(1)[:, None] - 2 * probes @ gallery.T + (gallery ** 2).sum(1)[None, :]
    return float((d.argmin(1) == np.arange(len(probes))).mean()), d.argmin(1)


def check_pixels(exp_dir):
    out = {}
    for run, sub in PIXEL_RUNS:
        folder = os.path.join(exp_dir, run, sub)
        imgs = np.stack([np.asarray(Image.open(os.path.join(folder, f)).convert('RGB').resize((256, 256)))
                         for f in list_images(folder)])
        gallery = imgs.reshape(len(imgs), -1).astype(np.float32)
        accs = []
        for r in LEVELS:
            probes = np.stack([occlude(x, r) for x in imgs]).reshape(len(imgs), -1).astype(np.float32)
            accs.append(nearest_is_self(probes, gallery)[0])
        out[run] = accs
        print(f'{run} (N = {len(imgs)}): ' + ' '.join(f'{a:.2f}' for a in accs), flush=True)
    df = pd.DataFrame(out, index=pd.Index(LEVELS, name='Occlusion'))
    df.to_csv(os.path.join(OUT, 'check_pixel_baseline.csv'))


def check_custom(exp_dir, weights_dir):
    import torch
    import torchvision.transforms as T
    torch.set_num_threads(2)
    sys.path.insert(0, ROOT)
    from frfpv1.models import OcclusionAwareFaceModel
    to_tensor = T.ToTensor()
    rows = []
    for label, run, sub, wpath, result_file in CUSTOM_RUNS:
        model = OcclusionAwareFaceModel(128)
        model.load_state_dict(torch.load(os.path.join(weights_dir, wpath), map_location='cpu', weights_only=True))
        model.eval()
        folder = os.path.join(exp_dir, run, sub)
        imgs = [np.asarray(Image.open(os.path.join(folder, f)).convert('RGB').resize((256, 256)))
                for f in list_images(folder)]
        saved = pd.read_csv(os.path.join(RESULTS, result_file + '.csv')).round({'Occlusion': 2}) \
            .set_index('Occlusion')['Custom'] if result_file else None

        def embed(ratio):
            x = torch.stack([to_tensor(occlude(a, ratio)) for a in imgs])
            with torch.no_grad():
                return torch.cat([model(x[i:i + 16]) for i in range(0, len(x), 16)]).numpy()

        t0 = time.time()
        gallery = embed(0.0)
        for r in LEVELS:
            acc = nearest_is_self(embed(r), gallery)[0]
            rows.append({'run': label, 'N': len(imgs), 'Occlusion': r, 'rerun': acc,
                         'saved': saved.loc[r] if saved is not None else np.nan})
        print(f'{label}: done in {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df['diff_points'] = ((df['rerun'] - df['saved']) * 100).round(1)
    df.to_csv(os.path.join(OUT, 'check_custom_rerun.csv'), index=False)
    for label, g in df.groupby('run', sort=False):
        if g['saved'].notna().any():
            print(f'{label}: largest difference from the result file {g["diff_points"].abs().max():.1f} points; '
                  f'{int((g["diff_points"] == 0).sum())} of 21 levels identical')


def check_detector(exp_dir, sface_path, run='exper__size_100__20251130_235656'):
    import cv2
    cv2.setNumThreads(2)
    folder = os.path.join(exp_dir, run, 'sample_photos')
    imgs = [cv2.imread(os.path.join(folder, f)) for f in list_images(folder)]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rec = cv2.FaceRecognizerSF.create(sface_path, '')

    def detect(img):
        h, w = img.shape[:2]
        bh, bw = h // 2, w // 2
        big = cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        try:
            faces, _, _ = cascade.detectMultiScale3(big, 1.1, 10, outputRejectLevels=True)
        except cv2.error:
            faces = ()
        if len(faces) == 0:
            return None
        x, y, fw, fh = faces[0]
        x0, y0, x1, y1 = max(0, x - bw), max(0, y - bh), min(w, x - bw + fw), min(h, y - bh + fh)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def to112(face):
        f = min(112 / face.shape[0], 112 / face.shape[1])
        r = cv2.resize(face, (int(face.shape[1] * f), int(face.shape[0] * f)))
        d0, d1 = 112 - r.shape[0], 112 - r.shape[1]
        r = np.pad(r, ((d0 // 2, d0 - d0 // 2), (d1 // 2, d1 - d1 // 2), (0, 0)))
        return cv2.resize(r, (112, 112)) if r.shape[:2] != (112, 112) else r

    def crop(img, box):
        return img if box is None else img[box[1]:box[3], box[0]:box[2]]

    def emb(face):
        return rec.feature(np.ascontiguousarray(to112(face))).ravel()

    boxes = [detect(im) for im in imgs]
    gallery = {'deepface_path': np.stack([emb(crop(im, b)) for im, b in zip(imgs, boxes)]),
               'whole_image': np.stack([emb(im) for im in imgs])}
    gallery['same_crop'] = gallery['deepface_path']
    rows = []
    for r in LEVELS:
        occ = [occlude(im, r) for im in imgs]
        found = [detect(o) for o in occ]
        probes = {'deepface_path': np.stack([emb(crop(o, b)) for o, b in zip(occ, found)]),
                  'same_crop': np.stack([emb(crop(o, b)) for o, b in zip(occ, boxes)]),
                  'whole_image': np.stack([emb(o) for o in occ])}
        row = {'Occlusion': r, 'face_detected': float(np.mean([b is not None for b in found]))}
        for name, p in probes.items():
            acc, nn = nearest_is_self(p, gallery[name])
            row[name] = acc
            row[name + '_largest_share_to_one_gallery_image'] = float(np.bincount(nn).max() / len(nn))
        rows.append(row)
        print(f"{r:.2f}: detected {row['face_detected']:.2f}  deepface path {row['deepface_path']:.2f}  "
              f"same crop {row['same_crop']:.2f}  whole image {row['whole_image']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    saved = pd.read_csv(os.path.join(RESULTS, run + '.csv')).round({'Occlusion': 2}).set_index('Occlusion')
    df['saved_sface'] = df['Occlusion'].map(saved['SFace'])
    df.to_csv(os.path.join(OUT, 'check_detector_sface.csv'), index=False)

    cover = []
    found = [b for b in boxes if b is not None]
    for r in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        side = int(np.sqrt(r * 256 * 256))
        t = (256 - side) // 2
        shares = [max(0, min(x1, t + side) - max(x0, t)) * max(0, min(y1, t + side) - max(y0, t)) /
                  ((x1 - x0) * (y1 - y0)) for x0, y0, x1, y1 in found]
        cover.append({'Occlusion': r, 'square_side_px': side, 'faces': len(found),
                      'median_face_box_side_px': float(np.median([x1 - x0 for x0, _, x1, _ in found])),
                      'median_share_of_face_box_covered': float(np.median(shares))})
    pd.DataFrame(cover).to_csv(os.path.join(OUT, 'check_detector_face_coverage.csv'), index=False)
    print(pd.DataFrame(cover).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--experiments', required=True, help='folder holding the original run folders')
    p.add_argument('--check', required=True, choices=['pixels', 'custom', 'detector'])
    p.add_argument('--weights', help='folder holding novel_model/ and experiments/<run>/novel_model/')
    p.add_argument('--sface', help='path to face_recognition_sface_2021dec.onnx')
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.check == 'pixels':
        check_pixels(a.experiments)
    elif a.check == 'custom':
        check_custom(a.experiments, a.weights or os.path.dirname(a.experiments))
    else:
        check_detector(a.experiments, a.sface)


if __name__ == '__main__':
    main()
