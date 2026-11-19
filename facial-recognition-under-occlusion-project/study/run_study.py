#!/usr/bin/env python3
"""
run_study.py - how does face identification hold up under occlusion?

The original project matched each occluded image to its own clean copy. This study asks the
identification question: given a different photo of the same person, with part of the face
covered, does the model still pick that person out of 1,000 people it never trained on? And
do the project's two ideas, the spatial attention layer and training with occluded examples,
help when every model sees the same aligned face crop?

Arms (all on the same identity-disjoint split of CelebA-HQ; see data.py):
  pretrained   SFace and ArcFace ResNet-50 (ONNX, CPU) on aligned crops
  pixels       nearest neighbor on raw pixels of the crops and of the frames (no learning)
  original     the project's saved model on full 256 x 256 frames, as it was trained
  train        the project's architecture trained on the training identities: CosFace loss
               with attention on/off x occlusion augmentation on/off, plus a batch-hard
               triplet loss (attention and augmentation on), 3 seeds each
  evaluate     every trained run under every condition, plus an attention check
  deepface     (optional, not in the default stages) DeepFace's six models on the same crops,
               detector skipped; needs deepface and TensorFlow
  summary      one table of rank-1 and TAR per model, protocol, and condition

Run from the repository root:

    python study/run_study.py --images DIR --labels FILE --weights DIR --out DIR --device cuda

--images   CelebA-HQ at 256 x 256 (00000.jpg to 29999.jpg)
--labels   identity labels for those images, one "<image> <identity>" per line
--image-map  a CSV (file, hq_index) giving each image's CelebA-HQ index, for copies whose file
           numbers aren't that index (the Kaggle 256 x 256 copy); write it with match_images.py
           and test it with check_labels.py before a full run
--weights  folder with model.pth (the project's saved model), face_recognition_sface_2021dec.onnx,
           w600k_r50.onnx (ArcFace), and face_detection_yunet_2023mar.onnx (YuNet)
--smoke    a tiny synthetic version (no data, no weights, random or mock models) that finishes
           in under 2 minutes on 2 CPU threads; --labels may point it at another label file

After the pretrained stage, both pretrained models must reach a clean-probe rank-1 of at least
label_check_min_rank1 (config.json); otherwise the job writes HALTED.json and stops before any
training, since a mismatched label file would make every result look like a finding.

Every unit writes to a temporary name and is renamed into place when complete; a restarted run
skips finished units and resumes training from its last checkpoint. DONE.json is written last.
"""

import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ALL_STAGES = ['prepare', 'align', 'pretrained', 'pixels', 'original', 'train', 'evaluate', 'deepface', 'summary']
DEFAULT_STAGES = [s for s in ALL_STAGES if s != 'deepface']
ALIGN_CHUNK = 5000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images')
    p.add_argument('--labels')
    p.add_argument('--image-map', help="CSV (file, hq_index): each image's CelebA-HQ index")
    p.add_argument('--weights')
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], help='default: cuda, or cpu with --smoke')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stages', nargs='+', default=None, choices=ALL_STAGES)
    p.add_argument('--threads', type=int, default=0, help='CPU threads (0 = all; the smoke test uses 2)')
    a = p.parse_args()
    if not a.smoke and not (a.images and a.labels and a.weights):
        p.error('--images, --labels, and --weights are required unless --smoke')
    if a.device is None:
        a.device = 'cpu' if a.smoke else 'cuda'
    return a


def main():
    args = parse_args()
    threads = args.threads or (2 if args.smoke else os.cpu_count())
    # Thread caps must be set before NumPy, OpenCV, or PyTorch load.
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[var] = str(threads)
    global common
    import common  # also puts the repository root on sys.path
    import numpy as np
    import pandas as pd
    import torch
    import cv2
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    cv2.setNumThreads(threads)
    torch.backends.cudnn.benchmark = False
    import data
    import nets
    import evaluate
    from train import train_run

    cfg = common.load_config(args.smoke)
    device = torch.device(args.device)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    stages = args.stages or DEFAULT_STAGES

    # ---- inputs -----------------------------------------------------------------------------
    if args.smoke:
        s = cfg['smoke']
        images = os.path.join(out, 'smoke_data')
        labels_path = os.path.join(images, 'labels.txt')
        if not os.path.exists(labels_path):
            labels_path = data.make_smoke_images(images, s['identities'], s['images_per_identity'], cfg['frame_size'])
        labels_path = args.labels or labels_path  # a smoke run may test another label file
        weights = None
    else:
        images, labels_path, weights = args.images, args.labels, args.weights

    files = data.list_images(images)
    log_path = os.path.join(out, 'run_log.json')
    run_log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    run_log.setdefault('starts', []).append({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': stages,
                                              'smoke': args.smoke, 'threads': threads})
    run_log['environment'] = common.environment(device)
    run_log['config'] = cfg
    hashes = {}
    if weights:
        for name in [cfg['original_model']['file'], cfg['align']['detector_file']] + \
                [m['file'] for m in cfg['pretrained']]:
            path = os.path.join(weights, name)
            hashes[name] = common.sha256_of(path) if os.path.exists(path) else None
        for spec in cfg['pretrained'] + [cfg['original_model']]:
            want, got = spec.get('sha256'), hashes.get(spec['file'])
            if want and got and want != got:
                sys.exit(f"{spec['file']}: SHA-256 {got} does not match the expected {want}")
    run_log['weights_sha256'] = hashes
    labels_sha = common.sha256_of(labels_path)
    if run_log.get('labels_sha256') not in (None, labels_sha):
        sys.exit(f'{labels_path} differs from the label file this output folder was started with; '
                 'its split would be stale. Use a new --out.')
    run_log['labels_sha256'] = labels_sha
    map_sha = common.sha256_of(args.image_map) if args.image_map else None
    if run_log.get('image_map_sha256') != map_sha and os.path.exists(os.path.join(out, 'prepare', 'split.csv')):
        sys.exit('The image map differs from the one this output folder was started with (or one was '
                 'added or dropped); its split would be stale. Use a new --out.')
    run_log['image_map_sha256'] = map_sha
    common.write_json(log_path, run_log)

    # ---- unit plan --------------------------------------------------------------------------
    conds = data.conditions(cfg)
    same = data.same_image_conditions(cfg)
    frame_conds = {f'frame_square_{c:.2f}': {'kind': 'square', 'coverage': c, 'fill': 'black', 'position': 'center'}
                   for c in cfg['frame_levels']}
    runs = [(r, sd) for r in cfg['runs'] for sd in cfg['train']['seeds']]
    n_crop = len(conds) + len(same)
    n_frame = 2 * len(frame_conds)
    plan = {'prepare': 1, 'align': None, 'pretrained': len(cfg['pretrained']) * n_crop, 'pixels': n_crop + n_frame,
            'original': n_frame, 'train': len(runs),
            'evaluate': len(runs) * n_crop + sum(r['attention'] for r, _ in runs),
            'deepface': len(cfg['deepface_models']) * (len(cfg['conditions']['square_coverages']) + len(same)),
            'summary': 1}

    # ---- prepare: labels and split -----------------------------------------------------------
    split_path = os.path.join(out, 'prepare', 'split.csv')
    image_map = data.read_image_map(args.image_map, files) if args.image_map else None
    labels, skipped = data.read_labels(labels_path, files, image_map)
    plan['align'] = -(-int((data.make_split(labels, cfg, cfg['seed'])['role'] != 'unused').sum()) // ALIGN_CHUNK)
    units = common.Units(sum(plan[s] for s in stages))
    if 'prepare' in stages:
        def prepare():
            sp = data.make_split(labels, cfg, cfg['seed'])
            common.save_csv(split_path, sp)
            print(f'  labels: {len(labels)} images, {labels.identity.nunique()} identities, {skipped} lines skipped; '
                  f"roles: {sp['role'].value_counts().to_dict()}", flush=True)
        units.run('prepare split', split_path, prepare)
    split = pd.read_csv(split_path)
    needed = split[split.role != 'unused'].reset_index(drop=True)

    # ---- align: detect and align every needed image -------------------------------------------
    chunks = [needed.iloc[i:i + ALIGN_CHUNK] for i in range(0, len(needed), ALIGN_CHUNK)]
    chunk_paths = [os.path.join(out, 'prepare', f'aligned_{k:02d}.npz') for k in range(len(chunks))]
    if 'align' in stages:
        if args.smoke:
            aligner = data.ResizeAligner(cfg['crop_size'])
        else:
            aligner = data.Aligner(os.path.join(weights, cfg['align']['detector_file']),
                                   os.path.join(weights, 'face_recognition_sface_2021dec.onnx'),
                                   cfg['align']['score_threshold'], cfg['crop_size'])
        for k, (chunk, path) in enumerate(zip(chunks, chunk_paths)):
            def align(chunk=chunk, path=path):
                crops = np.zeros((len(chunk), cfg['crop_size'], cfg['crop_size'], 3), dtype=np.uint8)
                ok = np.zeros(len(chunk), dtype=bool)
                faces = np.full((len(chunk), 15), np.nan, dtype=np.float32)
                for i, f in enumerate(chunk['file']):
                    img = cv2.imread(os.path.join(images, f))
                    if img.shape[0] != 256 and not args.smoke:
                        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                    crop, face = aligner(img)
                    if crop is not None:
                        crops[i], ok[i] = crop, True
                        if face is not None:
                            faces[i] = face[:15]
                tmp = common.atomic_path(path)
                with open(tmp, 'wb') as fh:
                    np.savez(fh, files=chunk['file'].values, crops=crops, ok=ok, faces=faces)
                common.commit(tmp, path)
            units.run(f'align {k + 1}/{len(chunks)}', path, align)

    def load_aligned():
        parts = [np.load(p, allow_pickle=True) for p in chunk_paths]
        files_a = np.concatenate([p['files'] for p in parts])
        crops_a = np.concatenate([p['crops'] for p in parts])
        ok_a = np.concatenate([p['ok'] for p in parts])
        return dict(zip(files_a, range(len(files_a)))), crops_a, ok_a

    later = [s for s in stages if s not in ('prepare', 'align')]
    if not later:
        return finish(out, units, t_start, run_log)
    index, crops_all, ok_all = load_aligned()

    # Drop images whose face wasn't found; if a gallery image is lost, the identity's next image
    # takes its place, and identities left with fewer than two images leave the test set.
    usable = needed[[ok_all[index[f]] for f in needed['file']]].copy()
    test = usable[usable.role.isin(['gallery', 'probe'])].sort_values(['identity', 'index'])
    test = test.groupby('identity').filter(lambda g: len(g) >= 2)
    test['role'] = np.where(~test.duplicated('identity'), 'gallery', 'probe')
    train_df = usable[usable.role == 'train']
    run_log['counts'] = {'aligned': int(ok_all.sum()), 'not_aligned': int((~ok_all).sum()),
                         'test_identities': int(test.identity.nunique()),
                         'gallery': int((test.role == 'gallery').sum()), 'probes': int((test.role == 'probe').sum()),
                         'train_identities': int(train_df.identity.nunique()), 'train_images': len(train_df)}
    common.write_json(log_path, run_log)
    gal, prb = test[test.role == 'gallery'], test[test.role == 'probe']

    def crops_of(df):
        return crops_all[[index[f] for f in df['file']]]

    def frames_of(df):
        out_ = np.zeros((len(df), cfg['frame_size'], cfg['frame_size'], 3), dtype=np.uint8)
        for i, f in enumerate(df['file']):
            img = cv2.imread(os.path.join(images, f))[:, :, ::-1]
            if img.shape[0] != cfg['frame_size']:
                img = cv2.resize(img, (cfg['frame_size'],) * 2, interpolation=cv2.INTER_AREA)
            out_[i] = img
        return out_

    sets = {'crop': {'gallery': crops_of(gal), 'gallery_ids': gal['identity'].values,
                     'gallery_files': gal['file'].values, 'probes': crops_of(prb),
                     'probe_ids': prb['identity'].values, 'probe_files': prb['file'].values}}
    if any(s in stages for s in ('pixels', 'original')):
        sets['frame'] = dict(sets['crop'], gallery=frames_of(gal), probes=frames_of(prb))

    # ---- pretrained ---------------------------------------------------------------------------
    if 'pretrained' in stages:
        for k, spec in enumerate(cfg['pretrained']):
            if args.smoke:
                emb = nets.MockEmbedder(spec['dim'], cfg['crop_size'], seed=k)
            else:
                emb = nets.OnnxEmbedder(os.path.join(weights, spec['file']), spec, threads)
            evaluate.evaluate_model(spec['key'], emb, sets, conds, same, 'cosine', cfg, out, units)

    # ---- label check: stop before training if the labels look wrong ----------------------------
    # Pretrained recognizers identify unoccluded faces well; near-chance clean rank-1 for either
    # means the label file doesn't match the images (or a model's preprocessing is broken), and
    # every later result would look like a finding while measuring nothing.
    if any(s in stages for s in ('pixels', 'original', 'train', 'evaluate', 'deepface')):
        floor = cfg['label_check_min_rank1']
        ref = evaluate.reference_condition(conds)
        found = {spec['key']: evaluate.clean_rank1(out, spec['key'], ref) for spec in cfg['pretrained']}
        missing = [k for k, v in found.items() if v is None]
        low = {k: v for k, v in found.items() if v is not None and v < floor}
        if missing or low:
            msg = (f'Label check failed: run the pretrained stage first ({", ".join(missing)} missing).'
                   if missing else
                   'Label check failed: clean-probe rank-1 ' +
                   ', '.join(f'{k} {v:.3f}' for k, v in low.items()) +
                   f' is below {floor:.2f} (chance is {1 / len(gal):.4f}). The label file probably '
                   'does not match the images (wrong file, or numbering shifted), or a pretrained '
                   "model's input is wrong. Nothing was trained. Check the labels and start again "
                   'with a new --out.')
            common.write_json(os.path.join(out, 'HALTED.json'), {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                 'clean_rank1': found, 'floor': floor,
                                                                 'message': msg})
            print(msg, flush=True)
            sys.exit(2)
        run_log['label_check'] = {'clean_rank1': found, 'floor': floor, 'passed': True}
        common.write_json(log_path, run_log)
        print('  label check passed: clean rank-1 ' + ', '.join(f'{k} {v:.3f}' for k, v in found.items()),
              flush=True)

    # ---- pixels -------------------------------------------------------------------------------
    if 'pixels' in stages:
        def pixels(x):
            return x.reshape(len(x), -1).astype(np.float32) / 255.0

        def pixels_frame(x):
            small = cfg['frame_size'] // 2
            return np.stack([cv2.resize(f, (small, small), interpolation=cv2.INTER_AREA) for f in x]) \
                .reshape(len(x), -1).astype(np.float32) / 255.0
        evaluate.evaluate_model('pixels_crop', pixels, sets, conds, same, 'l2', cfg, out, units)
        evaluate.evaluate_model('pixels_frame', pixels_frame, sets, frame_conds, frame_conds, 'l2', cfg, out, units,
                                kind='frame')

    # ---- the original model on frames ---------------------------------------------------------
    if 'original' in stages:
        orig = nets.build_model(True, 128).to(device)
        if not args.smoke:
            orig.load_state_dict(torch.load(os.path.join(weights, cfg['original_model']['file']),
                                            map_location=device, weights_only=True))
        evaluate.evaluate_model('original_model', lambda x: nets.embed_frames_original(orig, x, device),
                                sets, frame_conds, frame_conds, 'l2', cfg, out, units, kind='frame')

    # ---- train --------------------------------------------------------------------------------
    run_dirs = {(r['name'], sd): os.path.join(out, 'train', f"{r['name']}_seed{sd}") for r, sd in runs}
    if 'train' in stages:
        ids = np.sort(train_df['identity'].unique())
        label_of = {v: i for i, v in enumerate(ids)}
        train_crops = torch.from_numpy(crops_of(train_df)).to(device)
        train_labels = torch.tensor([label_of[v] for v in train_df['identity']], device=device)
        for r, sd in runs:
            d = run_dirs[(r['name'], sd)]
            units.run(f"train {r['name']} seed {sd}", os.path.join(d, 'model.pt'),
                      lambda r=r, sd=sd, d=d: train_run(r, sd, train_crops, train_labels, cfg, device, d))
        del train_crops

    # ---- evaluate trained runs ----------------------------------------------------------------
    if 'evaluate' in stages:
        for r, sd in runs:
            key = f"{r['name']}_seed{sd}"
            ck = torch.load(os.path.join(run_dirs[(r['name'], sd)], 'model.pt'), map_location=device, weights_only=False)
            model = nets.build_model(r['attention'], cfg['train']['embedding_size']).to(device)
            model.load_state_dict(ck['model'])
            evaluate.evaluate_model(key, lambda x, model=model: nets.embed_torch(model, x, device),
                                    sets, conds, same, 'cosine', cfg, out, units)
            if r['attention']:
                final = os.path.join(out, 'eval', key, 'attention_check.csv')
                units.run(f'{key} attention check', final, lambda key=key, model=model, final=final:
                          evaluate.attention_check(key, model, sets, conds, cfg, device, final))

    # ---- optional: DeepFace's six models on the same crops ------------------------------------
    if 'deepface' in stages:
        first = prb[~prb.duplicated('identity')]
        small = {'crop': dict(sets['crop'], probes=crops_of(first), probe_ids=first['identity'].values,
                              probe_files=first['file'].values)}
        squares = {k: v for k, v in conds.items() if k.startswith('square_')}
        for name in cfg['deepface_models']:
            emb = nets.MockEmbedder(64, cfg['crop_size'], seed=99) if args.smoke else nets.DeepFaceEmbedder(name)
            evaluate.evaluate_model(f'deepface_{name}', emb, small, squares, same, 'cosine', cfg, out, units)

    # ---- summary ------------------------------------------------------------------------------
    if 'summary' in stages:
        final = os.path.join(out, 'summary', 'summary.csv')
        if os.path.exists(final):
            os.remove(final)  # always rebuilt from whatever units exist
        units.run('summary', final, lambda: common.save_csv(final, evaluate.summarize(out)))
    finish(out, units, t_start, run_log)


def finish(out, units, t_start, run_log):
    import common
    complete = units.ran + units.skipped == units.total
    run_log.setdefault('finishes', []).append({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'ran': units.ran,
                                               'skipped': units.skipped, 'planned': units.total,
                                               'seconds': round(time.time() - t_start, 1)})
    common.write_json(os.path.join(out, 'run_log.json'), run_log)
    if complete:
        common.write_json(os.path.join(out, 'DONE.json'), {
            'units': units.total, 'ran_this_start': units.ran, 'skipped_this_start': units.skipped,
            'wall_seconds_this_start': round(time.time() - t_start, 1),
            'environment': run_log.get('environment'), 'counts': run_log.get('counts'),
            'weights_sha256': run_log.get('weights_sha256')})
    print(f'{units.ran} units ran, {units.skipped} skipped, of {units.total}; '
          f'{time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
