"""
evaluate.py - identification and verification scores for one model under each condition.

Identity protocol: the gallery holds one clean image of each test identity; each probe is a
different image of one of them, with the condition drawn on it. Rank-1 is the share of probes
whose own identity scores highest.

Verification uses every probe-gallery pair, two ways:
- tar_at_far_<f>: the threshold is set on this condition's own impostor scores so that a share f
  of them pass; the true accept rate (TAR) at that threshold. This re-calibrates to each
  occluder, so it hides any shift in impostor scores.
- tar_fixed_far_<f> and far_fixed_far_<f>: the threshold is fixed once per model, on the clean
  identity condition (no occluder) for a target FAR f, and applied to every condition; the TAR and the
  actual false accept rate under the condition at that threshold.
The fixed thresholds are in eval/<model>/fixed_thresholds.json, and each unit also saves its
impostor-score quantiles (<unit>.impostors.npz) so other operating points can be computed later.

Same-image protocol (the original project's): each probe is the gallery image itself with the
condition drawn on it, so it measures image matching, not identification. It uses the same fixed
thresholds as the identity protocol.

Each (model, protocol, condition) is one unit with its own per-probe file.
"""

import os
import json

import numpy as np
import pandas as pd

from common import save_csv, save_npy, write_json, atomic_path, commit
from data import occlude, occlusion_mask

# Quantile levels saved for each unit's impostor scores: a uniform grid plus a dense upper tail.
IMPOSTOR_LEVELS = np.unique(np.concatenate([np.linspace(0, 1, 1001), 1 - np.logspace(-3, -6, 13)]))


def similarity(probes, gallery, metric):
    if metric == 'cosine':
        return probes @ gallery.T
    # negative squared L2 distance
    return -((probes ** 2).sum(1)[:, None] - 2 * probes @ gallery.T + (gallery ** 2).sum(1)[None, :])


def _impostors(s, own):
    rows = np.arange(len(s))
    genuine = s[rows, own]
    impostor = s.copy()
    impostor[rows, own] = -np.inf
    return genuine, impostor


def thresholds_at_far(probes, gallery, probe_ids, gallery_ids, metric, far_levels):
    """Score thresholds that let a share f of impostor pairs through, for each f."""
    s = similarity(probes.astype(np.float32), gallery.astype(np.float32), metric)
    col = {g: j for j, g in enumerate(gallery_ids)}
    _, impostor = _impostors(s, np.array([col[p] for p in probe_ids]))
    imp_flat = impostor[np.isfinite(impostor)]
    return {f'{far:g}': float(np.quantile(imp_flat, 1 - far)) for far in far_levels}


def score(probes, gallery, probe_ids, gallery_ids, metric, far_levels, fixed=None):
    s = similarity(probes.astype(np.float32), gallery.astype(np.float32), metric)
    col = {g: j for j, g in enumerate(gallery_ids)}
    own = np.array([col[p] for p in probe_ids])
    genuine, impostor = _impostors(s, own)
    rank = 1 + (s >= genuine[:, None]).sum(1) - 1  # ties count against the true identity
    max_imp = impostor.max(1)
    imp_flat = impostor[np.isfinite(impostor)]
    tar = {}
    for far in far_levels:
        thr = np.quantile(imp_flat, 1 - far)
        tar[f'tar_at_far_{far:g}'] = float((genuine > thr).mean())
        if fixed is not None:
            t = fixed[f'{far:g}']
            tar[f'tar_fixed_far_{far:g}'] = float((genuine > t).mean())
            tar[f'far_fixed_far_{far:g}'] = float((imp_flat > t).mean())
    quantiles = np.quantile(imp_flat, IMPOSTOR_LEVELS)
    return rank, genuine, max_imp, tar, quantiles, len(imp_flat)


def run_condition(model_key, protocol, cond_name, probes, gallery_emb, gallery_ids,
                  ids, files, metric, cfg, fixed, final):
    rank, genuine, max_imp, tar, quantiles, n_imp = score(probes, gallery_emb, ids, gallery_ids, metric,
                                                          cfg['far_levels'], fixed)
    imp_path = final[:-len('.csv.gz')] + '.impostors.npz'
    tmp = atomic_path(imp_path)
    with open(tmp, 'wb') as fh:
        np.savez(fh, levels=IMPOSTOR_LEVELS, values=quantiles, pairs=n_imp)
    commit(tmp, imp_path)
    df = pd.DataFrame({'model': model_key, 'protocol': protocol, 'condition': cond_name, 'probe': files,
                       'identity': ids, 'rank': rank, 'genuine': genuine.astype(np.float32),
                       'max_impostor': max_imp.astype(np.float32)})
    for k, v in tar.items():
        df[k] = v
    save_csv(final, df)


def reference_condition(conds):
    """The unoccluded condition (a square of coverage 0)."""
    for name, c in conds.items():
        if c.get('kind') == 'square' and c.get('coverage') == 0:
            return name
    raise ValueError('no unoccluded condition')


def evaluate_model(model_key, embed_fn, sets, conds, same_conds, metric, cfg, out_dir, units, kind='crop'):
    """Run every condition for one model. `sets` holds gallery/probe arrays for `kind` ('crop' or 'frame')."""
    g = sets[kind]
    d = os.path.join(out_dir, 'eval', model_key)
    gal_path = os.path.join(d, 'gallery_embeddings.npy')
    thr_path = os.path.join(d, 'fixed_thresholds.json')
    ref = reference_condition(conds)
    cache = {}

    def gallery():
        if 'g' not in cache:
            if os.path.exists(gal_path):
                cache['g'] = np.load(gal_path)
            else:
                cache['g'] = embed_fn(g['gallery'])
                save_npy(gal_path, cache['g'])
        return cache['g']

    def probes(cond_name, cond, crops, files):
        if cond_name == ref and crops is g['probes']:
            if 'clean' not in cache:
                cache['clean'] = embed_fn(crops)
            return cache['clean']
        return embed_fn(occlude(crops, cond, files))

    def fixed():
        """Thresholds set once on the clean identity condition, applied to every condition."""
        if 'thr' not in cache:
            if os.path.exists(thr_path):
                cache['thr'] = json.load(open(thr_path))['thresholds']
            else:
                cache['thr'] = thresholds_at_far(probes(ref, conds[ref], g['probes'], g['probe_files']),
                                                 gallery(), g['probe_ids'], g['gallery_ids'], metric,
                                                 cfg['far_levels'])
                write_json(thr_path, {'reference_condition': ref, 'protocol': 'identity', 'metric': metric,
                                      'thresholds': cache['thr']})
        return cache['thr']

    for name, cond in conds.items():
        final = os.path.join(d, f'identity__{name}.csv.gz')
        units.run(f'{model_key} identity {name}', final, lambda name=name, cond=cond, final=final: run_condition(
            model_key, 'identity', name, probes(name, cond, g['probes'], g['probe_files']), gallery(),
            g['gallery_ids'], g['probe_ids'], g['probe_files'], metric, cfg, fixed(), final))
    for name, cond in same_conds.items():
        final = os.path.join(d, f'same_image__{name}.csv.gz')
        units.run(f'{model_key} same_image {name}', final, lambda name=name, cond=cond, final=final: run_condition(
            model_key, 'same_image', name, probes(name, cond, g['gallery'], g['gallery_files']), gallery(),
            g['gallery_ids'], g['gallery_ids'], g['gallery_files'], metric, cfg, fixed(), final))


def clean_rank1(out_dir, model_key, ref='square_0.00'):
    """Rank-1 of a finished model on the clean identity condition, or None if not run yet."""
    path = os.path.join(out_dir, 'eval', model_key, f'identity__{ref}.csv.gz')
    if not os.path.exists(path):
        return None
    return float((pd.read_csv(path, usecols=['rank'])['rank'] == 1).mean())


def attention_check(model_key, model, sets, conds, cfg, device, final):
    """Mean attention inside and outside the occluded region, per condition, for the first probes."""
    from nets import attention_maps
    g = sets['crop']
    n = min(cfg['attention_probes'], len(g['probes']))
    crops, files = g['probes'][:n], g['probe_files'][:n]
    rows = []
    for name, cond in conds.items():
        maps = attention_maps(model, occlude(crops, cond, files), device)
        if maps is None:
            return
        h, w = crops.shape[1:3]
        inside, outside = [], []
        for i in range(n):
            m = occlusion_mask(cond, h, w, files[i])
            m = m.reshape(maps.shape[1], h // maps.shape[1], maps.shape[2], w // maps.shape[2]).mean((1, 3)) > 0.5
            if m.any():
                inside.append(maps[i][m].mean())
            outside.append(maps[i][~m].mean())
        rows.append({'model': model_key, 'condition': name, 'probes': n,
                     'attention_inside': float(np.mean(inside)) if inside else np.nan,
                     'attention_outside': float(np.mean(outside))})
    save_csv(final, pd.DataFrame(rows))


def summarize(out_dir):
    """One row per (model, protocol, condition): probes, rank-1, mean rank, and the TAR/FAR columns."""
    rows = []
    eval_dir = os.path.join(out_dir, 'eval')
    for model in sorted(os.listdir(eval_dir)):
        for f in sorted(os.listdir(os.path.join(eval_dir, model))):
            if not f.endswith('.csv.gz') or '.tmp' in f:
                continue
            df = pd.read_csv(os.path.join(eval_dir, model, f))
            r = {'model': model, 'protocol': df['protocol'].iloc[0], 'condition': df['condition'].iloc[0],
                 'probes': len(df), 'identities': df['identity'].nunique(),
                 'rank1': float((df['rank'] == 1).mean()), 'mean_rank': float(df['rank'].mean())}
            for c in df.columns:
                if c.startswith(('tar_at_far_', 'tar_fixed_far_', 'far_fixed_far_')):
                    r[c] = float(df[c].iloc[0])
            rows.append(r)
    return pd.DataFrame(rows)
