#!/usr/bin/env python3
"""
analyze.py - intervals, seed spread, and the three primary contrasts from the study's per-probe
results (CPU, a few minutes).

    python study/analyze.py --results DIR [--resamples 2000]

Reads DIR/eval/<model>/*.csv.gz (and each model's fixed_thresholds.json) and writes DIR/analysis/.

Resampling. Each bootstrap resample draws test identities with replacement (a cluster bootstrap:
a probe's identity shares its gallery image) and, for each trained configuration, draws its
training seeds with replacement; the configuration's value is the mean over the drawn seeds of
each seed's value on the drawn identities. So intervals carry both identity and training-seed
variance. Every model, protocol, and condition uses the same identity draw within a resample, and
a configuration uses the same seed draw across conditions, which keeps contrasts paired. With three
seeds the seed part of the resampling is coarse; no interval here can support a null result.

Metrics: rank-1; TAR at the fixed thresholds (set per model on clean probes at FAR 1% and 0.1%).

Primary contrasts (identity protocol; Holm correction over the three;
everything else in the output is descriptive):
  P1  occlusion augmentation, main effect (the mean of its effect with and without attention), on
      mean rank-1 over the trained-on occluder: centered black squares at 10 to 50%
  P2  occlusion augmentation, main effect, on mean rank-1 over occluders never seen in training:
      the eye band, the lower band, the off-center square, and the gray and noise fills
  P3  the attention layer, main effect (the mean of its effect with and without augmentation), on
      mean rank-1 over all occluded conditions of P1 and P2
A primary effect is claimed only if its Holm-adjusted bootstrap p-value is below 0.05; the
98.3% (Bonferroni) interval is reported beside the 95% one.

Bootstrap p-values are two-sided with the add-one correction, 2 (1 + k) / (B + 1), where k counts
the resamples on the far side of zero (the smaller tail) and B is the number of resamples. With
no resample past zero the value is its floor, 2 / (B + 1), the smallest value B resamples can
give (0.0009995 with 2,000); the `*_text` columns then quote it as a bound ("< 0.001").

Outputs in DIR/analysis/:
  intervals.csv            estimate and 95% interval per model (configuration), protocol,
                           condition, and metric
  seed_spread.csv          per-seed values, their minimum, maximum, and range
  contrasts_primary.csv    P1 to P3: estimate, 95% and 98.3% intervals, bootstrap p, Holm p,
                           the range over every one-seed-per-configuration combination, and the
                           claim decision
  main_effects_by_condition.csv  the augmentation and attention main effects at each condition,
                           the terms P1 to P3 average over (descriptive, unadjusted)
  contrasts_by_condition.csv  the five simple contrasts per condition (descriptive, unadjusted)
  seed_pairs.csv           each simple contrast's nine seed-pair differences per condition
  protocol_gap.csv         same-image minus identity rank-1 per model and square, with intervals
Percentile intervals have zero width where every resample gives the same value (for example
100%).
"""

import os
import json
import math
import itertools
import argparse

import numpy as np
import pandas as pd

SQUARES_TRAINED = ['square_0.10', 'square_0.15', 'square_0.20', 'square_0.25', 'square_0.30',
                   'square_0.40', 'square_0.50']
UNSEEN = ['eyes_band', 'lower_band', 'random_square_0.20', 'gray_square_0.20', 'noise_square_0.20',
          'gray_square_0.40', 'noise_square_0.40']
SIMPLE = [  # (label, configuration A, configuration B): A minus B
    ('attention, with augmentation', 'cosface_att_aug', 'cosface_noatt_aug'),
    ('attention, without augmentation', 'cosface_att_noaug', 'cosface_noatt_noaug'),
    ('augmentation, with attention', 'cosface_att_aug', 'cosface_att_noaug'),
    ('augmentation, without attention', 'cosface_noatt_aug', 'cosface_noatt_noaug'),
    ('CosFace minus batch-hard triplet', 'cosface_att_aug', 'triplet_att_aug'),
]
AUG_MAIN = [('cosface_att_aug', 'cosface_att_noaug'), ('cosface_noatt_aug', 'cosface_noatt_noaug')]
ATT_MAIN = [('cosface_att_aug', 'cosface_noatt_aug'), ('cosface_att_noaug', 'cosface_noatt_noaug')]
PRIMARY = [
    ('P1', 'augmentation main effect, trained-on squares 10 to 50%', AUG_MAIN, SQUARES_TRAINED),
    ('P2', 'augmentation main effect, unseen occluders', AUG_MAIN, UNSEEN),
    ('P3', 'attention main effect, the 14 occluded conditions of P1 and P2', ATT_MAIN,
     SQUARES_TRAINED + UNSEEN),
]


def load(results):
    frames = []
    eval_dir = os.path.join(results, 'eval')
    for model in sorted(os.listdir(eval_dir)):
        thr_path = os.path.join(eval_dir, model, 'fixed_thresholds.json')
        thr = json.load(open(thr_path))['thresholds'] if os.path.exists(thr_path) else {}
        for f in sorted(os.listdir(os.path.join(eval_dir, model))):
            if not f.endswith('.csv.gz') or '.tmp' in f:
                continue
            # evaluate.py compared the float32 scores in memory with each threshold, and the per-probe
            # files store those scores at float32 precision. Read them back as float32, so that a score
            # equal to the threshold is rejected here as it was there (it matters for runs whose scores
            # bunch up, such as the failed triplet runs).
            df = pd.read_csv(os.path.join(eval_dir, model, f),
                             usecols=['model', 'protocol', 'condition', 'identity', 'rank', 'genuine'],
                             dtype={'genuine': np.float32})
            df['rank1'] = (df['rank'] == 1).astype(float)
            genuine = df['genuine'].to_numpy(dtype=np.float64)
            for far, t in thr.items():
                df[f'tar_fixed_{far}'] = (genuine > t).astype(float)
            frames.append(df.drop(columns=['rank', 'genuine']))
    if not frames:
        raise SystemExit(f'no per-probe results (*.csv.gz) under {eval_dir}; analyze.py needs the '
                         '--out folder of a full run_study.py run, not the aggregates in results/study')
    df = pd.concat(frames, ignore_index=True)
    df['config'] = df['model'].str.replace(r'_seed\d+$', '', regex=True)
    df['seed'] = df['model'].str.extract(r'_seed(\d+)$')[0].fillna('-')
    return df


def p_two_sided(d):
    """Add-one two-sided bootstrap p-value, and k, the resamples on the far side of zero."""
    k = int(min((d <= 0).sum(), (d >= 0).sum()))
    return float(min(1.0, 2 * (1 + k) / (len(d) + 1))), k


def p_text(p, is_bound):
    """A p-value as it should be quoted: the value, or an upper bound when it sits at its floor."""
    if not is_bound:
        return f'{p:.2g}' if p < 0.1 else f'{p:.2f}'
    step = 10 ** math.floor(math.log10(p))
    bound = math.ceil(round(p / step, 9)) * step  # rounded up to one significant figure
    return f'< {bound:g}' if bound > p else f'<= {bound:g}'


def holm(ps, bounds):
    """Holm-adjusted p-values; an adjusted value is a bound when the term that sets it is one."""
    order = np.argsort(ps, kind='stable')
    adj, adj_bound = np.empty(len(ps)), np.zeros(len(ps), dtype=bool)
    running, running_bound = 0.0, False
    for k, i in enumerate(order):
        v = min(1.0, (len(ps) - k) * ps[i])
        if v > running or (v == running and not bounds[i]):
            running, running_bound = v, bool(bounds[i]) and v < 1.0
        adj[i], adj_bound[i] = running, running_bound
    return adj, adj_bound


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--results', required=True)
    p.add_argument('--resamples', type=int, default=2000)
    a = p.parse_args()
    out = os.path.join(a.results, 'analysis')
    os.makedirs(out, exist_ok=True)
    df = load(a.results)
    metrics = ['rank1'] + sorted(c for c in df.columns if c.startswith('tar_fixed_'))
    ids = np.sort(df['identity'].unique())
    id_pos = {v: i for i, v in enumerate(ids)}
    df['ipos'] = df['identity'].map(id_pos)
    rng = np.random.default_rng(0)
    B = a.resamples
    id_draw = rng.integers(0, len(ids), size=(B, len(ids)))
    seeds_of = {c: sorted(g['seed'].unique()) for c, g in df.groupby('config')}
    seed_draw = {c: rng.integers(0, len(s), size=(B, len(s))) for c, s in seeds_of.items()}

    boot, point, spread_rows, rows, per_seed_value = {}, {}, [], [], {}
    for (config, protocol, cond), g in df.groupby(['config', 'protocol', 'condition']):
        seeds = seeds_of[config]
        for m in metrics:
            if g[m].isna().all():
                continue
            hits = np.zeros((len(seeds), len(ids)))
            cnt = np.zeros((len(seeds), len(ids)))
            for si, sd in enumerate(seeds):
                gs = g[g['seed'] == sd]
                np.add.at(hits[si], gs['ipos'].values, gs[m].values)
                np.add.at(cnt[si], gs['ipos'].values, 1)
            per_seed = hits.sum(1) / np.maximum(cnt.sum(1), 1)
            rates = hits[:, id_draw].sum(2) / np.maximum(cnt[:, id_draw].sum(2), 1)  # (seeds, B)
            stat = rates[seed_draw[config].T, np.arange(B)].mean(0)
            key = (config, protocol, cond, m)
            boot[key], point[key] = stat, per_seed.mean()
            for sd, v in zip(seeds, per_seed):
                per_seed_value[(config, protocol, cond, m, sd)] = v
            lo, hi = np.percentile(stat, [2.5, 97.5])
            rows.append({'model': config, 'protocol': protocol, 'condition': cond, 'metric': m,
                         'estimate': per_seed.mean(), 'ci_low': lo, 'ci_high': hi, 'seeds': len(seeds),
                         'identities': int((cnt.sum(0) > 0).sum()), 'probes_per_seed': cnt.sum(1).mean()})
            if len(seeds) > 1:
                spread_rows.append({'model': config, 'protocol': protocol, 'condition': cond, 'metric': m,
                                    **{f'seed_{sd}': v for sd, v in zip(seeds, per_seed)},
                                    'min': per_seed.min(), 'max': per_seed.max(),
                                    'range': per_seed.max() - per_seed.min(), 'sd': per_seed.std(ddof=1)})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'intervals.csv'), index=False)
    pd.DataFrame(spread_rows).to_csv(os.path.join(out, 'seed_spread.csv'), index=False)

    # descriptive per-condition contrasts and their seed pairs (identity and same-image protocols)
    by_cond, pairs = [], []
    conds = sorted(df['condition'].unique())
    for label, x, y in SIMPLE:
        for protocol in ['identity', 'same_image']:
            for cond in conds:
                kx, ky = (x, protocol, cond, 'rank1'), (y, protocol, cond, 'rank1')
                if kx not in boot or ky not in boot:
                    continue
                d = boot[kx] - boot[ky]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p, k = p_two_sided(d)
                by_cond.append({'contrast': label, 'protocol': protocol, 'condition': cond,
                                'difference': point[kx] - point[ky], 'ci_low': lo, 'ci_high': hi,
                                'p_boot_unadjusted': p, 'resamples_past_zero': k,
                                'p_boot_unadjusted_text': p_text(p, k == 0)})
                for sx, sy in itertools.product(seeds_of[x], seeds_of[y]):
                    vx = per_seed_value.get((x, protocol, cond, 'rank1', sx))
                    vy = per_seed_value.get((y, protocol, cond, 'rank1', sy))
                    if vx is not None and vy is not None:
                        pairs.append({'contrast': label, 'protocol': protocol, 'condition': cond,
                                      'seed_a': sx, 'seed_b': sy, 'difference': vx - vy})
    pd.DataFrame(by_cond).to_csv(os.path.join(out, 'contrasts_by_condition.csv'), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(out, 'seed_pairs.csv'), index=False)

    # the main effects at each identity condition: the terms the primary contrasts average over
    mains = []
    for label, pairs_ in [('augmentation', AUG_MAIN), ('attention', ATT_MAIN)]:
        for cond in conds:
            keys = [((xa, 'identity', cond, 'rank1'), (xb, 'identity', cond, 'rank1')) for xa, xb in pairs_]
            if not all(kx in boot and ky in boot for kx, ky in keys):
                continue
            d = np.mean([boot[kx] - boot[ky] for kx, ky in keys], axis=0)
            p, k = p_two_sided(d)
            mains.append({'main_effect': label, 'condition': cond,
                          'in_P1': label == 'augmentation' and cond in SQUARES_TRAINED,
                          'in_P2': label == 'augmentation' and cond in UNSEEN,
                          'in_P3': label == 'attention' and cond in SQUARES_TRAINED + UNSEEN,
                          'estimate': np.mean([point[kx] - point[ky] for kx, ky in keys]),
                          'ci_low': np.percentile(d, 2.5), 'ci_high': np.percentile(d, 97.5),
                          'p_boot_unadjusted': p, 'resamples_past_zero': k,
                          'p_boot_unadjusted_text': p_text(p, k == 0)})
    pd.DataFrame(mains).to_csv(os.path.join(out, 'main_effects_by_condition.csv'), index=False)

    # primary contrasts
    prim = []
    for code, label, pairs_, cset in PRIMARY:
        keys_ok = all((c, 'identity', cond, 'rank1') in boot for pr in pairs_ for c in pr for cond in cset)
        if not keys_ok:
            prim.append({'contrast': code, 'definition': label, 'note': 'missing conditions or runs'})
            continue

        def group(config, values):
            return np.mean([values[(config, 'identity', cond, 'rank1')] for cond in cset], axis=0)
        d = np.mean([group(xa, boot) - group(xb, boot) for xa, xb in pairs_], axis=0)
        est = np.mean([group(xa, point) - group(xb, point) for xa, xb in pairs_])
        configs = sorted({c for pr in pairs_ for c in pr})
        combos = []
        for choice in itertools.product(*[seeds_of[c] for c in configs]):
            pick = dict(zip(configs, choice))
            combos.append(np.mean([np.mean([per_seed_value[(xa, 'identity', cond, 'rank1', pick[xa])] -
                                            per_seed_value[(xb, 'identity', cond, 'rank1', pick[xb])]
                                            for cond in cset]) for xa, xb in pairs_]))
        p, k = p_two_sided(d)
        prim.append({'contrast': code, 'definition': label, 'estimate': est,
                     'ci95_low': np.percentile(d, 2.5), 'ci95_high': np.percentile(d, 97.5),
                     'ci983_low': np.percentile(d, 2.5 / 3), 'ci983_high': np.percentile(d, 100 - 2.5 / 3),
                     'p_boot': p, 'resamples_past_zero': k, 'p_boot_text': p_text(p, k == 0),
                     'seed_combo_min': min(combos), 'seed_combo_max': max(combos),
                     'seed_combos': len(combos), 'resamples': B})
    prim_df = pd.DataFrame(prim)
    if 'p_boot' in prim_df and prim_df['p_boot'].notna().all():
        adj, adj_bound = holm(prim_df['p_boot'].values, (prim_df['resamples_past_zero'] == 0).values)
        prim_df['p_holm'] = adj
        prim_df['p_holm_text'] = [p_text(p, b) for p, b in zip(adj, adj_bound)]
        prim_df['claimed'] = prim_df['p_holm'] < 0.05
    prim_df.to_csv(os.path.join(out, 'contrasts_primary.csv'), index=False)

    # protocol gap, paired by identity draw
    gaps = []
    for (config, protocol, cond, m) in list(boot):
        if protocol != 'same_image' or m != 'rank1':
            continue
        ki = (config, 'identity', cond, 'rank1')
        if ki in boot:
            d = boot[(config, protocol, cond, m)] - boot[ki]
            gaps.append({'model': config, 'condition': cond,
                         'same_image_rank1': point[(config, protocol, cond, m)], 'identity_rank1': point[ki],
                         'gap': point[(config, protocol, cond, m)] - point[ki],
                         'ci_low': np.percentile(d, 2.5), 'ci_high': np.percentile(d, 97.5)})
    pd.DataFrame(gaps).to_csv(os.path.join(out, 'protocol_gap.csv'), index=False)
    print(f'{len(rows)} intervals, {len(spread_rows)} seed spreads, {len(by_cond)} descriptive contrasts, '
          f'{len(mains)} per-condition main effects, {len(prim)} primary contrasts, {len(gaps)} protocol gaps '
          f'-> {out}')
    if len(prim_df):
        print(prim_df.to_string(index=False))


if __name__ == '__main__':
    main()
