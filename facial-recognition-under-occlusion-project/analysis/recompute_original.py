#!/usr/bin/env python3
"""
recompute_original.py - recompute the report's summary numbers from the original result files
and check every table in the original report (PROJECT_REPORT.txt, dated December 4, 2025)
against them.

Run from the repository root (pandas and numpy only, a few seconds):

    python analysis/recompute_original.py [--report PROJECT_REPORT.txt]

Writes to analysis/out/:
    summary_original.csv       per run and model: area under the accuracy curve (trapezoid over
                               occlusion 0 to 1), the first level below 50%, the interpolated 50%
                               crossing, and the mean accuracy from 10% to 100% occlusion against
                               chance (1/N)
    report_table_check.csv     every cell of report Tables 1 to 5 next to the matching result file;
                               written only when the report's text is available (--report, or
                               PROJECT_REPORT.txt at the repository root)
"""

import os
import re
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results', 'original')
OUT = os.path.join(ROOT, 'analysis', 'out')
REPORT = os.path.join(ROOT, 'PROJECT_REPORT.txt')
MODELS = ['Custom', 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'ArcFace', 'SFace']

RUNS = {
    'N50': ('exper__size_50__20251201_192436', 50),
    'N100': ('exper__size_100__20251130_235656', 100),
    'N200': ('exper__size_200__20251201_203249', 200),
    'N500': ('exper__size_500__20251201_004627', 500),
    'N100_1pct': ('exper__size_100__20251130_194129', 100),
    'NOBG50': ('exper__NO_BG__size_50__20251203_001017', 50),
    'NOBG100': ('exper__NO_BG__size_100__20251203_012434', 100),
    'NOBG200': ('exper__NO_BG__size_200__20251203_022706', 200),
}


def load(name):
    df = pd.read_csv(os.path.join(RESULTS, name + '.csv'))
    df['Occlusion'] = df['Occlusion'].round(2)
    return df.set_index('Occlusion')


def crossing(x, y, level=0.5):
    """First level where accuracy is below `level`, and the linear-interpolated crossing."""
    below = np.flatnonzero(y < level)
    if len(below) == 0:
        return np.nan, np.nan
    j = below[0]
    if j == 0:
        return x[0], x[0]
    return x[j], x[j - 1] + (y[j - 1] - level) / (y[j - 1] - y[j]) * (x[j] - x[j - 1])


def summary(runs):
    rows = []
    for key, (name, n) in RUNS.items():
        df = runs[key]
        x = df.index.values
        for m in MODELS:
            y = df[m].values
            first, interp = crossing(x, y)
            tail = df.loc[0.10:1.0, m].values
            rows.append({'run': key, 'file': name, 'N': n, 'model': m,
                         'auc': np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x),
                         'first_level_below_50pct': first, 'crossing_50pct': interp,
                         'mean_acc_10_to_100pct': tail.mean(), 'chance': 1.0 / n,
                         'acc_10pct': df.loc[0.10, m], 'acc_20pct': df.loc[0.20, m],
                         'acc_30pct': df.loc[0.30, m], 'acc_50pct': df.loc[0.50, m]})
    return pd.DataFrame(rows)


def report_tables(runs, report):
    """Parse report Tables 1 to 5 and pair each cell with the result file it should come from."""
    lines = open(report, encoding='utf-8').read().splitlines()

    def table(title, ncols):
        start = next(i for i, l in enumerate(lines) if l.startswith(title))
        rows = []
        for l in lines[start:start + 40]:
            m = re.match(r'^(\d+)%\s*\|(.*)$', l)
            if m:
                vals = [float(v) for v in re.findall(r'-?\d+\.\d+|-?\d+', m.group(2))][:ncols]
                rows.append((round(int(m.group(1)) / 100, 2), vals))
            elif rows and not l.strip():
                break
        return rows

    out = []

    def add(tab, occ, col, run_key, model, reported):
        if run_key is None:
            out.append({'table': tab, 'occlusion': occ, 'column': col, 'file': '', 'reported_pct': reported,
                        'file_pct': np.nan, 'match': False})
            return
        value = runs[run_key].loc[occ, model] * 100
        out.append({'table': tab, 'occlusion': occ, 'column': col, 'file': RUNS[run_key][0],
                    'reported_pct': reported, 'file_pct': round(value, 1), 'match': abs(value - reported) < 0.05})

    for tab, key in [('Table 1', 'N100'), ('Table 2', 'N200')]:
        for occ, vals in table(tab + ':', 7):
            for m, v in zip(MODELS, vals):
                add(tab, occ, m, key, m, v)
    for occ, (std, nobg, _) in table('Table 3:', 3):
        add('Table 3', occ, 'Custom standard', 'N100', 'Custom', std)
        add('Table 3', occ, 'Custom no background', 'NOBG100', 'Custom', nobg)
    for occ, (std, nobg, _) in table('Table 4:', 3):
        add('Table 4', occ, 'VGG-Face standard', 'N100', 'VGG-Face', std)
        add('Table 4', occ, 'VGG-Face no background', 'NOBG100', 'VGG-Face', nobg)
    for occ, vals in table('Table 5:', 5):
        for key, v in zip(['N50', 'N100', 'N200', 'N500', None], vals):
            add('Table 5', occ, f'Custom N={RUNS[key][1]}' if key else 'Custom N=1000 (no result file)', key, 'Custom', v)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--report', default=REPORT, help="the original report's text (for the table check)")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runs = {k: load(name) for k, (name, _) in RUNS.items()}
    s = summary(runs)
    s.round(4).to_csv(os.path.join(OUT, 'summary_original.csv'), index=False)
    if os.path.exists(a.report):
        t = report_tables(runs, a.report)
        t.to_csv(os.path.join(OUT, 'report_table_check.csv'), index=False)
        print('Report tables against the result files (cells matching / cells):')
        for tab, g in t.groupby('table'):
            print(f'  {tab}: {int(g["match"].sum())} / {len(g)}')
            bad = g[~g['match']]
            for _, r in bad.iterrows():
                print(f'    {r.occlusion:.2f} {r.column}: report {r.reported_pct:g}, file {r.file_pct}')
    else:
        print(f'{a.report} not found: the table check is skipped and report_table_check.csv is left as it is.')
    print('\nArea under the accuracy curve, occlusion 0 to 1 (report: Custom 0.652 at N = 100, 0.583 at N = 200):')
    print(s.pivot(index='model', columns='run', values='auc').round(3).loc[MODELS].to_string())
    print('\nInterpolated 50% crossing (report: Custom ~38% at N = 100, ~35% at N = 200; all others <5%):')
    print(s.pivot(index='model', columns='run', values='crossing_50pct').round(3).loc[MODELS].to_string())
    print('\nPretrained models, mean accuracy from 10% to 100% occlusion against chance (1/N):')
    sota = s[s.model != 'Custom'].groupby(['run', 'N'])['mean_acc_10_to_100pct'].mean().reset_index()
    for _, r in sota.iterrows():
        print(f'  {r.run}: {r.mean_acc_10_to_100pct * 100:.2f}% (chance {100 / r.N:.2f}%)')


if __name__ == '__main__':
    main()
