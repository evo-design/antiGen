#!/usr/bin/env python
"""Emit the tidy CSVs behind the multi-tree paper figure.

Two evaluation regimes, written in one shape so the figure script treats them
identically:

  random    the random held-out split of each tree (decomp runs, joint5 + the five
            single-tree controls, 3 seeds). Not forecast-valid -- a val node's parent
            is usually in the training half -- which is exactly why the temporal arm
            exists.
  temporal  the D = 2025-01-01 holdout (forecast runs, joint_panel + nine single-tree
            controls, 3 seeds), scored on substitutions that did not exist when the
            models were trained.

Both use the **pooled-validation best epoch**, which is what --save-best ships and what
the temporal arm was scored at; applying it to the random arm too keeps the two columns
of the figure comparable. Note this is the conservative rule for the joint model: a
joint run must compromise across trees while a single-tree run cannot.

Outputs (to antiGen-figures/data/multitree/):
  ladder.csv  eval, rung, nll, n_mutations      -- pooled over trees, mean over seeds
  deltas.csv  eval, contrast, tree, est, lo, hi -- paired bootstrap, per tree + POOLED

Usage:  python experiments/multitree/figure_data.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random

DECOMP = '/large_storage/hielab/ispecht/multitree/decomp'
FCAST = '/large_storage/hielab/ispecht/multitree/forecast/scores'
OUT = 'antiGen-figures/data/multitree'
SEEDS = [0, 1, 2]
RANDOM_TREES = ['ncov', 'h3n2', 'h1n1pdm', 'rsv_a', 'rsv_b']
TEMPORAL_TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles',
                  'denv1', 'denv2', 'denv3', 'denv4']
RUNGS = ['uniform', 'codon', 'where_only', 'substitution', 'where_times_which',
         'frequency', 'single', 'joint']


# --------------------------------------------------------------- random split

def best_record(run_dir):
    """The record at the best *pooled* validation epoch (what --save-best keeps)."""
    path = sorted(glob.glob(os.path.join(run_dir, '*_val_metrics_by_epoch.json')))[-1]
    with open(path) as fh:
        recs = [r for r in json.load(fh) if r['epoch'] >= 0]
    return min(recs, key=lambda r: r['val']['overall']['nll_per_mutation'])


def random_split_data():
    ladder, pe = {}, {}
    for tree in RANDOM_TREES:
        for seed in SEEDS:
            j = best_record(f'{DECOMP}/joint5_s{seed}')
            s = best_record(f'{DECOMP}/single_{tree}_s{seed}')
            jt = j['val']['per_source'][tree]
            st = s['val']['per_source'][tree]
            b = j['baselines'][tree]
            n = jt['n_mutations']
            row = {
                'uniform': jt['uniform_nll_per_mutation'],
                'codon': b['codon']['nll_per_mutation'],
                'where_only': b['where_only']['nll_per_mutation'],
                'substitution': b['substitution']['nll_per_mutation'],
                'where_times_which': b['where_times_which']['nll_per_mutation'],
                'frequency': b['frequency']['nll_per_mutation'],
                'single': st['nll_per_mutation'],
                'joint': jt['nll_per_mutation'],
            }
            for k, v in row.items():
                a = ladder.setdefault(k, [0.0, 0.0])
                a[0] += v * n
                a[1] += n
            pe.setdefault((tree, 'joint'), []).append(jt['per_example'])
            pe.setdefault((tree, 'single'), []).append(st['per_example'])
            pe.setdefault((tree, 'where_times_which'), []).append(
                b['where_times_which']['per_example'])
    return {k: (v[0] / v[1], v[1] / len(SEEDS)) for k, v in ladder.items()}, pe


# ------------------------------------------------------------ temporal split

def temporal_data(subset='all'):
    runs = {}
    for p in glob.glob(f'{FCAST}/*.json'):
        with open(p) as fh:
            runs[os.path.basename(p)[:-5]] = json.load(fh)

    def cell(arm, seed, tree):
        r = runs.get(f'{arm}_s{seed}')
        return r['results'].get(subset, {}).get(tree) if r else None

    ladder, pe = {}, {}
    for tree in TEMPORAL_TREES:
        for seed in SEEDS:
            j = cell('joint_panel', seed, tree)
            s = cell(f'single_{tree}', seed, tree)
            if not (j and s):
                continue
            n = j['n_mutations']
            row = {k: j[k] for k in ('uniform', 'codon', 'where_only', 'substitution',
                                     'where_times_which', 'frequency')}
            row['single'] = s['model']
            row['joint'] = j['model']
            for k, v in row.items():
                a = ladder.setdefault(k, [0.0, 0.0])
                a[0] += v * n
                a[1] += n
            pe.setdefault((tree, 'joint'), []).append(j['per_example']['model'])
            pe.setdefault((tree, 'single'), []).append(s['per_example']['model'])
            pe.setdefault((tree, 'where_times_which'), []).append(
                j['per_example']['where_times_which'])
    return {k: (v[0] / v[1], v[1] / len(SEEDS)) for k, v in ladder.items()}, pe


# ------------------------------------------------------------------ bootstrap

def mean_over_seeds(reps):
    n = len(reps[0]['loss'])
    if any(len(r['loss']) != n for r in reps):
        return None
    return {'loss': [sum(r['loss'][i] for r in reps) / len(reps) for i in range(n)],
            'n_mutations': reps[0]['n_mutations']}


def paired(a, b, nb=10000, sd=0):
    al, bl, nn = a['loss'], b['loss'], a['n_mutations']
    n = len(al)
    pt = (sum(al) - sum(bl)) / max(sum(nn), 1e-9)
    rng = random.Random(sd)
    d = []
    for _ in range(nb):
        i = [rng.randrange(n) for _ in range(n)]
        d.append(sum(al[k] - bl[k] for k in i) / max(sum(nn[k] for k in i), 1e-9))
    d.sort()
    return pt, d[int(.025 * nb)], d[int(.975 * nb)]


def pooled(pairs, nb=10000, sd=0):
    num = sum(sum(a['loss']) - sum(b['loss']) for a, b in pairs)
    den = sum(sum(a['n_mutations']) for a, _ in pairs)
    rng = random.Random(sd)
    d = []
    for _ in range(nb):
        tn = td = 0.0
        for a, b in pairs:
            n = len(a['loss'])
            i = [rng.randrange(n) for _ in range(n)]
            tn += sum(a['loss'][k] - b['loss'][k] for k in i)
            td += sum(a['n_mutations'][k] for k in i)
        d.append(tn / max(td, 1e-9))
    d.sort()
    return num / den, d[int(.025 * nb)], d[int(.975 * nb)]


def deltas(pe, trees, label):
    out = []
    for contrast, other in (('vs single-tree model', 'single'),
                            ('vs where x which', 'where_times_which')):
        pairs = []
        for tree in trees:
            a = pe.get((tree, 'joint'))
            b = pe.get((tree, other))
            if not (a and b):
                continue
            a, b = mean_over_seeds(a), mean_over_seeds(b)
            if not (a and b) or len(a['loss']) != len(b['loss']):
                continue
            pairs.append((a, b))
            est, lo, hi = paired(a, b)
            out.append(dict(eval=label, contrast=contrast, tree=tree,
                            est=est, lo=lo, hi=hi, pooled=0))
        est, lo, hi = pooled(pairs)
        out.append(dict(eval=label, contrast=contrast, tree='POOLED',
                        est=est, lo=lo, hi=hi, pooled=1))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rl, rpe = random_split_data()
    tl, tpe = temporal_data('all')

    with open(f'{OUT}/ladder.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval', 'rung', 'nll', 'n_mutations'])
        for label, lad in (('Random holdout', rl), ('Temporal holdout', tl)):
            for rung in RUNGS:
                if rung in lad:
                    w.writerow([label, rung, f'{lad[rung][0]:.6f}',
                                f'{lad[rung][1]:.1f}'])

    rows = (deltas(rpe, RANDOM_TREES, 'Random holdout')
            + deltas(tpe, TEMPORAL_TREES, 'Temporal holdout'))
    with open(f'{OUT}/deltas.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['eval', 'contrast', 'tree', 'est',
                                           'lo', 'hi', 'pooled'])
        w.writeheader()
        for r in rows:
            for k in ('est', 'lo', 'hi'):
                r[k] = f'{r[k]:.6f}'
            w.writerow(r)
    print(f"wrote {OUT}/ladder.csv and {OUT}/deltas.csv")
    for label, lad in (('Random holdout', rl), ('Temporal holdout', tl)):
        print(f"\n{label}: " + '  '.join(
            f'{k}={lad[k][0]:.3f}' for k in RUNGS if k in lad))


if __name__ == '__main__':
    main()
