#!/usr/bin/env python
"""Aggregate the forecast arms' scores into the ladder and the joint-vs-single test.

`score_forecast.py` scores one arm against the sealed post-D test set and writes a JSON
holding, per tree and subset, the NLL/mutation of every rung plus the per-example losses
behind them. This reads a directory of those JSONs and reports:

  * the baseline ladder per tree, averaged over seeds;
  * joint_panel and joint_all against each tree's single-tree control, and against the
    strongest baseline, with the paired bootstrap from compare_runs.py (resample test
    examples jointly, since every arm scores the same file in the same order);
  * the same for the `novel` subset, which is the forecast that matters.

Seeds are averaged per example before bootstrapping, so the interval reflects
example-to-example variance with seed noise averaged down rather than treated as
independent evidence. Per-seed signs are reported alongside.

Usage:
    python experiments/multitree/forecast_summary.py <scores_dir> [--subset all novel]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

PANEL = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'denv1', 'denv2', 'denv3', 'denv4']
RUNGS = ['uniform', 'codon', 'where_only', 'substitution', 'where_times_which',
         'frequency']
SEEDS = [0, 1, 2]


def load(scores_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(scores_dir, '*.json'))):
        name = os.path.basename(path)[:-5]
        with open(path) as handle:
            out[name] = json.load(handle)
    return out


def cell(runs, arm, seed, subset, tree):
    rec = runs.get(f'{arm}_s{seed}')
    if not rec:
        return None
    return rec['results'].get(subset, {}).get(tree)


def mean_per_example(runs, arm, subset, tree, key):
    """Per-example (loss, n) averaged over the seeds that scored this tree."""
    got = []
    for seed in SEEDS:
        c = cell(runs, arm, seed, subset, tree)
        if c and c.get('per_example', {}).get(key):
            got.append(c['per_example'][key])
    if not got:
        return None
    n_ex = len(got[0]['loss'])
    if any(len(g['loss']) != n_ex for g in got):
        return None
    loss = [sum(g['loss'][i] for g in got) / len(got) for i in range(n_ex)]
    return {'loss': loss, 'n_mutations': got[0]['n_mutations'], 'n_seeds': len(got)}


def paired_interval(a, b, n_boot=10000, seed=0):
    """95% CI for (a - b) NLL per mutation, resampling examples jointly."""
    if not (a and b) or len(a['loss']) != len(b['loss']):
        return None
    al, bl, nn = a['loss'], b['loss'], a['n_mutations']
    n = len(al)
    point = (sum(al) - sum(bl)) / max(sum(nn), 1e-9)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(al[i] - bl[i] for i in idx) / max(sum(nn[i] for i in idx), 1e-9))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def pooled_interval(pairs, n_boot=10000, seed=0):
    """Pool trees: resample within each tree, sum numerators and denominators."""
    pairs = [(a, b) for a, b in pairs if a and b and len(a['loss']) == len(b['loss'])]
    if not pairs:
        return None
    num = sum(sum(a['loss']) - sum(b['loss']) for a, b in pairs)
    den = sum(sum(a['n_mutations']) for a, _ in pairs)
    point = num / max(den, 1e-9)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        tot_n = tot_d = 0.0
        for a, b in pairs:
            n = len(a['loss'])
            idx = [rng.randrange(n) for _ in range(n)]
            tot_n += sum(a['loss'][i] - b['loss'][i] for i in idx)
            tot_d += sum(a['n_mutations'][i] for i in idx)
        draws.append(tot_n / max(tot_d, 1e-9))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scores_dir')
    p.add_argument('--subset', nargs='+', default=['all', 'novel'])
    p.add_argument('--trees', nargs='+', default=PANEL)
    args = p.parse_args()
    runs = load(args.scores_dir)
    print(f"loaded {len(runs)} scored arms from {args.scores_dir}\n")

    for subset in args.subset:
        print(f"{'='*100}\n=== {subset.upper()} post-D substitutions\n{'='*100}")
        cols = ['uniform', 'codon', 'where_only', 'substitution',
                'where_times_which', 'frequency']
        hdr = (f"{'tree':9s} {'n':>5s} " + ' '.join(f'{c[:7]:>7s}' for c in cols)
               + f" | {'single':>7s} {'jpanel':>7s} {'jall':>7s}")
        print(hdr); print('-' * len(hdr))
        for tree in args.trees:
            ref = next((cell(runs, f'single_{tree}', s, subset, tree) for s in SEEDS
                        if cell(runs, f'single_{tree}', s, subset, tree)), None)
            if not ref:
                continue

            def avg(arm, key):
                vals = [cell(runs, arm, s, subset, tree)[key] for s in SEEDS
                        if cell(runs, arm, s, subset, tree)]
                return sum(vals) / len(vals) if vals else float('nan')

            base = ' '.join(f"{avg(f'single_{tree}', c):7.3f}" for c in cols)
            line = f"{tree:9s} {ref['n_mutations']:5.0f} {base} | "
            line += f"{avg(f'single_{tree}', 'model'):7.3f} "
            line += f"{avg('joint_panel', 'model'):7.3f} "
            line += f"{avg('joint_all', 'model'):7.3f}"
            print(line)

        # ---- paired tests
        for arm, label in (('joint_panel', 'joint_panel'), ('joint_all', 'joint_all')):
            for against, aname in (('single', 'its single-tree control'),
                                   ('where_times_which', 'where_times_which'),
                                   ('frequency', 'frequency')):
                pairs, rows = [], []
                for tree in args.trees:
                    a = mean_per_example(runs, arm, subset, tree, 'model')
                    if against == 'single':
                        b = mean_per_example(runs, f'single_{tree}', subset, tree,
                                             'model')
                    else:
                        b = mean_per_example(runs, f'single_{tree}', subset, tree,
                                             against)
                    if not (a and b):
                        continue
                    pairs.append((a, b))
                    rows.append((tree, paired_interval(a, b)))
                if not pairs:
                    continue
                print(f"\n{label} - {aname}  ({subset})")
                for tree, ci in rows:
                    if ci:
                        flag = 'better' if ci[2] < 0 else ('worse' if ci[1] > 0 else '')
                        print(f"  {tree:9s} {ci[0]:+7.3f}  [{ci[1]:+.3f}, {ci[2]:+.3f}]"
                              f"  {flag}")
                pl = pooled_interval(pairs)
                print(f"  {'POOLED':9s} {pl[0]:+7.3f}  [{pl[1]:+.3f}, {pl[2]:+.3f}]")
        print()


if __name__ == '__main__':
    main()
