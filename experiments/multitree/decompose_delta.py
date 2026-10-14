#!/usr/bin/env python
"""Split a joint-vs-single NLL difference into a "where" part and a "which" part.

The model emits one distribution over the whole L x E map, so for an observed mutation
at (i, a) the NLL factors exactly:

    -log P(i, a)  =  -log P(i)        +   -log P(a | i)
                     "where"              "which"
                     did it pick the      given the site, did it pick
                     right site           the right residue

Both terms come from the same dumped per-cell map (`--dump-cells`), with
P(i) = sum_a P(i, a). The split is exact, so the joint-minus-single delta decomposes
additively and the two components add back to the delta reported by sweep_summary.

Scored at each run's best *pooled* validation epoch -- that is the epoch `--dump-cells`
writes, which corresponds to sweep_summary's `--epoch-rule overall`, the conservative one.

Usage:
    python experiments/multitree/decompose_delta.py <sweep_dir> \
        [--arm joint5] [--trees ncov h3n2 ...] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_index import val_examples                                   # noqa: E402

TREES = ['ncov', 'h3n2', 'h1n1pdm', 'rsv_a', 'rsv_b']


def load_cells(run_dir, tree):
    matches = sorted(glob.glob(os.path.join(run_dir, "*_val_cells.npz")))
    if not matches:
        return None
    with np.load(matches[-1]) as handle:
        if tree not in handle.files:
            return None
        return handle[tree].astype(np.float64)


def terms(cells, examples):
    """Per-mutation (where, which, total) NLL arrays, one row per observed mutation."""
    where, which, weight = [], [], []
    for idx, example in enumerate(examples):
        mat = example.mat.numpy()
        nz = np.argwhere(mat > 0)
        if not len(nz):
            continue
        row = cells[idx]
        for pos, alt in nz:
            p_pos = row[pos].sum()
            p_joint = row[pos, alt]
            if p_pos <= 0 or p_joint <= 0:
                continue
            where.append(-np.log(p_pos))
            which.append(-np.log(p_joint / p_pos))
            weight.append(mat[pos, alt])
    return np.array(where), np.array(which), np.array(weight)


def paired_bootstrap(diffs, weights, n=2000, seed=0):
    """Mean weighted difference and its 95% interval, resampling mutations."""
    rng = random.Random(seed)
    k = len(diffs)
    if k == 0:
        return float('nan'), float('nan'), float('nan')
    point = float(np.sum(diffs * weights) / np.sum(weights))
    draws = []
    idx = np.arange(k)
    for _ in range(n):
        pick = np.array([rng.randrange(k) for _ in range(k)])
        draws.append(np.sum(diffs[pick] * weights[pick]) / np.sum(weights[pick]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sweep_dir')
    parser.add_argument('--arm', default='joint5')
    parser.add_argument('--trees', nargs='+', default=TREES)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()

    print(f"arm={args.arm} vs single-tree controls, scored at each run's best pooled epoch")
    print(f"{'tree':9s} {'seed':>4s} {'n_mut':>6s} | {'dWhere':>8s} {'dWhich':>8s} "
          f"{'dTotal':>8s} | {'where%':>7s}")
    print('-' * 68)

    pooled = {'where': [], 'which': [], 'weight': []}
    per_tree = {}
    for tree in args.trees:
        rows = {'where': [], 'which': [], 'weight': []}
        for seed in args.seeds:
            jd = os.path.join(args.sweep_dir, f"{args.arm}_s{seed}")
            sd = os.path.join(args.sweep_dir, f"single_{tree}_s{seed}")
            jc, sc = load_cells(jd, tree), load_cells(sd, tree)
            if jc is None or sc is None:
                print(f"{tree:9s} {seed:4d} {'':>6s} | missing cell dump, skipped")
                continue
            ex = val_examples(tree, split_seed=seed)
            if len(ex) != jc.shape[0] or len(ex) != sc.shape[0]:
                print(f"{tree:9s} {seed:4d} | split mismatch, skipped")
                continue
            jw, jh, wt = terms(jc, ex)
            sw, sh, _ = terms(sc, ex)
            d_where, d_which = jw - sw, jh - sh
            d_tot = d_where + d_which
            tw = float(np.sum(d_where * wt) / np.sum(wt))
            th = float(np.sum(d_which * wt) / np.sum(wt))
            share = 100 * abs(tw) / max(abs(tw) + abs(th), 1e-12)
            print(f"{tree:9s} {seed:4d} {int(np.sum(wt)):6d} | {tw:+8.3f} {th:+8.3f} "
                  f"{tw + th:+8.3f} | {share:6.0f}%")
            for key, val in (('where', d_where), ('which', d_which), ('weight', wt)):
                rows[key].append(val)
                pooled[key].append(val)
        if rows['weight']:
            per_tree[tree] = {k: np.concatenate(v) for k, v in rows.items()}

    print('-' * 68)
    print(f"{'POOLED PER TREE':25s} {'dWhere [95% CI]':>26s} {'dWhich [95% CI]':>26s}")
    for tree, data in per_tree.items():
        pw, lw, hw = paired_bootstrap(data['where'], data['weight'])
        ph, lh, hh = paired_bootstrap(data['which'], data['weight'])
        print(f"{tree:25s} {pw:+7.3f} [{lw:+6.3f},{hw:+6.3f}] "
              f"{ph:+7.3f} [{lh:+6.3f},{hh:+6.3f}]")

    if pooled['weight']:
        w = {k: np.concatenate(v) for k, v in pooled.items()}
        pw, lw, hw = paired_bootstrap(w['where'], w['weight'])
        ph, lh, hh = paired_bootstrap(w['which'], w['weight'])
        pt, lt, ht = paired_bootstrap(w['where'] + w['which'], w['weight'])
        print('-' * 68)
        print(f"{'ALL TREES':25s} {pw:+7.3f} [{lw:+6.3f},{hw:+6.3f}] "
              f"{ph:+7.3f} [{lh:+6.3f},{hh:+6.3f}]")
        print(f"{'  total':25s} {pt:+7.3f} [{lt:+6.3f},{ht:+6.3f}]  "
              f"(negative favours {args.arm})")


if __name__ == '__main__':
    main()
