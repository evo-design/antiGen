#!/usr/bin/env python
"""Does the model's learned P(a | i) reproduce a known substitution baseline?

`substitution_profile.py` asked this for one tree and one seed and found the implied
matrix most resembles one-step codon accessibility. This repeats it across all five
evaluation trees and three seeds, and adds the three controls that result needs before
it can be believed:

1. *The empirical reference.* Codon accessibility and BLOSUM62 are not the only
   candidate baselines -- the most obvious one is the substitution matrix the model's
   own training mutations imply. Added here as `empirical(train)`.

2. *The additive control.* A matrix in which some residues are simply popular
   destinations -- log P(a|wt) = u_wt + v_a, no pairwise structure at all -- would still
   correlate with any reference that shares those marginals. Every reference is
   therefore also scored after row and column effects are projected out of both sides,
   which leaves only genuine wt x alt interaction.

3. *The averaging artefact* flagged as open in NOTES.md s10: the arithmetic mean of
   conditionals over positions is dominated by the most confident positions. The
   geometric mean (`--mean geometric`) is the robustness check.

Residual structure is characterised two ways: how much of it is shared across trees
(a chemistry term the references miss) versus tree-specific, and whether it loads on
hydrophobicity or volume.

Usage:
    python experiments/multitree/substitution_profile_sweep.py <sweep_dir>
    python experiments/multitree/substitution_profile_sweep.py <sweep_dir> --mean geometric
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_index import val_examples, examples_path                   # noqa: E402

from antiGen.models.codon_fitness_predictor import build_codon_lookup          # noqa: E402
from antiGen.models.substitution_fitness_predictor import (                    # noqa: E402
    build_blosum_matrix, build_transition_transversion_counts)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALPHABET)
AA20 = list(range(20))
TREES = ['ncov', 'h3n2', 'h1n1pdm', 'rsv_a', 'rsv_b']
SEEDS = [0, 1, 2]

# Kyte-Doolittle hydropathy and van der Waals volume, indexed by ALPHABET order.
HYDRO = dict(A=1.8, C=2.5, D=-3.5, E=-3.5, F=2.8, G=-0.4, H=-3.2, I=4.5, K=-3.9,
             L=3.8, M=1.9, N=-3.5, P=-1.6, Q=-3.5, R=-4.5, S=-0.8, T=-0.7, V=4.2,
             W=-0.9, Y=-1.3)
VOLUME = dict(A=88.6, C=108.5, D=111.1, E=138.4, F=189.9, G=60.1, H=153.2, I=166.7,
              K=168.6, L=166.7, M=162.9, N=114.1, P=112.7, Q=143.8, R=173.4, S=89.0,
              T=116.1, V=140.0, W=227.8, Y=193.6)


# --------------------------------------------------------------------------- matrices

def implied_matrix(cells, examples, mean='arithmetic'):
    """(vocab, vocab) mean of P(a | i) over positions, grouped by wild-type residue.

    `arithmetic` averages the conditionals, `geometric` averages their logs; the latter
    is far less dominated by a handful of confident positions.
    """
    total = np.zeros((E, E))
    count = np.zeros(E)
    for idx, example in enumerate(examples):
        row = cells[idx]                      # (L, E) normalized joint
        wt = example.aa_idx.numpy()
        p_pos = row.sum(axis=1, keepdims=True)
        good = p_pos[:, 0] > 0
        cond = np.zeros_like(row)
        cond[good] = row[good] / p_pos[good]
        if mean == 'geometric':
            cond = np.log(np.clip(cond, 1e-12, None))
        np.add.at(total, wt[good], cond[good])
        np.add.at(count, wt[good], 1.0)
    keep = count > 0
    out = np.full((E, E), np.nan)
    out[keep] = total[keep] / count[keep, None]
    if mean == 'arithmetic':
        with np.errstate(divide='ignore'):
            out = np.log(np.where(out > 0, out, np.nan))
    return out                                 # always returned on the log scale


def empirical_matrix(examples, pseudocount=0.5):
    """log (counts + pseudocount) of observed wt -> alt substitutions, row-normalized."""
    counts = np.zeros((E, E))
    for example in examples:
        wt = example.aa_idx.numpy()
        nz = (example.mat > 0).nonzero().tolist()
        for pos, alt in nz:
            counts[wt[pos], alt] += float(example.mat[pos, alt])
    raw = counts + pseudocount
    return np.log(raw / raw.sum(axis=1, keepdims=True)), counts


def train_examples(source, split_seed, val_frac=0.1):
    from antiGen.load import load_examples
    from antiGen.train import split_examples
    with contextlib.redirect_stdout(io.StringIO()):
        ex = load_examples(examples_path(source), ALPHABET, None, source=source)
        train_by, _ = split_examples({source: ex}, 'random', val_frac, split_seed)
    return train_by[source]


# ------------------------------------------------------------------------ stats utils

def offdiag(mat):
    m = np.asarray(mat, dtype=float)[np.ix_(AA20, AA20)].copy()
    np.fill_diagonal(m, np.nan)
    return m.ravel()


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float('nan')


def _row_col_design():
    """Design matrix of row (wt) and column (alt) dummies over the off-diagonal cells."""
    rows, cols = np.meshgrid(AA20, AA20, indexing='ij')
    rows, cols = rows.ravel(), cols.ravel()
    X = np.zeros((len(rows), 2 * 20 + 1))
    X[np.arange(len(rows)), rows] = 1.0
    X[np.arange(len(rows)), 20 + cols] = 1.0
    X[:, -1] = 1.0
    return X


DESIGN = _row_col_design()


def deadditive(y):
    """Residual of y after projecting out per-wild-type and per-alt main effects.

    What survives is the pairwise interaction: the part of the matrix that cannot be
    explained by "this residue mutates often" plus "that residue is a popular target".
    """
    ok = np.isfinite(y)
    beta, *_ = np.linalg.lstsq(DESIGN[ok], y[ok], rcond=None)
    out = np.full_like(y, np.nan)
    out[ok] = y[ok] - DESIGN[ok] @ beta
    return out


def r2_additive(y):
    r = deadditive(y)
    ok = np.isfinite(y)
    return 1 - float((r[ok] ** 2).sum()) / float(((y[ok] - y[ok].mean()) ** 2).sum())


def ols_r2(y, cols):
    X = np.column_stack(list(cols) + [np.ones_like(y)])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid = y[ok] - X[ok] @ beta
    ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
    full = np.full_like(y, np.nan)
    full[ok] = resid
    return 1 - float((resid ** 2).sum()) / ss_tot, beta, full


# ----------------------------------------------------------------------------- loader

def load_cells(run_dir, tree):
    m = sorted(glob.glob(os.path.join(run_dir, "*_val_cells.npz")))
    if not m:
        return None
    with np.load(m[-1]) as h:
        return h[tree].astype(np.float64) if tree in h.files else None


def references():
    codon = build_codon_lookup(ALPHABET, kappa=1.0).numpy()      # log accessibility
    blosum, _ = build_blosum_matrix(ALPHABET)
    ts, tv = build_transition_transversion_counts(ALPHABET)
    ts, tv = ts.numpy(), tv.numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        ts_frac = np.where(ts + tv > 0, ts / np.maximum(ts + tv, 1e-12), np.nan)
    hyd = np.array([HYDRO.get(a, np.nan) for a in ALPHABET])
    vol = np.array([VOLUME.get(a, np.nan) for a in ALPHABET])
    d_hyd = np.abs(hyd[:, None] - hyd[None, :])
    d_vol = np.abs(vol[:, None] - vol[None, :])
    return ({'codon': codon, 'BLOSUM62': blosum.numpy(), 'ts_frac': ts_frac},
            {'|d hydropathy|': d_hyd, '|d volume|': d_vol})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sweep_dir')
    p.add_argument('--mean', choices=['arithmetic', 'geometric'], default='arithmetic')
    p.add_argument('--trees', nargs='+', default=TREES)
    p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = p.parse_args()

    refs, chem = references()
    ref_cols = {k: offdiag(v) for k, v in refs.items()}
    chem_cols = {k: offdiag(v) for k, v in chem.items()}
    ref_int = {k: deadditive(v) for k, v in ref_cols.items()}

    print(f"implied substitution matrix, {args.mean} mean over positions")
    print("off-diagonal 20x20 cells; 'interaction' = after row/col main effects removed\n")

    store = {}
    rows = []
    for tree, seed in itertools.product(args.trees, args.seeds):
        ex_val = val_examples(tree, split_seed=seed)
        emp_log, emp_counts = empirical_matrix(train_examples(tree, seed))
        emp_col = offdiag(emp_log)
        emp_seen = offdiag(emp_counts) > 0

        for arm, run in (('joint5', f'joint5_s{seed}'),
                         ('single', f'single_{tree}_s{seed}')):
            cells = load_cells(os.path.join(args.sweep_dir, run), tree)
            if cells is None or len(ex_val) != cells.shape[0]:
                print(f"  {tree}/{arm}/s{seed}: missing or split mismatch")
                continue
            y = offdiag(implied_matrix(cells, ex_val, args.mean))
            store[(tree, seed, arm)] = y

            ycol = {**ref_cols, 'empirical': emp_col}
            cors = {k: spearman(y, v) for k, v in ycol.items()}
            yi = deadditive(y)
            ints = {k: spearman(yi, v) for k, v in
                    {**ref_int, 'empirical': deadditive(emp_col)}.items()}
            r2cb, _, _ = ols_r2(y, [ref_cols['codon'], ref_cols['BLOSUM62']])
            rows.append(dict(tree=tree, seed=seed, arm=arm, add=r2_additive(y),
                             r2cb=r2cb, cor=cors, inter=ints,
                             emp_seen_cor=spearman(y[emp_seen], emp_col[emp_seen])))

    keys = ['codon', 'BLOSUM62', 'ts_frac', 'empirical']
    hdr = f"{'tree':9s} {'s':>1s} {'arm':7s} {'R2 add':>7s} {'R2 c+b':>7s}  "
    hdr += ' '.join(f'{k:>9s}' for k in keys) + '   |  ' + \
           ' '.join(f'{k[:6]+"*":>9s}' for k in keys)
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        line = (f"{r['tree']:9s} {r['seed']:1d} {r['arm']:7s} {r['add']:7.3f} "
                f"{r['r2cb']:7.3f}  "
                + ' '.join(f"{r['cor'][k]:9.3f}" for k in keys) + '   |  '
                + ' '.join(f"{r['inter'][k]:9.3f}" for k in keys))
        print(line)
    print("\n* = interaction only (row/col main effects projected out of both sides)")

    # ---- pooled means per arm
    print(f"\n{'arm':8s} {'R2 add':>7s} {'R2 c+b':>7s}  "
          + ' '.join(f'{k:>9s}' for k in keys) + '   |  '
          + ' '.join(f'{k[:6]+"*":>9s}' for k in keys))
    for arm in ('joint5', 'single'):
        sub = [r for r in rows if r['arm'] == arm]
        if not sub:
            continue
        m = lambda f: np.mean([f(r) for r in sub])                    # noqa: E731
        print(f"{arm:8s} {m(lambda r: r['add']):7.3f} {m(lambda r: r['r2cb']):7.3f}  "
              + ' '.join(f"{m(lambda r, k=k: r['cor'][k]):9.3f}" for k in keys)
              + '   |  '
              + ' '.join(f"{m(lambda r, k=k: r['inter'][k]):9.3f}" for k in keys))

    # ---- joint minus single: where does joint training move the matrix?
    print("\njoint5 - single, per tree (mean over seeds): Spearman of the delta")
    print(f"{'tree':9s} " + ' '.join(f'{k:>12s}' for k in refs)
          + f" {'mean|delta|':>12s}")
    deltas = {}
    for tree in args.trees:
        ds = [store[(tree, s, 'joint5')] - store[(tree, s, 'single')]
              for s in args.seeds
              if (tree, s, 'joint5') in store and (tree, s, 'single') in store]
        if not ds:
            continue
        d = np.nanmean(np.stack(ds), axis=0)
        deltas[tree] = d
        print(f"{tree:9s} " + ' '.join(f'{spearman(d, ref_cols[k]):12.3f}' for k in refs)
              + f" {np.nanmean(np.abs(d)):12.4f}")

    # ---- residual structure
    print("\nresidual after regressing out codon + BLOSUM (joint5, mean over seeds)")
    resids = {}
    for tree in args.trees:
        ys = [store[(tree, s, 'joint5')] for s in args.seeds
              if (tree, s, 'joint5') in store]
        if not ys:
            continue
        y = np.nanmean(np.stack(ys), axis=0)
        _, _, r = ols_r2(y, [ref_cols['codon'], ref_cols['BLOSUM62']])
        resids[tree] = r
        print(f"  {tree:9s} " + '  '.join(
            f'{k}={spearman(r, v):+.3f}' for k, v in chem_cols.items()))

    print("\n  cross-tree Spearman of that residual (shared => a real missing term,"
          "\n  tree-specific => the model's own per-tree quirks):")
    names = [t for t in args.trees if t in resids]
    print('           ' + ' '.join(f'{t[:8]:>9s}' for t in names))
    for a in names:
        print(f"  {a:9s}" + ' '.join(
            f'{spearman(resids[a], resids[b]):9.3f}' for b in names))

    # Seed-to-seed agreement bounds how much of any cross-tree number is even real.
    print("\n  seed-to-seed Spearman of the raw joint5 matrix (the noise ceiling):")
    for tree in args.trees:
        ss = [spearman(store[(tree, a, 'joint5')], store[(tree, b, 'joint5')])
              for a, b in itertools.combinations(args.seeds, 2)
              if (tree, a, 'joint5') in store and (tree, b, 'joint5') in store]
        if ss:
            print(f"    {tree:9s} {np.mean(ss):.3f}")


if __name__ == '__main__':
    main()
