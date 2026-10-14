#!/usr/bin/env python
"""What does the model's learned P(a | i) actually look like?

Joint training improves the "which residue" term and leaves the "where" term alone
(see decompose_delta.py). This asks what that term has learned: does the implied
substitution preference reproduce known chemistry -- one-step codon accessibility,
transition/transversion bias, BLOSUM62 -- or something those do not capture?

Method. From the dumped per-cell maps, the conditional at each position is
P(a | i) = P(i, a) / P(i). Averaging those conditionals over every position sharing a
wild-type residue gives an implied (vocab x vocab) substitution matrix, directly
comparable to the reference matrices the codon and substitution baselines are built
from. Averaging is unweighted over positions, so the matrix describes the model's
chemistry rather than its view of which sites are mutable.

Reported per run:
  * Spearman correlation of log P(a|wt) against each reference, off-diagonal only;
  * an OLS fit of log P(a|wt) on [log codon accessibility, BLOSUM62], which is exactly
    the substitution baseline's parameterization, so the coefficients are comparable to
    its fitted w_codon / w_blosum;
  * the residual structure -- what is left once both references are regressed out.

Usage:
    python experiments/multitree/substitution_profile.py <sweep_dir> \
        [--runs joint5_s0 single_h3n2_s0] [--tree h3n2]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_index import val_examples                                   # noqa: E402

from antiGen.models.codon_fitness_predictor import build_codon_lookup          # noqa: E402
from antiGen.models.substitution_fitness_predictor import (                    # noqa: E402
    build_blosum_matrix, build_transition_transversion_counts)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALPHABET)
AA20 = list(range(20))          # drop '*' and '-' from the comparison


def implied_matrix(cells, examples):
    """(vocab, vocab) mean of P(a | i) over positions, grouped by wild-type residue."""
    total = np.zeros((E, E))
    count = np.zeros(E)
    for idx, example in enumerate(examples):
        row = cells[idx]                      # (L, E) normalized joint
        wt = example.aa_idx.numpy()
        p_pos = row.sum(axis=1, keepdims=True)
        good = p_pos[:, 0] > 0
        cond = np.zeros_like(row)
        cond[good] = row[good] / p_pos[good]
        np.add.at(total, wt[good], cond[good])
        np.add.at(count, wt[good], 1.0)
    keep = count > 0
    out = np.full((E, E), np.nan)
    out[keep] = total[keep] / count[keep, None]
    return out


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def references():
    codon = build_codon_lookup(ALPHABET, kappa=1.0).numpy()      # already log
    blosum, _ = build_blosum_matrix(ALPHABET)
    ts, tv = build_transition_transversion_counts(ALPHABET)
    ts, tv = ts.numpy(), tv.numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        ts_frac = np.where(ts + tv > 0, ts / np.maximum(ts + tv, 1e-12), np.nan)
    return {'codon(log acc)': codon, 'BLOSUM62': blosum.numpy(),
            'transition frac': ts_frac}


def offdiag(mat):
    m = mat[np.ix_(AA20, AA20)].astype(float).copy()
    np.fill_diagonal(m, np.nan)
    return m.ravel()


def load_cells(run_dir, tree):
    m = sorted(glob.glob(os.path.join(run_dir, "*_val_cells.npz")))
    if not m:
        return None
    with np.load(m[-1]) as h:
        return h[tree].astype(np.float64) if tree in h.files else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sweep_dir')
    p.add_argument('--tree', default='h3n2')
    p.add_argument('--runs', nargs='+', default=None)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    runs = args.runs or [f'joint5_s{args.seed}', f'single_{args.tree}_s{args.seed}']
    refs = references()
    ref_cols = {k: offdiag(v) for k, v in refs.items()}

    print(f"implied substitution matrix on {args.tree}, from P(a|i) averaged over positions")
    print(f"off-diagonal 20x20 cells only ('*' and '-' excluded)\n")
    print(f"{'run':22s} " + ' '.join(f'{k:>16s}' for k in refs) + f" {'R2(codon+blos)':>15s}")
    print('-' * 95)

    stash = {}
    for run in runs:
        d = os.path.join(args.sweep_dir, run)
        cells = load_cells(d, args.tree)
        if cells is None:
            print(f"{run:22s} no cell dump for {args.tree}")
            continue
        ex = val_examples(args.tree, split_seed=args.seed)
        if len(ex) != cells.shape[0]:
            print(f"{run:22s} split mismatch")
            continue
        mat = implied_matrix(cells, ex)
        with np.errstate(divide='ignore'):
            logm = np.log(np.where(mat > 0, mat, np.nan))
        y = offdiag(logm)
        stash[run] = y

        cors = [spearman(y, ref_cols[k]) for k in refs]

        # OLS on the substitution baseline's own two features.
        X = np.column_stack([ref_cols['codon(log acc)'], ref_cols['BLOSUM62'],
                             np.ones_like(y)])
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        pred = X[ok] @ beta
        ss_res = float(((y[ok] - pred) ** 2).sum())
        ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        print(f"{run:22s} " + ' '.join(f'{c:16.3f}' for c in cors) + f" {r2:15.3f}")
        stash[run + '::resid'] = (ok, y[ok] - pred)
        stash[run + '::beta'] = beta

    print()
    for run in runs:
        if run + '::beta' in stash:
            b = stash[run + '::beta']
            print(f"{run:22s} log P(a|wt) ~ {b[0]:+.3f}*log_codon {b[1]:+.3f}*BLOSUM "
                  f"{b[2]:+.3f}")

    # Does joint training move the matrix, and if so is the movement explained by the
    # two chemistry features or orthogonal to them?
    if len(runs) == 2 and all(r in stash for r in runs):
        a, b = stash[runs[0]], stash[runs[1]]
        delta = a - b
        print(f"\n{runs[0]} minus {runs[1]}:")
        print(f"  Spearman(delta, .) " + '  '.join(
            f'{k}={spearman(delta, ref_cols[k]):+.3f}' for k in refs))
        ok = np.isfinite(delta)
        print(f"  mean |delta log P(a|wt)| = {np.abs(delta[ok]).mean():.4f} "
              f"over {ok.sum()} cells")


if __name__ == '__main__':
    main()
