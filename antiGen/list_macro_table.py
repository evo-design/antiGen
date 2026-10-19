#!/usr/bin/env python
"""Render ``list_macro.py``'s outputs as the CoG-UK comparison table.

Merges every ``--dir`` given (the macro run is split across several jobs so the
headline checkpoints are not queued behind the 650M passes and evcouplings' plmc
rebuild), and prints three things:

1. the ranking, cell-weighted, with the event-weighted column beside it;
2. the event-weighted column checked against the published
   ``quantile_hit_rates_list_coguk*_all-models.csv`` -- it must reproduce them, and a
   disagreement means the scoring path here is wrong, not the weighting;
3. slices of the cell-weighted number: by recurrence (k) and by whether the cell
   occurs in the training window, which is what the re-weighting was for.

Models scored by more than one job are reported as a reproducibility check.

Usage:
    python antiGen/list_macro_table.py                       # the default set of dirs
    python antiGen/list_macro_table.py --dir A --dir B --qtile 0.2
"""
from __future__ import annotations

import argparse
import csv
import os

ROOT = '/large_storage/hielab/ispecht/fitness-inference'
DEFAULT_DIRS = ['evals_sc2_lists_macro', 'evals_sc2_lists_macro_fast',
                'evals_sc2_lists_macro_head', 'evals_sc2_lists_macro_mt2021']
PUB = os.path.join(ROOT, 'evals_sc2_lists')
AL = "ACDEFGHIKLMNPQRSTVWY*-"

NAMES = {
    "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4": "antiGen as ESM 2 (35M) partial FT",
    "esm2_esm2_t12_35M_UR50D_lr0.001_frozen": "antiGen as ESM 2 (35M) LM head FT",
    "esm2_esm2_t12_35M_UR50D_lr0.001_noft": "ESM 2 (35M), untrained",
    "esm2_esm2_t12_35M_UR50D_lr1e-05_full": "antiGen as ESM 2 (35M) full FT",
    "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4": "antiGen as ESM 2 (650M) partial FT",
    "esm2_esm2_t33_650M_UR50D_lr0.001_frozen": "antiGen as ESM 2 (650M) LM head FT",
    "esm2_esm2_t33_650M_UR50D_lr0.001_noft": "ESM 2 (650M), untrained",
    "esm2_esm2_t33_650M_UR50D_lr1e-05_full": "antiGen as ESM 2 (650M) full FT",
    "eve_ft_lr0.0001": "antiGen as EVE full FT",
    "eve_frozen": "EVE",
    "simple_d128_n2_lr0.0001": "antiGen",
    "tranception_frozen_both": "Tranception",
    "tranception_ft_both_lr0.0001": "antiGen as Tranception full FT",
    "blosum62": "BLOSUM62",
    "evcouplings": "EVmutation",
    "pssm_root": "PSSM",
    "pssm_root_position": "PSSM Conservation",
    "CoVFit": "CoVFit",
    "EVEscape": "EVEscape",
    "MSA_Transformer": "MSA Transformer",
    "MSA_Transformer_BA2": "MSA Trans. (BA.2 eval)",
    "MSA_Transformer_XFG": "MSA Trans. (XFG eval)",
    "Omicron_BA1_bind": "BA.1 DMS (bind)",
    "Omicron_BA1_expr": "BA.1 DMS (expression)",
    "Omicron_BA2_bind": "BA.2 DMS (bind)",
    "Omicron_BA2_expr": "BA.2 DMS (expression)",
    "pyr0": "PyR0",
    "mt2021joint_s0_partial4": "multi-tree 2021 joint (seed 0)",
    "mt2021joint_s1_partial4": "multi-tree 2021 joint (seed 1)",
    "mt2021joint_s2_partial4": "multi-tree 2021 joint (seed 2)",
    "mt2021single_s0_partial4": "multi-tree 2021 single (seed 0)",
    "mt2021single_s1_partial4": "multi-tree 2021 single (seed 1)",
    "mt2021single_s2_partial4": "multi-tree 2021 single (seed 2)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', action='append', default=None)
    ap.add_argument('--qtile', default='0.10')
    ap.add_argument('--list', default='coguk')
    args = ap.parse_args()
    dirs = args.dir or [os.path.join(ROOT, d) for d in DEFAULT_DIRS]
    q = f"{float(args.qtile):.2f}"
    L = args.list
    fp, rp = f'list_{L}_fullpool', f'list_{L}'
    dn = f'list_{L}_unseen_fullpool'

    rows, dup = {}, []
    for d in dirs:
        p = os.path.join(d, 'list_macro_summary.csv')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                k = (r['model'], r['variant'], r['qtile'])
                if k in rows and rows[k]['macro'] != r['macro']:
                    dup.append((k, float(rows[k]['macro']), float(r['macro'])))
                elif k in rows:
                    dup.append((k, float(r['macro']), float(r['macro'])))
                rows[k] = r

    pub = {}
    for var, fn in [(fp, f'quantile_hit_rates_list_{L}_fullpool_all-models.csv'),
                    (rp, f'quantile_hit_rates_list_{L}_all-models.csv')]:
        p = os.path.join(PUB, fn)
        if os.path.exists(p):
            with open(p) as f:
                for r in csv.DictReader(f):
                    pub[(r['model'], var)] = float(r[f'top_{100 * float(args.qtile):.1f}%'])

    def get(m, var, col, qq=None):
        r = rows.get((m, var, qq or q))
        return float(r[col]) if r else None

    fmt = lambda v: f"{v:.3f}" if v is not None else "--"
    models = sorted({m for m, _, _ in rows}, key=lambda m: -(get(m, fp, 'macro') or -1))
    pct = f"{100 * float(args.qtile):.0f}%"
    print(f"models scored: {len(models)}   (list={L}, q={pct})\n")
    print(f"| # | model | macro@{pct} | macro@20% | micro@{pct} (published) | "
          f"de novo macro | de novo micro | restricted macro |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, m in enumerate(models, 1):
        print(f"| {i} | {NAMES.get(m, m)} | {fmt(get(m, fp, 'macro'))} "
              f"| {fmt(get(m, fp, 'macro', '0.20'))} | {fmt(get(m, fp, 'micro'))} "
              f"| {fmt(get(m, dn, 'macro'))} | {fmt(get(m, dn, 'micro'))} "
              f"| {fmt(get(m, rp, 'macro'))} |")

    print("\nmicro vs the published event-weighted CSVs:")
    worst, n = (0.0, None), 0
    for m in models:
        for var in (fp, rp):
            a, b = get(m, var, 'micro'), pub.get((m, var))
            if a is None or b is None:
                continue
            n += 1
            if abs(a - b) > worst[0]:
                worst = (abs(a - b), f"{m}/{var}: {a:.8f} vs {b:.8f}")
    print(f"  compared {n} cells; worst |delta| = {worst[0]:.2e} ({worst[1]})")
    if dup:
        w = max(abs(a - b) for _, a, b in dup)
        print(f"  {len(dup)} cell(s) scored by more than one job; worst |delta| = {w:.2e}")

    # ── Slices, from the per-cell file ──────────────────────────────────────
    import numpy as np
    pssm = np.loadtxt('/home/ivan.specht/antiGen-data/output1/pssms/root.csv',
                      delimiter=',', skiprows=1, usecols=range(1, len(AL) + 1))
    pc = {}
    for d in dirs:
        p = os.path.join(d, 'list_macro_percell.csv')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r['variant'] != fp or r['qtile'] != q:
                    continue
                pc.setdefault(r['model'], {})[(int(r['pos1']), AL.index(r['alt']))] = (
                    float(r['k_events']), float(r['rate']))
    if not pc:
        return
    seen = lambda pos1, alt: pssm[pos1 - 1, alt] > 0

    def cut(m, pred):
        v = [rate for (pos1, alt), (k, rate) in pc[m].items() if pred(pos1, alt, k)]
        return sum(v) / len(v) if v else None

    print(f"\nfullpool @{pct}, cell-weighted, sliced "
          "(seen = the cell occurs in the training window, pssm_root > 0):")
    print("| model | all cells | k>=5 | k>=20 | seen cells | de novo cells |")
    print("|---|---:|---:|---:|---:|---:|")
    for m in models:
        if m not in pc:
            continue
        print(f"| {NAMES.get(m, m)} | {fmt(cut(m, lambda p_, a_, k: True))} "
              f"| {fmt(cut(m, lambda p_, a_, k: k >= 5))} "
              f"| {fmt(cut(m, lambda p_, a_, k: k >= 20))} "
              f"| {fmt(cut(m, lambda p_, a_, k: seen(p_, a_)))} "
              f"| {fmt(cut(m, lambda p_, a_, k: not seen(p_, a_)))} |")
    any_m = next(iter(pc))
    ks = sorted(k for k, _ in pc[any_m].values())
    n_seen = sum(1 for (p_, a_) in pc[any_m] if seen(p_, a_))
    print(f"\ncells: {len(ks)} ({n_seen} seen, {len(ks) - n_seen} de novo); "
          f"k events per cell: min {ks[0]:.0f} median {ks[len(ks) // 2]:.0f} "
          f"max {ks[-1]:.0f}; {sum(1 for k in ks if k < 5)} cells with k<5")


if __name__ == "__main__":
    main()
