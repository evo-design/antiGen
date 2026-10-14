#!/usr/bin/env python
"""Do the multi-tree model and `where x which` make the SAME forecasts?

On the temporal holdout (D = 2025-01-01) the two score almost identically on de novo
recall at top 10% -- 0.501 vs 0.497 (NOTES s12). That is either one predictor computed
twice, or two different predictors whose aggregate happens to coincide. The difference
decides whether there is anything to gain from the model over the factored baseline,
and the ladder cannot tell them apart.

This compares them cell by cell rather than in aggregate:

  rank agreement   Spearman of the two (L x E) score maps per genotype, over the
                   cells both predictors are allowed to use (non-wild-type,
                   non-stop). The seed-to-seed agreement of the joint model with
                   ITSELF is computed as the reference scale -- "how similar is
                   similar" is otherwise unanswerable.
  set agreement    overlap of the two top-10% cell sets, as a fraction of the budget.
  target agreement 2x2 over the observed post-D substitutions: caught by both /
                   by the model only / by the baseline only / by neither. Two
                   predictors with the same recall and different hits have
                   headroom; the same recall on the same hits does not.
  headroom         recall of the union of the two top-10% sets (an oracle: it spends
                   up to twice the budget) and of a rank-average ensemble at the
                   honest 10% budget. Neither is a proposal -- they bound what any
                   combination of these two rankings could buy.

Conventions. The stop column is NaN-masked for ESM-2 rungs and not for the baselines
(`OUTPUTS_LEGITIMATE_STOP`), so the two rungs do not even rank the same cell set. For
the agreement numbers both are restricted to the common set; the canonical recall under
each rung's own mask is printed alongside so these tie out to the figure. De novo uses
the composition-aware rule of `collect_figure_metrics.seen_cells`.

Usage:
    python experiments/multitree/rank_agreement.py [--out DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu                              # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen.eval_utils import (fit_substitution_baseline, load_checkpoint,  # noqa: E402
                                positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.load import load_examples                                     # noqa: E402
from antiGen.models.factory import create_model                            # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import seen_cells                              # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402

PANEL = ['ncov', 'rsv_a', 'rsv_b', 'h5nx', 'measles',
         'denv1', 'denv2', 'denv3', 'denv4']
SEEDS = [0, 1, 2]
D = '2025-01-01'
DATA = 'experiments/multitree/data'
FORECAST_PT = '/large_storage/hielab/ispecht/multitree/forecast_pertree'
STOP = AL.index('*')
Q = 0.1
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_joint(seed):
    path = sorted(glob.glob(f'{FORECAST_PT}/joint_panel_s{seed}/*_best.pt'))[-1]
    cp, cfg = load_checkpoint(path, map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def split_temporal(tree):
    with contextlib.redirect_stdout(io.StringIO()):
        tr = load_examples(f'{DATA}/forecast/examples_{tree}@{D}_train.txt', AL,
                           None, source=tree)
        te = load_examples(f'{DATA}/forecast/examples_{tree}@{D}_test.txt', AL,
                           None, source=tree)
    return tr, te


def ranks(x, valid):
    """Mid-ranks of `x` within `valid`, per row. (B, L*E) float, NaN outside valid."""
    B = x.shape[0]
    flat = x.reshape(B, -1).double().clone()
    vf = valid.reshape(B, -1)
    flat[~vf] = float('inf')
    order = flat.argsort(dim=1)
    r = torch.empty_like(flat)
    ar = torch.arange(flat.shape[1], device=flat.device, dtype=torch.float64)
    r.scatter_(1, order, ar.expand(B, -1))
    # average ranks within tied blocks: sort-based, cheap enough at this size
    srt = flat.gather(1, order)
    for b in range(B):
        s = srt[b]
        n = int(vf[b].sum())
        if n == 0:
            continue
        uniq, inv, cnt = torch.unique_consecutive(s[:n], return_inverse=True,
                                                  return_counts=True)
        if len(uniq) < n:
            start = torch.cumsum(cnt, 0) - cnt
            mid = (start.double() + (cnt.double() - 1) / 2)
            r[b, order[b, :n]] = mid[inv]
    r[~vf] = float('nan')
    return r


def spearman_rows(ra, rb, valid):
    """Per-row Spearman from precomputed mid-ranks."""
    out = []
    for b in range(ra.shape[0]):
        m = valid[b].reshape(-1)
        a, c = ra[b][m], rb[b][m]
        a = a - a.mean()
        c = c - c.mean()
        den = (a.norm() * c.norm()).clamp(min=1e-30)
        out.append(float((a * c).sum() / den))
    return np.array(out)


def topk_mask(score, valid, q=Q):
    """Top-q cells per row within `valid`; k = round(q * n_valid), ties by index."""
    B = score.shape[0]
    flat = score.reshape(B, -1).double().clone()
    vf = valid.reshape(B, -1)
    flat[~vf] = -float('inf')
    out = torch.zeros_like(vf)
    order = flat.argsort(dim=1, descending=True)
    nval = vf.sum(1)
    for b in range(B):
        k = int(round(q * float(nval[b])))
        if k:
            out[b, order[b, :k]] = True
    return out.reshape(score.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []           # per (tree, seed, subset) aggregates
    tgt = []            # per target: one record per observed post-D substitution
    pairs = []          # joint-vs-joint seed pairs, the agreement reference scale

    models = {s: load_joint(s) for s in SEEDS}
    for tree in PANEL:
        train_ex, eval_ex = split_temporal(tree)
        if not eval_ex:
            continue
        L = len(eval_ex[0].aa_idx)
        aa = torch.stack([e.aa_idx for e in eval_ex]).to(dev)
        tg = torch.stack([e.mat for e in eval_ex]).to(dev)
        novel = torch.tensor(~seen_cells(train_ex, L), device=dev)

        fit = fit_substitution_baseline(train_ex, dev, AL)
        fac = WhereTimesWhichPredictor(
            fit.config, positional_marginal_counts(train_ex, L, E), fit).to(dev)
        w_sc = torch.tensor(collect(fac, eval_ex, dev)[0], device=dev)
        del fit, fac

        # cells each rung is allowed to rank, and the common set used for agreement
        not_wt = torch.ones_like(w_sc, dtype=torch.bool)
        not_wt.scatter_(2, aa.unsqueeze(-1), False)
        v_base = not_wt.clone()                       # baselines may use the stop column
        v_model = not_wt.clone()
        v_model[:, :, STOP] = False                   # ESM-2 rungs may not
        common = v_model

        j_sc = {s: torch.tensor(collect(models[s], eval_ex, dev)[0], device=dev)
                for s in SEEDS}

        r_w = ranks(w_sc, common)
        r_j = {s: ranks(j_sc[s], common) for s in SEEDS}
        top_w_c = topk_mask(w_sc, common)
        top_j_c = {s: topk_mask(j_sc[s], common) for s in SEEDS}

        nv = common.reshape(len(eval_ex), -1).sum(1).cpu().numpy()
        for s in SEEDS:
            rho = spearman_rows(r_j[s], r_w, common)
            inter = (top_j_c[s] & top_w_c).reshape(len(eval_ex), -1).sum(1
                                                                        ).cpu().numpy()
            union = (top_j_c[s] | top_w_c).reshape(len(eval_ex), -1).sum(1
                                                                        ).cpu().numpy()
            kk = np.round(Q * nv)
            rows.append(dict(tree=tree, seed=s, rho=rho, overlap=inter / np.maximum(kk, 1),
                             jaccard=inter / np.maximum(union, 1), n_valid=nv,
                             budget=union / np.maximum(nv, 1)))
            # canonical recall, each rung under its own mask
            for sub, t in (('all', tg), ('novel', tg * novel.unsqueeze(0).to(tg.dtype))):
                rows[-1][f'canon_joint_{sub}'] = per_example_hits(
                    j_sc[s], t, aa, True).cpu().numpy()
                rows[-1][f'canon_wxw_{sub}'] = per_example_hits(
                    w_sc, t, aa, False).cpu().numpy()
                rows[-1][f'n_{sub}'] = t.sum((1, 2)).cpu().numpy()

        # the reference scale: how much two seeds of the SAME model agree
        for a in SEEDS:
            for b in SEEDS:
                if a >= b:
                    continue
                it = (top_j_c[a] & top_j_c[b]).reshape(len(eval_ex), -1
                                                       ).sum(1).cpu().numpy()
                pairs.append(dict(tree=tree, a=a, b=b,
                                  rho=spearman_rows(r_j[a], r_j[b], common),
                                  overlap=it / np.maximum(np.round(Q * nv), 1)))

        # ---- per-target records
        nvv = common.reshape(len(eval_ex), -1).sum(1)
        for s in SEEDS:
            # ensemble on the common set: mean of the two normalized mid-ranks
            ens = torch.full_like(r_w, float('nan'))
            cf = common.reshape(len(eval_ex), -1)
            nrm = (nvv.double() - 1).clamp(min=1).unsqueeze(1)
            ens[cf] = ((r_j[s] / nrm)[cf] + (r_w / nrm)[cf]) / 2
            ens_sc = torch.nan_to_num(ens, nan=-1.0).reshape(w_sc.shape)
            top_e = topk_mask(ens_sc, common)
            top_u = top_j_c[s] | top_w_c
            for b, pos, alt in tg.nonzero().tolist():
                tgt.append(dict(
                    tree=tree, seed=s, ex=b, pos=pos, alt=alt,
                    novel=bool(novel[pos, alt]),
                    in_common=bool(common[b, pos, alt]),
                    hit_j=bool(top_j_c[s][b, pos, alt]),
                    hit_w=bool(top_w_c[b, pos, alt]),
                    hit_e=bool(top_e[b, pos, alt]),
                    hit_u=bool(top_u[b, pos, alt]),
                    pct_j=float(r_j[s][b, pos * E + alt] / max(float(nvv[b]) - 1, 1)),
                    pct_w=float(r_w[b, pos * E + alt] / max(float(nvv[b]) - 1, 1)),
                ))
        del j_sc, r_j, r_w, w_sc
        print(f'  {tree}', flush=True)

    np.save(f'{args.out}/rank_agreement_rows.npy', np.array(rows, dtype=object),
            allow_pickle=True)
    with open(f'{args.out}/rank_agreement_targets.json', 'w') as fh:
        json.dump(tgt, fh)
    np.save(f'{args.out}/rank_agreement_pairs.npy', np.array(pairs, dtype=object),
            allow_pickle=True)
    report(rows, pairs, tgt)


def report(rows, pairs, tgt):
    import collections
    # per-tree arrays have different lengths, so concatenate before summing
    W = lambda v, n: (float(np.concatenate(v).sum())
                      / max(float(np.concatenate(n).sum()), 1e-9))

    print('\n== rank agreement, joint vs `where x which`, per genotype '
          '(Spearman over common cells) ==')
    print(f'{"tree":10s} {"n_ex":>5s} {"vs wxw":>9s} {"vs itself":>10s}')
    for tree in PANEL:
        rr = [r for r in rows if r['tree'] == tree]
        pp = [p for p in pairs if p['tree'] == tree]
        if not rr:
            continue
        print(f'{tree:10s} {len(rr[0]["rho"]):5d} '
              f'{np.mean([r["rho"].mean() for r in rr]):9.3f} '
              f'{np.mean([p["rho"].mean() for p in pp]):10.3f}')
    print(f'{"POOLED":10s} {"":5s} '
          f'{np.mean([r["rho"].mean() for r in rows]):9.3f} '
          f'{np.mean([p["rho"].mean() for p in pairs]):10.3f}')

    print('\n== top-10% set overlap (share of the budget held in common) ==')
    print(f'{"tree":10s} {"vs wxw":>9s} {"vs itself":>10s} {"jaccard":>9s}')
    for tree in PANEL:
        rr = [r for r in rows if r['tree'] == tree]
        pp = [p for p in pairs if p['tree'] == tree]
        if not rr:
            continue
        print(f'{tree:10s} {np.mean([r["overlap"].mean() for r in rr]):9.3f} '
              f'{np.mean([p["overlap"].mean() for p in pp]):10.3f} '
              f'{np.mean([r["jaccard"].mean() for r in rr]):9.3f}')
    print(f'{"POOLED":10s} {np.mean([r["overlap"].mean() for r in rows]):9.3f} '
          f'{np.mean([p["overlap"].mean() for p in pairs]):10.3f} '
          f'{np.mean([r["jaccard"].mean() for r in rows]):9.3f}')

    print('\n== canonical recall at top 10% (each rung under its own stop mask) ==')
    for sub in ('all', 'novel'):
        j = W([r[f'canon_joint_{sub}'] for r in rows], [r[f'n_{sub}'] for r in rows])
        w = W([r[f'canon_wxw_{sub}'] for r in rows], [r[f'n_{sub}'] for r in rows])
        n = np.concatenate([r[f"n_{sub}"] for r in rows]).sum() / len(SEEDS)
        print(f'  {sub:6s} n={n:5.0f}  joint {j:.3f}   wxw {w:.3f}')

    print('\n== per-target agreement, common cell set ==')
    for sub, sel in (('all', lambda t: True), ('novel', lambda t: t['novel'])):
        T = [t for t in tgt if sel(t)]
        n = len(T) / len(SEEDS)
        both = sum(t['hit_j'] and t['hit_w'] for t in T) / len(SEEDS)
        jo = sum(t['hit_j'] and not t['hit_w'] for t in T) / len(SEEDS)
        wo = sum(t['hit_w'] and not t['hit_j'] for t in T) / len(SEEDS)
        nei = sum(not t['hit_j'] and not t['hit_w'] for t in T) / len(SEEDS)
        print(f'\n  {sub} (n = {n:.0f} targets, mean over {len(SEEDS)} seeds)')
        print(f'    both          {both:6.1f}  ({both / n:.3f})')
        print(f'    model only    {jo:6.1f}  ({jo / n:.3f})')
        print(f'    wxw only      {wo:6.1f}  ({wo / n:.3f})')
        print(f'    neither       {nei:6.1f}  ({nei / n:.3f})')
        print(f'    recall  model {(both + jo) / n:.3f}   wxw {(both + wo) / n:.3f}')
        print(f'    union  (2x budget, oracle)  {(both + jo + wo) / n:.3f}')
        print(f'    rank-average ensemble @10%  '
              f'{sum(t["hit_e"] for t in T) / len(SEEDS) / n:.3f}')
        print(f'    realized union budget       '
              f'{np.mean([r["budget"].mean() for r in rows]):.3f} of valid cells')
        pj = np.array([t['pct_j'] for t in T])
        pw = np.array([t['pct_w'] for t in T])
        from scipy.stats import spearmanr
        print(f'    target percentile rank, model vs wxw: '
              f'spearman {spearmanr(pj, pw).statistic:.3f}, '
              f'median {np.median(pj):.3f} vs {np.median(pw):.3f}')


if __name__ == '__main__':
    main()
