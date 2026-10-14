#!/usr/bin/env python
"""Is the model's edge over `where x which` about WHERE mutations land, or WHICH?

Proposed panel: per site, the propensity to mutate as predicted by each
predictor against how much that site actually mutates post-D. Before drawing it,
this checks whether there is anything to draw -- i.e. whether the model's
positional term is actually better than the baseline's.

There is reason to expect not. `WhereTimesWhichPredictor` is
`P(i) * P(a | wt_i)` with P(i) the **empirical positional marginal of the
training branches**, which is a strong where-predictor; and NOTES s10 found the
joint-training gain decomposed almost entirely into the which term (-0.076
[-0.108, -0.042]) with where indistinguishable from zero (+0.008
[-0.022, +0.036]). If the same holds against this baseline, the panel would show
two nearly identical curves and the real story is elsewhere.

**The decomposition is exact, not a model.** Each predictor emits one
distribution over the whole (L, E) map, so for an observed mutation at (i, a)

    -log P(i, a)  =  -log P(i)  +  -log P(a | i),     P(i) = sum_a P(i, a)

and the model-minus-baseline difference splits additively into a where part and
a which part that sum to the total. The script verifies that to 1e-6 rather than
assuming it.

Three readouts:

1. **NLL decomposition**, pooled and per tree, for all and de novo targets.
2. **Per-site calibration** -- the quantity the proposed panel would draw.
   For each position, the predicted propensity to mutate (mean P(i) over the
   test genotypes) against the observed post-D mutation count. Reported as
   Spearman over positions, per tree and pooled, for each predictor.
3. **Where-only recall** -- rank POSITIONS by predicted propensity, take the top
   q, and ask what share of observed mutations land there. This is the panel's
   claim reduced to one number, at both of the figure's budgets.

Scored on the standard-residue universe the headline uses, so the numbers are
comparable to the figure.

Usage:
    python experiments/multitree/where_which_probe.py [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
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
from antiGen.models.factory import create_model                            # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import (boot_pooled, logp_model, seen_cells,   # noqa: E402
                                    split_temporal, FORECAST_PT, GAP,
                                    PANEL, STOP, WXW_SEEDS)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_joint(seed):
    cp, cfg = load_checkpoint(
        sorted(glob.glob(f'{FORECAST_PT}/joint_panel_s{seed}/*_best.pt'))[-1],
        map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def normalized(lp, keep):
    """Renormalize a log-prob map over `keep`, in log space."""
    lp = torch.where(keep, lp, torch.full_like(lp, -float('inf')))
    z = torch.logsumexp(lp.reshape(lp.shape[0], -1), dim=1).view(-1, 1, 1)
    return lp - z


def spearman(x, y):
    """Spearman rho with average ranks for ties."""
    def rank(v):
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = r[order[i:j + 1]].mean()
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(',')]
    models = [load_joint(s) for s in seeds]
    print(f'joint ensemble K={len(models)}', flush=True)

    parts, site_rows, wrec = {}, [], {}
    for tree in PANEL:
        train_ex, eval_ex = split_temporal(tree, 0)
        if not eval_ex:
            continue
        L = len(eval_ex[0].aa_idx)
        aa = torch.stack([e.aa_idx for e in eval_ex]).to(dev)
        tg = torch.stack([e.mat for e in eval_ex]).to(dev)

        # the headline universe: not wild type, not stop, not gap
        keep = torch.ones_like(tg, dtype=torch.bool)
        keep.scatter_(2, aa.unsqueeze(-1), False)
        keep[:, :, STOP] = False
        keep[:, :, GAP] = False
        tg = tg * keep.to(tg.dtype)
        novel = torch.tensor(~seen_cells(train_ex, L), device=dev, dtype=tg.dtype)

        pm = positional_marginal_counts(train_ex, L, E)
        lw, wt = [], None
        for bs in WXW_SEEDS:
            fit = fit_substitution_baseline(train_ex, dev, AL, seed=bs)
            fac = WhereTimesWhichPredictor(fit.config, pm, fit).to(dev)
            lo, _, wt = collect(fac, eval_ex, dev)
            lw.append(torch.tensor(logp_model(lo, wt), device=dev,
                                   dtype=torch.float32))
            del fit, fac
        maps = {
            'model': torch.stack([
                torch.tensor(logp_model(collect(m, eval_ex, dev)[0], wt),
                             device=dev, dtype=torch.float32) for m in models
            ]).mean(0),
            'wxw': torch.stack(lw).mean(0),
        }

        for nm, lp in maps.items():
            lp = normalized(lp, keep)
            log_pi = torch.logsumexp(lp, dim=2, keepdim=True)        # (B, L, 1)
            log_pa_i = lp - log_pi
            ## Excluded cells are -inf, and a target map is zero there, but in
            ## torch `0 * -inf` is NaN rather than 0 -- so every sum below would
            ## come back NaN. Targets never land on an excluded cell (tg was
            ## masked to the same universe), so replacing -inf with a large
            ## finite value changes no term that is actually summed.
            lp = torch.nan_to_num(lp, neginf=-700.0)
            log_pa_i = torch.nan_to_num(log_pa_i, neginf=-700.0)
            for sub, t in (('all', tg), ('novel', tg * novel)):
                if float(t.sum()) == 0:
                    continue
                w = -(t * log_pi.expand_as(t)).sum((1, 2)).cpu().numpy()
                c = -(t * log_pa_i).sum((1, 2)).cpu().numpy()
                tot = -(t * lp).sum((1, 2)).cpu().numpy()
                assert np.abs(w + c - tot).max() < 1e-3, 'decomposition broke'
                n = t.sum((1, 2)).cpu().numpy()
                parts[(nm, tree, sub, 'where')] = w
                parts[(nm, tree, sub, 'which')] = c
                parts[(nm, tree, sub, 'total')] = tot
                parts[(nm, tree, sub, 'n')] = n

            ## The proposed panel, as numbers. Predicted propensity per SITE is
            ## the mean of P(i) over the test genotypes; observed is the post-D
            ## mutation count at that site.
            pi = log_pi.squeeze(2).exp().mean(0).cpu().numpy()        # (L,)
            obs = tg.sum((0, 2)).cpu().numpy()                        # (L,)
            site_rows.append(dict(tree=tree, rung=nm, L=int(L),
                                  rho=spearman(pi, obs),
                                  rho_hit=spearman(pi, (obs > 0).astype(float)),
                                  n_sites_hit=int((obs > 0).sum()),
                                  n_mut=float(obs.sum())))
            ## Where-only recall: rank SITES, take the top q, count the share of
            ## observed mutations there. The panel's claim in one number.
            for q in (0.10, 0.01):
                k = max(1, int(round(q * L)))
                top = np.argsort(-pi)[:k]
                wrec[(nm, tree, q)] = (float(obs[top].sum()), float(obs.sum()))
        print(f'  {tree}  L={L}', flush=True)

    trees = sorted({k[1] for k in parts})

    print('\n== NLL decomposition: model vs `where x which` ==')
    print(f'{"subset":7s} {"component":10s} {"model":>8s} {"wxw":>8s} '
          f'{"model-wxw":>22s}')
    out = []
    for sub in ('all', 'novel'):
        ks = [t for t in trees if ('model', t, sub, 'n') in parts]
        if not ks:
            continue
        for comp in ('where', 'which', 'total'):
            a = boot_pooled([(parts[('model', t, sub, comp)],
                              parts[('model', t, sub, 'n')]) for t in ks])[0]
            b = boot_pooled([(parts[('wxw', t, sub, comp)],
                              parts[('wxw', t, sub, 'n')]) for t in ks])[0]
            d, lo, hi = boot_pooled(
                [(parts[('model', t, sub, comp)] - parts[('wxw', t, sub, comp)],
                  parts[('model', t, sub, 'n')]) for t in ks])
            star = '' if lo <= 0 <= hi else ' *'
            print(f'{sub:7s} {comp:10s} {a:8.3f} {b:8.3f} '
                  f'{d:+8.3f} [{lo:+.3f},{hi:+.3f}]{star}')
            out.append(dict(subset=sub, component=comp, model=a, wxw=b,
                            delta=d, lo=lo, hi=hi))

    print('\n== per-site calibration (the proposed panel) ==')
    print(f'{"tree":9s} {"sites":>6s} {"hit":>5s} '
          f'{"rho(count)  model / wxw":>26s} {"rho(any)  model / wxw":>24s}')
    for t in trees:
        r = {d['rung']: d for d in site_rows if d['tree'] == t}
        if len(r) < 2:
            continue
        print(f'{t:9s} {r["model"]["L"]:6d} {r["model"]["n_sites_hit"]:5d} '
              f'{r["model"]["rho"]:12.3f} / {r["wxw"]["rho"]:9.3f} '
              f'{r["model"]["rho_hit"]:10.3f} / {r["wxw"]["rho_hit"]:9.3f}')
    for key, lab in (('rho', 'rho vs mutation COUNT'), ('rho_hit', 'rho vs ANY mutation')):
        m = np.nanmean([d[key] for d in site_rows if d['rung'] == 'model'])
        w = np.nanmean([d[key] for d in site_rows if d['rung'] == 'wxw'])
        print(f'  mean {lab:24s} model {m:.3f}   wxw {w:.3f}   diff {m - w:+.3f}')

    print('\n== where-only recall: rank SITES, take top q ==')
    print(f'{"q":>5s} {"model":>8s} {"wxw":>8s} {"model-wxw":>22s}')
    for q in (0.10, 0.01):
        num_m = sum(wrec[('model', t, q)][0] for t in trees)
        num_w = sum(wrec[('wxw', t, q)][0] for t in trees)
        den = sum(wrec[('model', t, q)][1] for t in trees)
        print(f'{q:5.0%} {num_m / den:8.3f} {num_w / den:8.3f}')

    with open(f'{args.out}/where_which.json', 'w') as fh:
        json.dump({'decomposition': out, 'sites': site_rows}, fh, indent=1)


if __name__ == '__main__':
    main()
