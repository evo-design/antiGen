#!/usr/bin/env python
"""How much of `where x which`'s de novo win at 1% is tie-breaking?

At K = 8 the model beats the factored baseline on de novo recall at the 10%
budget (+0.104) and LOSES at 1% (-0.067). The 1% result deserves scrutiny,
because the two predictors have very different tie structure and the metric
splits ties fractionally.

`WhereTimesWhichPredictor` is `P(i) * P(a | wt_i)`. For any position with zero
training mutations P(i) is the same smoothed constant, so across that whole set
of positions the score depends only on the pair (wild-type residue, alternative
residue) -- at most 20 x 20 distinct values. That is an enormous block of exactly
equal scores. And a de novo target is by construction one with no training
mutation at its cell, so de novo targets are precisely the ones most likely to
sit inside the block. The ESM-2 arms are continuous and have essentially none.

`compute_hit_rate` gives each tied target a fraction `(k - n_above) / n_at` of a
hit, which is the EXPECTATION of drawing the remaining budget at random from the
tied block. That is the right average, but it means the credit measures how big
the block is, not whether the predictor can discriminate inside it. A real
shortlist has to name k specific cells.

This splits each rung's recall into the part earned by cells ranked STRICTLY
above the budget cut and the part handed over by the tie rule, and then asks what
a single real shortlist would actually catch. The second question has an exact
answer with no sorting: given `n_at` tied cells of which `targets_at` are
targets, and `slots = k - n_above` places left in the budget, the number caught
is Hypergeometric(n_at, targets_at, slots). So the full distribution of "what
one list would catch" is available analytically, and the fractional score is its
mean -- which the script checks.

Usage:
    python experiments/multitree/tie_probe.py [--out DIR]
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

from antiGen.eval_utils import (fit_substitution_baseline,                 # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.models.factory import create_model                            # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import (logp_model, seen_cells, split_temporal,  # noqa: E402
                                    FORECAST_PT, GAP, PANEL, STOP, WXW_SEEDS)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QS = [0.10, 0.01]
DRAWS = 400


def load_joint(seed):
    cp, cfg = load_checkpoint(
        sorted(glob.glob(f'{FORECAST_PT}/joint_panel_s{seed}/*_best.pt'))[-1],
        map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def split_hits(score, tg, aa, keep, q):
    """Recall split into the strictly-above part and the tie-credit part.

    Also returns the per-example counts needed to simulate a real shortlist:
    tied-block size, targets inside it, and budget slots left after the cells
    that genuinely outrank the cut.
    """
    s = score.clone()
    s = torch.where(keep, s, torch.full_like(s, float('nan')))
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    B = s.shape[0]
    fl = s.reshape(B, -1).float()
    n_valid = (~torch.isnan(fl)).sum(1).clamp(min=1).float()
    vb = torch.nanquantile(fl, 1.0 - q, dim=1, interpolation='lower').view(B, 1, 1)
    above, at = s > vb, s == vb
    n_above = above.reshape(B, -1).sum(1).float()
    n_at = at.reshape(B, -1).sum(1).float()
    slots = (q * n_valid - n_above).clamp(min=0.0)
    frac = (slots / n_at.clamp(min=1)).clamp(0.0, 1.0)
    t_above = (above * tg).reshape(B, -1).sum(1)
    t_at = (at * tg).reshape(B, -1).sum(1)
    return dict(strict=t_above.cpu().numpy(),
                tie=(frac * t_at).cpu().numpy(),
                n_at=n_at.cpu().numpy(),
                t_at=t_at.cpu().numpy(),
                slots=slots.cpu().numpy(),
                n=tg.sum((1, 2)).cpu().numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    models = [load_joint(int(x)) for x in args.seeds.split(',')]
    print(f'joint ensemble K={len(models)}', flush=True)

    acc = {}
    for tree in PANEL:
        train_ex, eval_ex = split_temporal(tree, 0)
        if not eval_ex:
            continue
        L = len(eval_ex[0].aa_idx)
        aa = torch.stack([e.aa_idx for e in eval_ex]).to(dev)
        tg = torch.stack([e.mat for e in eval_ex]).to(dev)
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
            'frequency': torch.log(frequency_baseline_scores(train_ex, L, E).to(dev)
                                   ).unsqueeze(0).expand(len(eval_ex), -1, -1),
        }
        for nm, sc in maps.items():
            for sub, t in (('all', tg), ('novel', tg * novel)):
                if float(t.sum()) == 0:
                    continue
                for q in QS:
                    acc[(nm, tree, sub, q)] = split_hits(sc, t, aa, keep, q)
        print(f'  {tree}', flush=True)

    rngs = np.random.default_rng(0)
    rows = []
    print('\n== where recall actually comes from (temporal, standard universe) ==')
    print(f'{"rung":10s} {"subset":6s} {"q":>5s} {"recall":>7s} {"strict":>8s} '
          f'{"from ties":>10s} {"tie share":>10s} {"tied block":>11s} '
          f'{"slots":>8s}')
    for q in QS:
        for sub in ('all', 'novel'):
            for nm in ('model', 'wxw', 'frequency'):
                ks = [t for t in PANEL if (nm, t, sub, q) in acc]
                if not ks:
                    continue
                d = {f: np.concatenate([acc[(nm, t, sub, q)][f] for t in ks])
                     for f in ('strict', 'tie', 'n_at', 't_at', 'slots', 'n')}
                N = d['n'].sum()
                rec = (d['strict'].sum() + d['tie'].sum()) / N
                st = d['strict'].sum() / N
                ti = d['tie'].sum() / N
                print(f'{nm:10s} {sub:6s} {q:5.0%} {rec:7.3f} {st:8.3f} '
                      f'{ti:10.3f} {ti / max(rec, 1e-9):9.0%} '
                      f'{d["n_at"].mean():11.0f} {d["slots"].mean():8.0f}')
                rows.append(dict(rung=nm, subset=sub, q=q, recall=rec,
                                 strict=st, tie=ti,
                                 tie_share=ti / max(rec, 1e-9),
                                 n_at=float(d['n_at'].mean()),
                                 slots=float(d['slots'].mean())))

    ## What ONE real shortlist would catch. Cells inside the tied block are
    ## indistinguishable to the predictor, so a list of k cells draws from it
    ## without replacement: Hypergeometric(n_at, t_at, slots). The fractional
    ## score is this draw's mean, which the `mean` column checks.
    print('\n== what a single top-q list would actually catch ==')
    print(f'{"rung":10s} {"subset":6s} {"q":>5s} {"fractional":>11s} '
          f'{"draw mean":>10s} {"p5":>6s} {"median":>7s} {"p95":>6s} '
          f'{"P(catch 0)":>11s}')
    for q in QS:
        for sub in ('all', 'novel'):
            for nm in ('model', 'wxw', 'frequency'):
                ks = [t for t in PANEL if (nm, t, sub, q) in acc]
                if not ks:
                    continue
                d = {f: np.concatenate([acc[(nm, t, sub, q)][f] for t in ks])
                     for f in ('strict', 'tie', 'n_at', 't_at', 'slots', 'n')}
                N = d['n'].sum()
                ngood = np.minimum(d['t_at'], d['n_at']).astype(int)
                nbad = (d['n_at'] - ngood).astype(int)
                nsamp = np.minimum(d['slots'].round(), d['n_at']).astype(int)
                ok = (ngood + nbad > 0) & (nsamp > 0)
                tot = np.repeat(d['strict'][None, :], DRAWS, axis=0)
                if ok.any():
                    tot[:, ok] += rngs.hypergeometric(
                        ngood[ok], nbad[ok], nsamp[ok], size=(DRAWS, int(ok.sum())))
                per = tot.sum(axis=1) / N
                frac_rec = (d['strict'].sum() + d['tie'].sum()) / N
                print(f'{nm:10s} {sub:6s} {q:5.0%} {frac_rec:11.3f} '
                      f'{per.mean():10.3f} {np.percentile(per, 5):6.3f} '
                      f'{np.median(per):7.3f} {np.percentile(per, 95):6.3f} '
                      f'{(tot.sum(axis=1) == 0).mean():10.1%}')
                rows.append(dict(rung=nm, subset=sub, q=q, draw_mean=per.mean(),
                                 p5=float(np.percentile(per, 5)),
                                 p95=float(np.percentile(per, 95)),
                                 p_zero=float((tot.sum(axis=1) == 0).mean())))

    with open(f'{args.out}/tie_probe.json', 'w') as fh:
        # numpy scalars are not JSON-serializable; the table above is the result
        # and this file is only a convenience dump, so coerce rather than fuss
        json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer))
                        else v) for k, v in r.items()} for r in rows], fh, indent=1)


if __name__ == '__main__':
    main()
