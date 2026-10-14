#!/usr/bin/env python
"""One rule that beats `where x which` on BOTH all and de novo substitutions.

The two predictors are complementary on de novo targets (`rank_agreement.py`): same
recall, different mutations caught. The arithmetic mixture of NOTES s11 does not cash
that in for ranking -- it is +0.009 on de novo recall -- and there are two reasons,
both of which this script addresses.

**Arithmetic pooling cannot reorder the head of the list.** With
`p = (1-lam) p_model + lam p_wxw` at lam = 0.20, the model is far peakier than the
baseline, so in the top decile the model term dominates and the blend's ranking IS the
model's ranking. recall@10% reads only that decile, so it barely moves -- while NLL,
which reads the whole map including the tail where `wxw` takes over, improves a lot.
That is exactly the pattern s11 reports. Log-linear (geometric) pooling,
`log p = (1-w) log p_model + w log p_wxw`, moves every cell's rank and is the
pooling rule a ranking metric actually wants.

**A single global weight is the wrong shape.** The model dominates on cells it has
seen and merely ties on cells it has not, so the right weight is a function of the
evidence at the cell -- not one constant. The gated rules below carry one weight for
seen cells and another for unseen ones, which lets a single rule keep the model's
large margin on all substitutions and still hand authority to the baseline exactly
where the model has nothing to say. Seen-ness is a property of the pre-D training
data, so the gate is computable at prediction time and causally legal.

**Selection.** Every weight is chosen on the pre-D validation split -- the same split
that chose the epoch, which the model never trained on -- and, for the de novo
objective, on the subset of it that is de novo with respect to the inner training
half. That subset holds 182 targets against the 145 being forecast, so it is a
novelty-matched selection set of adequate size. This is what NOTES s12 lacked when it
ruled the nested TEMPORAL design infeasible at ~37 examples: conditioning the existing
pre-D split on novelty is a different and much cheaper way to get forecast-shaped
selection. It is still temporally interpolating, and that is its honest limitation.

`where x which` and the seen-cell gate are refit on the inner training half for
selection, so nothing fitted on a selection target is used to score it. For the post-D
pass both are fit on the whole pre-D file, exactly as the figure does.

Post-D oracle optima are printed for every family, clearly marked. They are upper
bounds, never results -- the standing causal-selection rule forbids quoting them.

Usage:
    python experiments/multitree/fuse_forecast.py [--out DIR]
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
from antiGen.train import split_examples                                   # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import logp_model, seen_cells                  # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402

PANEL = ['ncov', 'rsv_a', 'rsv_b', 'h5nx', 'measles',
         'denv1', 'denv2', 'denv3', 'denv4']
SEEDS = [0, 1, 2]
D = '2025-01-01'
DATA = 'experiments/multitree/data/forecast'
FORECAST_PT = '/large_storage/hielab/ispecht/multitree/forecast_pertree'
STOP = AL.index('*')
Q = 0.1
LOW = -30.0            # log-prob floor, below the 1e-8 clamp temp_probs applies
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_joint(seed):
    path = sorted(glob.glob(f'{FORECAST_PT}/joint_panel_s{seed}/*_best.pt'))[-1]
    cp, cfg = load_checkpoint(path, map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def prep(models, examples, train_ex, L):
    """log p_model (per seed, and the seed ensemble), log p_wxw, gate, targets."""
    aa = torch.stack([e.aa_idx for e in examples]).to(dev)
    tg = torch.stack([e.mat for e in examples]).to(dev)

    fit = fit_substitution_baseline(train_ex, dev, AL)
    fac = WhereTimesWhichPredictor(
        fit.config, positional_marginal_counts(train_ex, L, E), fit).to(dev)
    wl, _, wt = collect(fac, examples, dev)
    del fit, fac
    lw = torch.tensor(logp_model(wl, wt), device=dev, dtype=torch.float32)

    lm = {s: torch.tensor(logp_model(collect(models[s], examples, dev)[0], wt),
                          device=dev, dtype=torch.float32) for s in SEEDS}
    # seed ensemble: mean log-prob, i.e. the geometric mean of the three maps
    lm['ens'] = torch.stack([lm[s] for s in SEEDS]).mean(0)

    seen = torch.tensor(seen_cells(train_ex, L), device=dev)
    return aa, tg, lm, lw, seen


def mid_ranks(x):
    """Per-example normalized ranks of a score map, in [0, 1]."""
    B = x.shape[0]
    f = x.reshape(B, -1).double()
    order = f.argsort(dim=1)
    r = torch.empty_like(f)
    ar = torch.arange(f.shape[1], device=f.device, dtype=torch.float64)
    r.scatter_(1, order, ar.expand(B, -1))
    return (r / max(f.shape[1] - 1, 1)).reshape(x.shape).float()


def hit_parts(score, aa):
    """compute_hit_rate's top-10% decision, computed once and reused per target set.

    The quantile and the fractional tie-break depend only on the score map, so for a
    weight sweep they are the expensive part and they do not need the targets at all.
    Splitting them out is what makes a 121-point gated grid affordable.
    """
    s = score.clone()
    s[:, :, STOP] = float('nan')                    # every fused rule carries ESM-2
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    B = s.shape[0]
    fl = s.reshape(B, -1).float()
    n_valid = (~torch.isnan(fl)).sum(1).clamp(min=1).float()
    vb = torch.nanquantile(fl, 1.0 - Q, dim=1, interpolation='lower').view(B, 1, 1)
    above, at = s > vb, s == vb
    frac = ((Q * n_valid - above.reshape(B, -1).sum(1).float())
            / at.reshape(B, -1).sum(1).float().clamp(min=1)).clamp(0.0, 1.0)
    return above, at, frac


def hits_from(parts, tg):
    above, at, frac = parts
    B = tg.shape[0]
    return float((above * tg).reshape(B, -1).sum(1).sum()
                 + (frac * (at * tg).reshape(B, -1).sum(1)).sum())


GRID_G = np.round(np.arange(0.0, 1.001, 0.05), 3)       # one global weight
GRID_D = np.round(np.arange(0.0, 1.001, 0.10), 3)       # gated: two weights


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--base', default='ens', choices=['ens', '0', '1', '2'],
                    help='model map used at SCORE time: 3-seed ensemble, or one seed. '
                         'Selection always uses the seed whose inner split is clean.')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = {s: load_joint(s) for s in SEEDS}
    hits, ref = {}, {}

    for tree in PANEL:
        with contextlib.redirect_stdout(io.StringIO()):
            pre = load_examples(f'{DATA}/examples_{tree}@{D}_train.txt', AL, None,
                                source=tree)
            post = load_examples(f'{DATA}/examples_{tree}@{D}_test.txt', AL, None,
                                 source=tree)
        L = len(pre[0].aa_idx)

        jobs = []
        for s in SEEDS:
            # Each arm's own --split-seed, so these targets were never trained on.
            # The baseline and the gate are refit on the inner training half alone.
            with contextlib.redirect_stdout(io.StringIO()):
                trb, vab = split_examples({tree: pre}, 'random', 0.1, s)
            jobs.append(('select', trb[tree], vab[tree], s))
        jobs.append(('score', pre, post, args.base))

        for phase, train_ex, eval_ex, which in jobs:
            if not eval_ex:
                continue
            aa, tg, lm_all, lw, seen = prep(models, eval_ex, train_ex, L)
            base = lm_all[which if which == 'ens' else int(which)]
            novel = ~seen
            subs = {'all': tg, 'novel': tg * novel.unsqueeze(0).to(tg.dtype)}
            subs = {k: v for k, v in subs.items() if float(v.sum()) > 0}

            for nm, sc, ms in (('model', base, True), ('wxw', lw, False)):
                for sub, t in subs.items():
                    h = float(per_example_hits(sc, t, aa, ms).sum())
                    p, q = ref.get((phase, sub, nm), (0.0, 0.0))
                    ref[(phase, sub, nm)] = (p + h, q + float(t.sum()))

            rm, rw = mid_ranks(base), mid_ranks(lw)
            sv = seen.unsqueeze(0)
            for fam in ('arith', 'loglin', 'rank'):
                for gated in (False, True):
                    if fam == 'arith' and gated:
                        continue                      # the s11 rule has no gated form
                    grid = GRID_D if gated else GRID_G
                    combos = ([(a, b) for a in grid for b in grid] if gated
                              else [(a, a) for a in grid])
                    name = f'{fam} ({"gated" if gated else "global"})'
                    for a, b in combos:
                        w = torch.where(sv, torch.full_like(base, float(a)),
                                        torch.full_like(base, float(b)))
                        if fam == 'loglin':
                            sc = (1 - w) * base + w * lw
                        elif fam == 'rank':
                            sc = (1 - w) * rm + w * rw
                        else:
                            sc = torch.log(((1 - w) * base.exp()
                                            + w * lw.exp()).clamp(min=1e-30))
                        parts = hit_parts(sc, aa)
                        for sub, t in subs.items():
                            k = (phase, sub, name, float(a), float(b))
                            p, q = hits.get(k, (0.0, 0.0))
                            hits[k] = (p + hits_from(parts, t), q + float(t.sum()))
                        del sc, parts, w
            del aa, tg, lm_all, lw, seen, rm, rw
        print(f'  {tree}', flush=True)

    np.save(f'{args.out}/fuse_hits.npy',
            np.array([{'hits': {str(k): v for k, v in hits.items()},
                       'ref': {str(k): v for k, v in ref.items()}}], dtype=object),
            allow_pickle=True)
    report(hits, ref, args.out)


def report(hits, ref, outdir):
    R = lambda k: ref[k][0] / max(ref[k][1], 1e-9)
    print('\n== reference rungs ==')
    for phase in ('select', 'score'):
        for sub in ('all', 'novel'):
            m, w = ref[(phase, sub, 'model')], ref[(phase, sub, 'wxw')]
            print(f'  {phase:7s} {sub:6s} n={m[1]:6.0f}   model {m[0] / m[1]:.3f}   '
                  f'wxw {w[0] / w[1]:.3f}')

    fams = sorted({k[2] for k in hits})
    print('\n== each family, weight chosen on the pre-D split, reported post-D ==')
    print(f'{"family":18s} {"objective":12s} {"w_seen":>7s} {"w_unsn":>7s} '
          f'{"ALL":>7s} {"vs wxw":>8s} {"vs mdl":>8s} {"DE NOVO":>8s} '
          f'{"vs wxw":>8s} {"vs mdl":>8s}')
    out = []
    for fam in fams:
        ks = sorted({(k[3], k[4]) for k in hits if k[2] == fam})
        for obj in ('novel', 'all', 'both'):
            def val(ph, sub, a, b):
                h, n = hits[(ph, sub, fam, a, b)]
                return h / max(n, 1e-9)
            if obj == 'both':
                best = max(ks, key=lambda ab: val('select', 'novel', *ab)
                           + val('select', 'all', *ab))
            else:
                best = max(ks, key=lambda ab: (val('select', obj, *ab),
                                               val('select', 'all', *ab)))
            a, b = best
            sa, sn = val('score', 'all', a, b), val('score', 'novel', a, b)
            print(f'{fam:18s} {obj:12s} {a:7.2f} {b:7.2f} '
                  f'{sa:7.3f} {sa - R(("score", "all", "wxw")):+8.3f} '
                  f'{sa - R(("score", "all", "model")):+8.3f} '
                  f'{sn:8.3f} {sn - R(("score", "novel", "wxw")):+8.3f} '
                  f'{sn - R(("score", "novel", "model")):+8.3f}')
            out.append(dict(family=fam, objective=obj, w_seen=float(a),
                            w_unseen=float(b), all=sa, novel=sn))

    print('\n== POST-D ORACLE optima -- upper bounds, NOT results ==')
    for fam in fams:
        ks = sorted({(k[3], k[4]) for k in hits if k[2] == fam})
        for sub in ('all', 'novel'):
            a, b = max(ks, key=lambda ab: hits[('score', sub, fam, *ab)][0])
            h, n = hits[('score', sub, fam, a, b)]
            print(f'  {fam:18s} {sub:6s} w=({a:.2f},{b:.2f})  {h / n:.3f}')
    with open(f'{outdir}/fuse_summary.json', 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
