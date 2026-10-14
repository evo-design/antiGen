#!/usr/bin/env python
"""Does the figure's NLL column have a mass leak, the way the zero-shot one does?

`zeroshot_eval.py` turned the `substitution` rung's raw score matrix into
probabilities with a plain softmax, so that rung spent a share of its mass on
wild-type cells -- cells that are wild-type by construction and can never be a
target -- while the ESM-2 rungs, which had been through `logp_model`, did not.
The question here is whether the same class of error is in Figure 5.

**Wild type is handled correctly in the figure.** Every rung goes through
`temp_probs(sc, wtm, 1.0)`, whose `p * wt` zeroes the wild-type cell before the
clamp, and `wtm` comes from `collect`, which scatters False onto the query's own
residue. So all nine rungs are masked identically. The clamp then lifts those
zeros to 1e-8 each, which over an L x E map is ~1e-5 of the total mass -- real,
but five orders of magnitude below anything that moves a number. This script
measures it rather than asserting it.

**The stop and gap columns are NOT handled consistently, and that is a genuine
asymmetry between the figure's two metric columns.** The alphabet is
`ACDEFGHIKLMNPQRSTVWY*-`: 22 letters, with `*` (stop) at 20 and `-` (gap) at 21.

  recall  `per_example_hits` NaN-masks the stop column for predictors whose
          OUTPUTS_LEGITIMATE_STOP is False -- the ESM-2 arms -- and leaves it in
          for the baselines. Nothing masks the gap column anywhere.
  NLL     masks neither, for any rung.

So an ESM-2 arm is charged, in NLL, for whatever probability it puts on a column
that is removed from its ranking before recall is computed. The two metrics are
scored over different cell sets, and the direction is exactly the one that would
make the model look worse on NLL than on recall -- which is the pattern NOTES
s11 recorded on the de novo subset and attributed entirely to density-versus-
ranking. Some of that gap may be this instead.

The script rescores the temporal panel under four conventions and prints the
mass each rung puts on wild-type, stop and gap cells, so the size of the effect
is visible rather than argued:

    figure        exactly what collect_figure_metrics.py does today
    +stop legit   stop masked for the ESM-2-derived rungs only, i.e. the NLL
                  column scored over the same cells its recall column uses
    all masked    stop and gap masked for EVERY rung -- the symmetric choice
    unfloored     the figure's masking without the 1e-8 clamp, since s11 found
                  that floor worth ~1 nat to the model and ~0 to the baselines

Usage:
    python experiments/multitree/figure_nll_audit.py [--out DIR]
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

from antiGen.config import (AlphabetConfig, CodonConfig, ModelConfig,      # noqa: E402
                            ModelType)
from antiGen.eval_utils import (fit_substitution_baseline,                 # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts,
                                where_only_baseline_scores,
                                WhereTimesWhichPredictor)
from antiGen.models.factory import create_model                            # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import (boot_pooled, seen_cells, split_temporal,  # noqa: E402
                                    FORECAST, FORECAST_PT, PANEL, WXW_SEEDS)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FLOOR = 1e-8
STOP, GAP = AL.index('*'), AL.index('-')
# Which rungs carry ESM-2, i.e. have OUTPUTS_LEGITIMATE_STOP = False and get the
# stop column removed before their recall is computed.
ESM2_RUNGS = ('single', 'joint')


def load_any(path):
    cp, cfg = load_checkpoint(sorted(glob.glob(path + '/*_best.pt'))[-1],
                              map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def to_logp(logits, wt, cols, floor):
    """Raw scores -> log-probabilities over the cells a rung may legitimately use.

    `cols` are column indices zeroed in addition to the wild-type cell. `floor`
    reproduces `temp_probs`'s clamp, which lifts every masked or sub-floor cell
    to 1e-8 before renormalizing -- that lift is itself part of the figure's
    objective, so it has to be reproduced exactly to say what it is worth.
    """
    keep = wt.copy()
    for c in cols:
        keep[:, :, c] = False
    B = logits.shape[0]
    x = logits - logits.reshape(B, -1).max(axis=1)[:, None, None]
    p = np.exp(x)
    p = p / p.reshape(B, -1).sum(axis=1)[:, None, None]
    if floor:
        # temp_probs: mask, then clamp -- which LIFTS every masked cell to the
        # floor rather than leaving it at zero. Reproduced exactly, because that
        # lift is part of the objective every number in this arm was scored on.
        p = np.clip(p * keep, floor, 1 - floor)
        p = p / p.reshape(B, -1).sum(axis=1)[:, None, None]
        return np.log(np.clip(p, 1e-300, None))
    # Unfloored: a masked cell is genuinely impossible, so normalize in LOG
    # space. Doing it in probability space divides by a sum that underflows to
    # zero whenever one cell holds nearly all the mass -- pssm puts 92% on the
    # wild-type cell -- and returns NaN for the whole example.
    lp = np.where(keep, np.log(np.clip(p, 1e-300, None)), -np.inf)
    mx = lp.reshape(B, -1).max(axis=1)[:, None, None]
    z = mx + np.log(np.exp(lp - mx).reshape(B, -1).sum(axis=1))[:, None, None]
    return np.clip(lp - z, -700.0, None)


def mass_on(logits, wt):
    """Plain-softmax mass on wild-type, stop and gap cells (no masking at all)."""
    x = logits - logits.reshape(logits.shape[0], -1).max(axis=1)[:, None, None]
    p = np.exp(x)
    p = p / p.reshape(p.shape[0], -1).sum(axis=1)[:, None, None]
    B = p.shape[0]
    return (float((p * ~wt).sum() / B), float(p[:, :, STOP].sum() / B),
            float(p[:, :, GAP].sum() / B))


def hits_at_q(scores, tg, aa, cols, q=0.1):
    """compute_hit_rate's fractional tie-breaking, with explicit column masking.

    `cols` is what this rung may NOT use. The figure passes (STOP,) for the
    ESM-2 rungs and () for the baselines, which is the asymmetry being tested:
    n_valid is counted after masking, so the two get different budgets from the
    same q.
    """
    s = scores.clone()
    for c in cols:
        s[:, :, c] = float('nan')
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    B = s.shape[0]
    fl = s.reshape(B, -1).float()
    n_valid = (~torch.isnan(fl)).sum(1).clamp(min=1).float()
    vb = torch.nanquantile(fl, 1.0 - q, dim=1, interpolation='lower').view(B, 1, 1)
    above, at = s > vb, s == vb
    n_above = above.reshape(B, -1).sum(1).float()
    frac = ((q * n_valid - n_above)
            / at.reshape(B, -1).sum(1).float().clamp(min=1)).clamp(0.0, 1.0)
    sel = above.float() + frac.view(-1, 1, 1) * at.float()
    return sel, float(n_valid.float().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--trees', default=','.join(PANEL),
                    help='subset to audit. The quantity being measured -- how much '
                         'of a rung\'s mass lands on cells it cannot score -- is a '
                         'property of the predictor, not of the panel, so a few '
                         'trees settle it and the whole panel is only needed to '
                         'restate the ladder.')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(',')]
    panel = [t for t in args.trees.split(',') if t]

    CONV = ['figure', '+stop legit', 'all masked', 'unfloored']
    nll, nvec, mass = {}, {}, {}
    rec, nvalid = {}, {}

    for tree in panel:
        train_ex, eval_ex = split_temporal(tree, 0)
        if not eval_ex:
            continue
        L = len(eval_ex[0].aa_idx)
        tg = torch.stack([e.mat for e in eval_ex]).to(dev)
        novel = torch.tensor(~seen_cells(train_ex, L), device=dev, dtype=tg.dtype)
        ## Column-class masks over the alphabet: standard residues 0..19, the
        ## stop column, the gap column. A target's class decides whether a rung
        ## that cannot score that column had any chance at it.
        ## A target can land on the example's OWN wild-type residue -- a
        ## degenerate wt->wt "substitution". Every rung masks that cell to zero
        ## by construction, so no predictor can ever score it: under the
        ## figure's 1e-8 floor each costs ~18 nats instead of ~9, and with the
        ## floor removed each costs ~700. Two of the panel's 564 post-D targets
        ## are like this (both h5nx; the training files have none). They are
        ## split out rather than left in `standard`, because they are not a
        ## prediction task -- they are a defect in the test set.
        aa = torch.stack([e.aa_idx for e in eval_ex]).to(dev)
        own = torch.zeros_like(tg)
        own.scatter_(2, aa.unsqueeze(-1), 1.0)
        col = {}
        for cls, idx in (('standard', None), ('gap', GAP), ('stop', STOP)):
            m = torch.zeros(E, device=dev, dtype=tg.dtype)
            if idx is None:
                m[:20] = 1
            else:
                m[idx] = 1
            col[cls] = m.view(1, 1, E)
        targets = {}
        for sub, base in (('all', tg), ('novel', tg * novel)):
            targets[(sub, 'any')] = base
            targets[(sub, 'degenerate')] = base * own
            for cls, m in col.items():
                targets[(sub, cls)] = base * m * (1 - own)
            for cls, t in list(targets.items()):
                if cls[0] == sub:
                    nvec[(tree, sub, cls[1])] = t.sum((1, 2)).cpu().numpy()

        pm = positional_marginal_counts(train_ex, L, E)
        raw, wt = {}, None
        subs, wxws = [], []
        for bs in WXW_SEEDS:
            fit = fit_substitution_baseline(train_ex, dev, AL, seed=bs)
            fac = WhereTimesWhichPredictor(fit.config, pm, fit).to(dev)
            lo, _, wt = collect(fit, eval_ex, dev)
            subs.append(lo)
            lo, _, wt = collect(fac, eval_ex, dev)
            wxws.append(lo)
            del fit, fac
        cm = create_model(ModelConfig(
            model_type=ModelType.CODON, seq_len=L,
            alphabet=AlphabetConfig(amino_acids=AL), codon=CodonConfig()), dev)
        raw['codon'] = [collect(cm, eval_ex, dev)[0]]
        del cm
        raw['substitution'] = subs
        raw['where_times_which'] = wxws
        raw['uniform'] = [np.zeros((len(eval_ex), L, E), dtype=np.float32)]
        for nm, fn in (('frequency', frequency_baseline_scores),
                       ('where_only', where_only_baseline_scores)):
            m = torch.log(fn(train_ex, L, E)).cpu().numpy()
            raw[nm] = [np.broadcast_to(m, (len(eval_ex), L, E)).copy()]
        pssm = np.zeros((L, E), dtype=np.float32)
        for ex in train_ex:
            pssm[np.arange(L), ex.aa_idx.numpy()] += 1.0
        raw['pssm'] = [np.broadcast_to(np.log1p(pssm),
                                       (len(eval_ex), L, E)).copy()]
        raw['single'] = [collect(load_any(f'{FORECAST}/single_{tree}_s{s}'),
                                 eval_ex, dev)[0] for s in seeds]
        raw['joint'] = [collect(load_any(f'{FORECAST_PT}/joint_panel_s{s}'),
                                eval_ex, dev)[0] for s in seeds]

        for nm, los in raw.items():
            mass[(nm, tree)] = tuple(float(np.mean(v)) for v in
                                     zip(*[mass_on(l, wt) for l in los]))
            for cname in CONV:
                cols = ()
                floor = FLOOR
                if cname == '+stop legit' and nm in ESM2_RUNGS:
                    cols = (STOP,)
                elif cname == 'all masked':
                    cols = (STOP, GAP)
                elif cname == 'unfloored':
                    floor = None
                lp = np.stack([to_logp(l, wt, cols, floor) for l in los]).mean(0)
                z = np.log(np.exp(lp).reshape(lp.shape[0], -1).sum(axis=1))
                t = torch.tensor(lp - z[:, None, None], device=dev,
                                 dtype=torch.float32)
                for (sub, cls), tt in targets.items():
                    nll[(cname, nm, tree, sub, cls)] = (
                        -(tt * t).sum((1, 2)).cpu().numpy())
                ## Recall on the same map. `figure` reproduces today's rule --
                ## the stop column removed for the ESM-2 rungs only, so each
                ## rung's budget is q * ITS OWN n_valid. `common` gives every
                ## rung the same universe and therefore the same budget;
                ## `common nogap` additionally drops the gap column, which only
                ## the baselines can score.
                if cname in ('figure', 'unfloored'):
                    for rc, cols_r in (
                            (cname, (STOP,) if nm in ESM2_RUNGS else ()),
                            (cname + ' common', (STOP,)),
                            (cname + ' common nogap', (STOP, GAP))):
                        if cname == 'unfloored' and 'common' in rc:
                            continue
                        sel, nv = hits_at_q(t, tg, aa, cols_r)
                        nvalid[(rc, nm, tree)] = nv
                        for (sub, cls), tt in targets.items():
                            rec[(rc, nm, tree, sub, cls)] = (
                                (sel * tt).sum((1, 2)).cpu().numpy())
                del t
        print(f'  {tree}  L={L}', flush=True)

    rungs = ['uniform', 'codon', 'where_only', 'pssm', 'substitution',
             'frequency', 'where_times_which', 'single', 'joint']
    trees = [t for t in panel if (t, 'all', 'any') in nvec]

    print('\n== softmax mass that lands on cells a rung cannot score ==')
    print(f'{"rung":22s} {"wild type":>11s} {"stop *":>11s} {"gap -":>11s}'
          f'   (mean over trees, before any masking)')
    for nm in rungs:
        v = np.mean([mass[(nm, t)] for t in trees], axis=0)
        print(f'{nm:22s} {v[0]:10.4%} {v[1]:10.4%} {v[2]:10.4%}')

    ## Targets split by which column they land in. `gap` targets are deletions
    ## in the alignment -- 2.3% of the panel's post-D substitutions, and a real
    ## residue state (the test set has 741 positions whose wild type IS a gap).
    ## A rung that puts no mass on the gap column cannot predict any of them, so
    ## this split says how much of each rung's score rides on that one column.
    print('\n== targets by column class ==')
    for sub in ('all', 'novel'):
        for cls in ('standard', 'gap', 'stop', 'degenerate'):
            n = sum(nvec[(t, sub, cls)].sum() for t in trees)
            tot = sum(nvec[(t, sub, 'any')].sum() for t in trees)
            print(f'  {sub:6s} {cls:9s} {n:6.0f}  ({n / max(tot, 1):5.1%})')

    for sub in ('all', 'novel'):
        print(f'\n== NLL by target class, {sub} (figure convention) ==')
        print(f'{"rung":22s}' + ''.join(f'{c:>12s}' for c in
                                        ('standard', 'gap', 'stop', 'degenerate')))
        for nm in rungs:
            row = ''
            for cls in ('standard', 'gap', 'stop', 'degenerate'):
                ks = [t for t in trees if nvec[(t, sub, cls)].sum() > 0]
                if not ks:
                    row += f'{"--":>12s}'
                    continue
                est, _, _ = boot_pooled([(nll[('figure', nm, t, sub, cls)],
                                          nvec[(t, sub, cls)]) for t in ks])
                row += f'{est:12.3f}'
            print(f'{nm:22s}{row}')
        print(f'  paired joint - where_times_which on STANDARD targets only, {sub}:')
        ks = [t for t in trees if nvec[(t, sub, 'standard')].sum() > 0]
        for cname in CONV:
            d, lo, hi = boot_pooled(
                [(nll[(cname, 'joint', t, sub, 'standard')]
                  - nll[(cname, 'where_times_which', t, sub, 'standard')],
                  nvec[(t, sub, 'standard')]) for t in ks])
            star = '' if lo <= 0 <= hi else '  *'
            print(f'    {cname:14s} {d:+7.3f} [{lo:+.3f},{hi:+.3f}]{star}')

    for sub in ('all', 'novel'):
        n = sum(nvec[(t, sub, 'any')].sum() for t in trees)
        print(f'\n== NLL per mutation, {sub} (n={n:.0f}) ==')
        print(f'{"rung":22s}' + ''.join(f'{c:>14s}' for c in CONV)
              + f'{"fig - allmask":>15s}')
        for nm in rungs:
            vals = []
            for cname in CONV:
                est, _, _ = boot_pooled([(nll[(cname, nm, t, sub, 'any')],
                                          nvec[(t, sub, 'any')]) for t in trees])
                vals.append(est)
            print(f'{nm:22s}' + ''.join(f'{v:14.3f}' for v in vals)
                  + f'{vals[0] - vals[2]:15.3f}')

        print('  -- ranking under `figure`     : '
              + ' > '.join(sorted(rungs, key=lambda nm: boot_pooled(
                  [(nll[("figure", nm, t, sub, 'any')], nvec[(t, sub, 'any')])
                   for t in trees])[0])))
        print('  -- ranking under `all masked` : '
              + ' > '.join(sorted(rungs, key=lambda nm: boot_pooled(
                  [(nll[("all masked", nm, t, sub, 'any')], nvec[(t, sub, 'any')])
                   for t in trees])[0])))

        for other in ('where_times_which', 'single'):
            print(f'\n  paired joint - {other}, {sub} '
                  f'(negative = multi-tree better):')
            for cname in CONV:
                d, lo, hi = boot_pooled(
                    [(nll[(cname, 'joint', t, sub, 'any')]
                      - nll[(cname, other, t, sub, 'any')],
                      nvec[(t, sub, 'any')]) for t in trees])
                star = '' if lo <= 0 <= hi else '  *'
                print(f'    {cname:14s} {d:+7.3f} [{lo:+.3f},{hi:+.3f}]{star}')

    RCONV = ['figure', 'figure common', 'figure common nogap', 'unfloored']

    print('\n== cells each rung may rank, and the budget q=10% buys ==')
    print(f'{"rung":22s} {"n_valid (figure)":>18s} {"budget":>9s}'
          f'   {"n_valid (common)":>18s} {"budget":>9s}')
    for nm in rungs:
        a = np.mean([nvalid[('figure', nm, t)] for t in trees])
        b = np.mean([nvalid[('figure common', nm, t)] for t in trees])
        print(f'{nm:22s} {a:18.0f} {0.1 * a:9.0f}   {b:18.0f} {0.1 * b:9.0f}')

    for sub in ('all', 'novel'):
        n = sum(nvec[(t, sub, 'any')].sum() for t in trees)
        print(f'\n== RECALL at top 10%, {sub} (n={n:.0f}) ==')
        print(f'{"rung":22s}' + ''.join(f'{c:>22s}' for c in RCONV))
        for nm in rungs:
            row = ''
            for rc in RCONV:
                est, lo, hi = boot_pooled([(rec[(rc, nm, t, sub, 'any')],
                                            nvec[(t, sub, 'any')]) for t in trees])
                row += f'{est:8.3f} [{lo:.2f},{hi:.2f}]'
            print(f'{nm:22s}{row}')
        for other in ('where_times_which', 'single'):
            print(f'  paired joint - {other}, {sub} (positive = multi-tree better):')
            for rc in RCONV:
                d, lo, hi = boot_pooled(
                    [(rec[(rc, 'joint', t, sub, 'any')]
                      - rec[(rc, other, t, sub, 'any')],
                      nvec[(t, sub, 'any')]) for t in trees])
                star = '' if lo <= 0 <= hi else '  *'
                print(f'    {rc:22s} {d:+7.3f} [{lo:+.3f},{hi:+.3f}]{star}')

        ## The contrast restricted to standard-residue targets. The gap column
        ## is the whole story on this subset: the ESM-2 arms rank NO gap cell in
        ## their top 10% while `where x which` catches most of them, so a
        ## handful of targets can erase a lead that is real on the other 92%.
        print(f'  paired joint - where_times_which by class, {sub}:')
        for cls in ('standard', 'gap'):
            ks = [t for t in trees if nvec[(t, sub, cls)].sum() > 0]
            if not ks:
                continue
            d, lo, hi = boot_pooled(
                [(rec[('figure common', 'joint', t, sub, cls)]
                  - rec[('figure common', 'where_times_which', t, sub, cls)],
                  nvec[(t, sub, cls)]) for t in ks])
            n = sum(nvec[(t, sub, cls)].sum() for t in ks)
            star = '' if lo <= 0 <= hi else '  *'
            print(f'    {cls:10s} n={n:4.0f}  {d:+7.3f} [{lo:+.3f},{hi:+.3f}]{star}')

        print(f'  -- recall by target class, {sub}, figure convention --')
        print(f'{"rung":22s}' + ''.join(f'{c:>12s}' for c in
                                        ('standard', 'gap', 'stop')))
        for nm in rungs:
            row = ''
            for cls in ('standard', 'gap', 'stop'):
                ks = [t for t in trees if nvec[(t, sub, cls)].sum() > 0]
                if not ks:
                    row += f'{"--":>12s}'
                    continue
                est, _, _ = boot_pooled([(rec[('figure', nm, t, sub, cls)],
                                          nvec[(t, sub, cls)]) for t in ks])
                row += f'{est:12.3f}'
            print(f'{nm:22s}{row}')

    with open(f'{args.out}/figure_nll_audit.json', 'w') as fh:
        json.dump({'mass': {f'{k[0]}|{k[1]}': v for k, v in mass.items()},
                   'nll': {'|'.join(map(str, k)): float(v.sum())
                           for k, v in nll.items()}},
                  fh, indent=1)


if __name__ == '__main__':
    main()
