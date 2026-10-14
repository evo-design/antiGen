#!/usr/bin/env python
"""Can recalibration rescue the model's novel-substitution forecasts?

The temporal holdout (NOTES s11) found the model *ranks* novel post-D substitutions as
well as `where_times_which` (hit@10% 0.511 vs 0.500) while scoring 0.74 nats worse in
NLL. That is a probability-mass problem, not a ranking problem, so it should be fixable
without retraining. This script measures where the mass goes and how much three
corrections can recover.

The corrections, in increasing order of what they assume:

  temperature   probs ~ softmax(logits / T). Monotone in the logits, so the cell
                ranking -- and therefore every hit@k -- is EXACTLY unchanged. Pure
                density fix.
  novelty       multiply every cell never seen in the pre-D snapshot by alpha, then
                renormalize. Seen-ness is a property of pre-D data, so the correction
                itself is causally legal; only its coefficient needs fitting.
  mixture       (1-lam) * p_model + lam * p_where_times_which. Changes ranking too.

**Every number this script prints with --oracle is an UPPER BOUND, not a result.** It
fits the coefficient on the post-D set it then scores, which violates the standing
causal-selection rule. It exists to decide whether the causal version (fit on a nested
pre-D holdout) is worth building. Nothing from an oracle run may be quoted as a forecast.

Usage:
    python experiments/multitree/calibrate_forecast.py <run_dir> --oracle
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers.utils.import_utils as _iu
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen.config import AlphabetConfig                                   # noqa: E402
from antiGen.eval_utils import (compute_probs, fit_substitution_baseline,   # noqa: E402
                                load_checkpoint, positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.load import load_examples                                      # noqa: E402
from antiGen.models.factory import create_model                             # noqa: E402
from antiGen.train import MutationDataset                                   # noqa: E402

AL = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(AL)
PANEL = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'denv1', 'denv2', 'denv3', 'denv4']


def seen_mask(train_ex, seq_len):
    """(L, E) bool: cells with at least one observed substitution before D."""
    m = np.zeros((seq_len, E), dtype=bool)
    for ex in train_ex:
        nz = (ex.mat > 0).nonzero().tolist()
        for pos, alt in nz:
            m[pos, alt] = True
    return m


@torch.no_grad()
def collect(model, examples, device, batch_size=8):
    """Raw logits, wild-type mask and targets for one tree, as dense numpy arrays."""
    loader = DataLoader(MutationDataset(examples), batch_size=batch_size, shuffle=False)
    LO, TG, WT = [], [], []
    for aa, tg in loader:
        aa, tg = aa.to(device), tg.to(device)
        lo = model(aa).float()
        wt = torch.ones_like(lo, dtype=torch.bool)
        wt.scatter_(2, aa.unsqueeze(-1), False)
        LO.append(lo.cpu().numpy()); TG.append(tg.cpu().numpy()); WT.append(wt.cpu().numpy())
    return (np.concatenate(LO), np.concatenate(TG), np.concatenate(WT))


def nll_from_probs(probs, targets):
    """Mean NLL per observed mutation, with the same renormalization training uses."""
    z = probs.reshape(probs.shape[0], -1).sum(axis=1).clip(1e-12)
    p = probs / z[:, None, None]
    return float(-(targets * np.log(p.clip(1e-12))).sum() / max(targets.sum(), 1e-9))


def temp_probs(logits, wt, T):
    """Exactly `compute_probs(logits / T, ...)`, including its 1e-8 floor.

    The floor is not cosmetic: `compute_probs` clamps the post-softmax map to
    [1e-8, 1-1e-8], which caps the NLL any single cell can contribute. On ncov that is
    worth 1.5 nats. Reproducing it is mandatory -- every number in NOTES s11 was scored
    through it, and a calibration study that dropped it would be measuring a different
    objective from the one it is trying to improve.
    """
    x = logits / T
    x = x - x.reshape(x.shape[0], -1).max(axis=1)[:, None, None]
    p = np.exp(x)
    p = p / p.reshape(p.shape[0], -1).sum(axis=1)[:, None, None]
    return np.clip(p * wt, 1e-8, 1 - 1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--data', default='experiments/multitree/data/forecast')
    p.add_argument('--date', default='2025-01-01')
    p.add_argument('--trees', nargs='+', default=PANEL)
    p.add_argument('--oracle', action='store_true',
                   help='fit coefficients on the post-D set being scored (UPPER BOUND)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = sorted(glob.glob(os.path.join(args.run_dir, '*_best.pt')))[-1]
    checkpoint, config = load_checkpoint(ck, map_location=device)
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"checkpoint {os.path.basename(ck)} (epoch {checkpoint.get('epoch')})\n")

    store = {}
    for tree in args.trees:
        tr_p = f'{args.data}/examples_{tree}@{args.date}_train.txt'
        tr = load_examples(tr_p, AL, None, source=tree)
        for sub, suf in (('all', 'test'), ('novel', 'novel')):
            tp = f'{args.data}/examples_{tree}@{args.date}_{suf}.txt'
            if not os.path.exists(tp) or os.path.getsize(tp) == 0:
                continue
            te = load_examples(tp, AL, None, source=tree)
            if not te:
                continue
            L = len(te[0].aa_idx)
            lo, tg, wt = collect(model, te, device)
            # where_times_which on the same examples, for the mixture arm
            fit = fit_substitution_baseline(tr, device, AL)
            fac = WhereTimesWhichPredictor(fit.config,
                                           positional_marginal_counts(tr, L, E),
                                           fit).to(device)
            wl, _, _ = collect(fac, te, device)
            wp = np.exp(wl - wl.reshape(wl.shape[0], -1).max(axis=1)[:, None, None]) * wt
            wp = wp / wp.reshape(wp.shape[0], -1).sum(axis=1).clip(1e-12)[:, None, None]
            store[(tree, sub)] = dict(logits=lo, targets=tg, wt=wt, wxw=wp,
                                      seen=seen_mask(tr, L))
            del fit, fac

    # ---------- diagnostic: where does the mass sit?
    print("=== mass allocation: model probability on cells already seen pre-D")
    print(f"{'tree':9s} {'subset':6s} {'n':>5s} {'P(seen cell)':>13s} "
          f"{'actual seen frac':>17s} {'overweight':>11s}")
    for (tree, sub), d in sorted(store.items()):
        p = temp_probs(d['logits'], d['wt'], 1.0)
        p = p / p.reshape(p.shape[0], -1).sum(axis=1).clip(1e-12)[:, None, None]
        mass = float((p * d['seen'][None]).sum() / p.shape[0])
        actual = float((d['targets'] * d['seen'][None]).sum() / max(d['targets'].sum(), 1))
        print(f"{tree:9s} {sub:6s} {d['targets'].sum():5.0f} {mass:13.3f} "
              f"{actual:17.3f} {mass - actual:+11.3f}")

    if not args.oracle:
        return

    # ---------- oracle sweeps (UPPER BOUNDS)
    print("\n=== ORACLE upper bounds -- coefficient fitted on the scored set. NOT a result.")
    Ts = np.concatenate([np.arange(0.5, 1.01, 0.1), np.arange(1.2, 6.01, 0.2),
                         np.arange(6.5, 20.01, 0.5)])
    As = np.concatenate([np.arange(1.0, 10.01, 0.5), np.arange(11.0, 60.01, 2.0)])
    Ls = np.arange(0.0, 1.001, 0.05)

    for sub in ('all', 'novel'):
        keys = [k for k in store if k[1] == sub]
        if not keys:
            continue
        print(f"\n--- {sub.upper()} (pooled over {len(keys)} trees)")

        def pooled(fn):
            num = den = 0.0
            for k in keys:
                d = store[k]
                pr = fn(d)
                z = pr.reshape(pr.shape[0], -1).sum(axis=1).clip(1e-12)
                pn = pr / z[:, None, None]
                num += -(d['targets'] * np.log(pn.clip(1e-12))).sum()
                den += d['targets'].sum()
            return num / den

        base = pooled(lambda d: temp_probs(d['logits'], d['wt'], 1.0))
        wxw = pooled(lambda d: d['wxw'])
        print(f"  model as-is        {base:7.3f}")
        print(f"  where_times_which  {wxw:7.3f}   (gap {base - wxw:+.3f})")

        tv = [(T, pooled(lambda d, T=T: temp_probs(d['logits'], d['wt'], T))) for T in Ts]
        bT, bN = min(tv, key=lambda x: x[1])
        print(f"  + temperature      {bN:7.3f}   at T={bT:.1f}   "
              f"(recovers {base - bN:.3f} of {max(base - wxw, 0):.3f}; ranking unchanged)")

        av = [(a, pooled(lambda d, a=a: temp_probs(d['logits'], d['wt'], 1.0)
                         * np.where(d['seen'][None], 1.0, a))) for a in As]
        bA, aN = min(av, key=lambda x: x[1])
        print(f"  + novelty weight   {aN:7.3f}   at alpha={bA:.1f}")

        cv = []
        for T in Ts:
            for a in As[::3]:
                cv.append(((T, a), pooled(
                    lambda d, T=T, a=a: temp_probs(d['logits'], d['wt'], T)
                    * np.where(d['seen'][None], 1.0, a))))
        (cT, cA), cN = min(cv, key=lambda x: x[1])
        print(f"  + both             {cN:7.3f}   at T={cT:.1f}, alpha={cA:.1f}")

        mv = [(l, pooled(lambda d, l=l: (1 - l) * (
                  temp_probs(d['logits'], d['wt'], 1.0)
                  / temp_probs(d['logits'], d['wt'], 1.0).reshape(
                      d['logits'].shape[0], -1).sum(axis=1).clip(1e-12)[:, None, None])
                  + l * d['wxw'])) for l in Ls]
        bL, mN = min(mv, key=lambda x: x[1])
        print(f"  + mixture w/ wxw   {mN:7.3f}   at lambda={bL:.2f}")


if __name__ == '__main__':
    main()
