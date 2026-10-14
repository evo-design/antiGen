#!/usr/bin/env python
"""Precision as well as recall for the post-D forecasts.

hit@q (what NOTES s11 reports) is a RECALL: of the observed post-D substitutions, the
fraction whose cell lands in the top q of the predictor's ranked (L x E) map.

The complement is precision: of the cells the predictor flags, the fraction that are
really observed. Two versions, and only one of them is informative:

  precision@q   At a fixed selection *fraction* q, precision = hits / (q * n_valid) and
                recall = hits / n_mut, so within an example the two are proportional --
                precision@q is recall rescaled by n_mut / (q * n_valid) and tells you
                nothing new. It is reported anyway because it is the number the question
                literally asks for, and because its smallness is the point: q=10% of an
                L x E map is ~1200 cells against ~2 observed mutations, so precision is
                bounded above by ~0.002 no matter how good the predictor is.

  precision@k   Of the top k cells, how many are observed substitutions. This is the
                decision-relevant quantity -- "we flag k candidates per genotype, how
                many are right" -- and is NOT a rescaling of recall, because the ceiling
                min(n_mut, k) / k varies with k.

Both are micro-averaged: total hits / total selected, pooled over examples and trees.
The per-example ceiling is reported alongside precision@k, since with ~2 observed
mutations per example no predictor can exceed min(n_mut, k)/k.

Ranking conventions match `compute_hit_rate`: wild-type cells excluded, and the stop
column NaN-masked for predictors with OUTPUTS_LEGITIMATE_STOP=False (the ESM-2 arms).

Usage:
    python experiments/multitree/forecast_precision.py [--trees ...] [--ks 1 5 10 20 50]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu                              # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen.config import AlphabetConfig, CodonConfig, ModelConfig, ModelType  # noqa: E402
from antiGen.eval_utils import (fit_substitution_baseline,                 # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts,
                                where_only_baseline_scores, WhereTimesWhichPredictor)
from antiGen.load import load_examples                                     # noqa: E402
from antiGen.models.factory import create_model                            # noqa: E402
from calibrate_forecast import collect, temp_probs, AL, E, PANEL           # noqa: E402

ROOT = '/large_storage/hielab/ispecht/multitree/forecast'
DATA = 'experiments/multitree/data/forecast'
D = '2025-01-01'
LAM = 0.20
SEEDS = [0, 1, 2]
STOP = AL.index('*')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ranked(scores, aa, mask_stop):
    """(B, n_valid) cell indices sorted best-first, wild-type and stop excluded."""
    s = scores.clone()
    if mask_stop:
        s[:, :, STOP] = float('nan')
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    flat = s.reshape(s.shape[0], -1)
    valid = ~torch.isnan(flat)
    order = torch.argsort(torch.where(valid, flat, torch.full_like(flat, -np.inf)),
                          dim=1, descending=True)
    return order, valid.sum(1)


def topk_hits(order, n_valid, tgt_flat, ks, qs):
    """Hits among the top k cells (for each k) and the top q fraction (for each q)."""
    B = order.shape[0]
    g = torch.gather(tgt_flat, 1, order)            # targets in ranked order
    cum = torch.cumsum(g, dim=1)
    out_k = {k: cum[:, min(k, cum.shape[1]) - 1].clone() for k in ks}
    out_q = {}
    for q in qs:
        nsel = torch.clamp((q * n_valid.float()).round().long(), min=1)
        out_q[q] = (torch.gather(cum, 1, (nsel - 1).unsqueeze(1)).squeeze(1),
                    nsel.float())
    return out_k, out_q


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trees', nargs='+', default=PANEL)
    p.add_argument('--ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
    args = p.parse_args()
    KS, QS = args.ks, [0.1]

    # ---- stage 1: per (subset, tree) fixtures and the seed-independent baselines.
    # Baselines depend only on the pre-D train file, so they are built once per tree.
    acc, fixtures = {}, {}
    for sub, suf in (('all', 'test'), ('novel', 'novel')):
        for t in args.trees:
            fp = f'{DATA}/examples_{t}@{D}_{suf}.txt'
            if not os.path.exists(fp) or os.path.getsize(fp) == 0:
                continue
            te = load_examples(fp, AL, None, source=t)
            if not te:
                continue
            tr = load_examples(f'{DATA}/examples_{t}@{D}_train.txt', AL, None, source=t)
            L = len(te[0].aa_idx)
            aa = torch.stack([e.aa_idx for e in te]).to(dev)
            tg = torch.stack([e.mat for e in te]).to(dev)
            a = acc.setdefault((sub, t), {})
            a['n_mut'] = tg.sum((1, 2)).cpu().numpy()
            a['n_ex'] = len(te)

            fit = fit_substitution_baseline(tr, dev, AL)
            fac = WhereTimesWhichPredictor(
                fit.config, positional_marginal_counts(tr, L, E), fit).to(dev)
            cm = create_model(ModelConfig(model_type=ModelType.CODON, seq_len=L,
                                          alphabet=AlphabetConfig(amino_acids=AL),
                                          codon=CodonConfig()), dev)
            wl = torch.tensor(collect(fac, te, dev)[0], device=dev)
            base = {
                'where_times_which': (wl, False),
                'substitution': (torch.tensor(collect(fit, te, dev)[0], device=dev), False),
                'codon': (torch.tensor(collect(cm, te, dev)[0], device=dev), False),
                'frequency': (torch.log(frequency_baseline_scores(tr, L, E).to(dev)
                                        ).unsqueeze(0).expand(len(te), -1, -1), False),
                'where_only': (torch.log(where_only_baseline_scores(tr, L, E).to(dev)
                                         ).unsqueeze(0).expand(len(te), -1, -1), False),
            }
            del fit, fac, cm
            for nm, (sc, ms) in base.items():
                order, nv = ranked(sc, aa, ms)
                hk, hq = topk_hits(order, nv, tg.reshape(tg.shape[0], -1), KS, QS)
                a[nm] = {'k': {k: v.cpu().numpy() for k, v in hk.items()},
                         'q': {q: (v[0].cpu().numpy(), v[1].cpu().numpy())
                               for q, v in hq.items()},
                         'n_valid': nv.cpu().numpy()}
            fixtures[(sub, t)] = dict(te=te, aa=aa, tg=tg, wl=wl)
            print(f"  baselines {sub}/{t}", flush=True)

    def score_into(sub, t, name, sc, mask_stop):
        f = fixtures[(sub, t)]
        order, nv = ranked(sc, f['aa'], mask_stop)
        hk, hq = topk_hits(order, nv, f['tg'].reshape(f['tg'].shape[0], -1), KS, QS)
        acc[(sub, t)][name] = {
            'k': {k: v.cpu().numpy() for k, v in hk.items()},
            'q': {q: (v[0].cpu().numpy(), v[1].cpu().numpy()) for q, v in hq.items()},
            'n_valid': nv.cpu().numpy()}

    def load(path):
        cp, cfg = load_checkpoint(sorted(glob.glob(path + '/*_best.pt'))[-1],
                                  map_location=dev)
        m = create_model(cfg, dev)
        m.load_state_dict(cp['model_state_dict'], strict=False)
        m.eval()
        return m

    # ---- stage 2: one checkpoint load per (arm, seed); every tree scored from it.
    for seed in SEEDS:
        for arm in ('joint_panel', 'joint_all'):
            m = load(f'{ROOT}/{arm}_{"s"}{seed}')
            for (sub, t), f in fixtures.items():
                lo_np, _, wtm = collect(m, f['te'], dev)
                score_into(sub, t, f'{arm}#{seed}',
                           torch.tensor(lo_np, device=dev), True)
                if arm == 'joint_panel':
                    pm = temp_probs(lo_np, wtm, 1.0)
                    pw = temp_probs(f['wl'].cpu().numpy(), wtm, 1.0)
                    pm = pm / pm.reshape(pm.shape[0], -1).sum(1)[:, None, None]
                    pw = pw / pw.reshape(pw.shape[0], -1).sum(1)[:, None, None]
                    score_into(sub, t, f'mixture#{seed}', torch.log(torch.tensor(
                        ((1 - LAM) * pm + LAM * pw).clip(1e-20), device=dev)), True)
            del m
            print(f"  {arm} seed {seed}", flush=True)
        for t in args.trees:
            keys = [(sub, t) for sub in ('all', 'novel') if (sub, t) in fixtures]
            if not keys:
                continue
            m = load(f'{ROOT}/single_{t}_s{seed}')
            for sub, tt in keys:
                lo_np, _, _ = collect(m, fixtures[(sub, tt)]['te'], dev)
                score_into(sub, tt, f'single#{seed}',
                           torch.tensor(lo_np, device=dev), True)
            del m
        print(f"  singles seed {seed}", flush=True)

    RUNGS = ['mixture', 'joint_panel', 'joint_all', 'single', 'where_times_which',
             'substitution', 'frequency', 'codon', 'where_only']

    def get(sub, t, rung):
        a = acc[(sub, t)]
        if rung in a:
            return [a[rung]]
        return [a[f'{rung}#{s}'] for s in SEEDS if f'{rung}#{s}' in a]

    for sub in ('all', 'novel'):
        trees = [t for t in args.trees if (sub, t) in acc]
        if not trees:
            continue
        n_ex = sum(acc[(sub, t)]['n_ex'] for t in trees)
        n_mut = sum(acc[(sub, t)]['n_mut'].sum() for t in trees)
        print(f"\n{'='*100}\n=== {sub.upper()}: {n_mut:.0f} observed substitutions over "
              f"{n_ex} parent genotypes ({n_mut/n_ex:.2f} per genotype)\n{'='*100}")

        ceil = {k: sum(np.minimum(acc[(sub, t)]['n_mut'], k).sum() for t in trees)
                / (k * n_ex) for k in KS}
        # Expected precision of a random top-k pick, micro-averaged: per example it is
        # n_mut_e / n_valid_e regardless of k.
        rnd = sum(float((acc[(sub, t)]['n_mut']
                         / acc[(sub, t)]['where_times_which']['n_valid']).sum())
                  for t in trees) / n_ex

        hdr = (f"{'rung':20s} " + ' '.join(f"{'P@'+str(k):>7s}" for k in KS)
               + f" | {'R@10%':>7s} {'P@10%':>9s} {'lift':>6s}")
        print(hdr)
        print('-' * len(hdr))
        for rung in RUNGS:
            reps = {t: get(sub, t, rung) for t in trees}
            nrep = len(reps[trees[0]])
            pk = []
            for k in KS:
                hits = np.mean([sum(float(reps[t][i]['k'][k].sum()) for t in trees)
                                for i in range(nrep)])
                pk.append(hits / (k * n_ex))
            hq = np.mean([sum(float(reps[t][i]['q'][0.1][0].sum()) for t in trees)
                          for i in range(nrep)])
            sel = np.mean([sum(float(reps[t][i]['q'][0.1][1].sum()) for t in trees)
                           for i in range(nrep)])
            rec, prec = hq / n_mut, hq / sel
            print(f"{rung:20s} " + ' '.join(f"{v:7.3f}" for v in pk)
                  + f" | {rec:7.3f} {prec:9.5f} {rec / 0.1:5.1f}x")
        print('-' * len(hdr))
        print(f"{'ceiling (perfect)':20s} " + ' '.join(f"{ceil[k]:7.3f}" for k in KS)
              + f" | {1.000:7.3f} {'':9s} {10.0:5.1f}x")
        print(f"{'random':20s} " + ' '.join(f"{rnd:7.4f}" for _ in KS)
              + f" | {0.100:7.3f} {rnd:9.5f} {1.0:5.1f}x")


if __name__ == '__main__':
    main()
