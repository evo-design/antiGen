#!/usr/bin/env python
"""The temporal-holdout contrast tables in hit@10% instead of NLL.

`forecast_summary.py` reports NLL/mutation, which is what the models are trained on but
which conflates ranking with how sharply probability is allocated. This recomputes the
same ladder and the same paired contrasts under hit@10%: the fraction of observed post-D
substitutions whose cell falls in the top 10% of the predictor's ranked (L x E) map.

**Sign convention is flipped relative to the NLL tables: here POSITIVE = better.**

Two details that must match the library or the model rungs come out ~0.01 too high:
  * fractional tie-breaking at the quantile boundary (`compute_hit_rate`), not a hard
    `>= threshold`, which would hand a coarse predictor like codon a whole tied block;
  * the stop column is NaN-masked for predictors with OUTPUTS_LEGITIMATE_STOP=False
    (every ESM-2 arm) and left alone for the baselines.
Both are asserted against `compute_hit_rate` on the first seed of every rung.

Baselines depend only on the pre-D train file, so they are computed once per tree rather
than once per arm. Results are cached to <out>/hit_cache.npz; delete it to recompute.

Usage:
    python experiments/multitree/forecast_hit_summary.py [--cache DIR] [--trees ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu                           # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen.config import AlphabetConfig, CodonConfig, ModelConfig, ModelType  # noqa: E402
from antiGen.eval_utils import (compute_hit_rate, fit_substitution_baseline,    # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts,
                                where_only_baseline_scores, WhereTimesWhichPredictor)
from antiGen.load import load_examples                                   # noqa: E402
from antiGen.models.factory import create_model                          # noqa: E402
from calibrate_forecast import collect, temp_probs, AL, E, PANEL         # noqa: E402

ROOT = '/large_storage/hielab/ispecht/multitree/forecast'
DATA = 'experiments/multitree/data/forecast'
D = '2025-01-01'
Q = 0.1
LAM = 0.20                      # mixture weight, chosen on pre-D data (NOTES s11)
SEEDS = [0, 1, 2]
STOP = AL.index('*')
BASELINES = ['codon', 'where_only', 'substitution', 'where_times_which', 'frequency']
ARMS = ['single', 'joint_panel', 'joint_all', 'mixture']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def per_example_hits(scores, tg, aa, mask_stop):
    """(B,) hits, reproducing compute_hit_rate's fractional tie-breaking per example."""
    s = scores.clone()
    if mask_stop:
        s[:, :, STOP] = float('nan')
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    B = s.shape[0]
    fl = s.reshape(B, -1).float()
    n_valid = (~torch.isnan(fl)).sum(1).clamp(min=1).float()
    need = Q * n_valid
    vb = torch.nanquantile(fl, 1.0 - Q, dim=1, interpolation='lower').view(B, 1, 1)
    above, at = s > vb, s == vb
    n_above = above.reshape(B, -1).sum(1).float()
    n_at = at.reshape(B, -1).sum(1).float().clamp(min=1)
    frac = ((need - n_above) / n_at).clamp(0.0, 1.0)
    return (above * tg).reshape(B, -1).sum(1) + frac * (at * tg).reshape(B, -1).sum(1)


def checked(name, scores, tg, aa, mask_stop, seed, L):
    h = per_example_hits(scores, tg, aa, mask_stop)
    if seed == 0:
        s2 = scores.clone()
        if mask_stop:
            s2[:, :, STOP] = float('nan')
        lib, _ = compute_hit_rate(s2, tg, aa, qtile=Q, eval_range=list(range(L)))
        assert abs(lib - float(h.sum())) < 1e-3, (name, lib, float(h.sum()))
    return h.cpu().numpy()


def load_model(path):
    cp, cfg = load_checkpoint(sorted(glob.glob(path + '/*_best.pt'))[-1], map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def build(trees):
    acc = {}
    for sub, suf in (('all', 'test'), ('novel', 'novel')):
        for t in trees:
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
            acc[(sub, t, 'n')] = tg.sum((1, 2)).cpu().numpy()

            # ---- baselines: one pass per tree, seed- and arm-independent
            fit = fit_substitution_baseline(tr, dev, AL)
            fac = WhereTimesWhichPredictor(
                fit.config, positional_marginal_counts(tr, L, E), fit).to(dev)
            cm = create_model(ModelConfig(model_type=ModelType.CODON, seq_len=L,
                                          alphabet=AlphabetConfig(amino_acids=AL),
                                          codon=CodonConfig()), dev)
            wl = torch.tensor(collect(fac, te, dev)[0], device=dev)
            sl = torch.tensor(collect(fit, te, dev)[0], device=dev)
            cl = torch.tensor(collect(cm, te, dev)[0], device=dev)
            fq = torch.log(frequency_baseline_scores(tr, L, E).to(dev)
                           ).unsqueeze(0).expand(len(te), -1, -1)
            wo = torch.log(where_only_baseline_scores(tr, L, E).to(dev)
                           ).unsqueeze(0).expand(len(te), -1, -1)
            for nm, sc in (('codon', cl), ('where_only', wo), ('substitution', sl),
                           ('where_times_which', wl), ('frequency', fq)):
                acc[(sub, t, nm)] = checked(nm, sc, tg, aa, False, 0, L)
            del fit, fac, cm

            # ---- arms
            for seed in SEEDS:
                for arm, path in (('single', f'{ROOT}/single_{t}_s{seed}'),
                                  ('joint_panel', f'{ROOT}/joint_panel_s{seed}'),
                                  ('joint_all', f'{ROOT}/joint_all_s{seed}')):
                    m = load_model(path)
                    lo_np, _, wtm = collect(m, te, dev)
                    lo = torch.tensor(lo_np, device=dev)
                    acc.setdefault((sub, t, arm), []).append(
                        checked(arm, lo, tg, aa, True, seed, L))
                    if arm == 'joint_panel':
                        pm = temp_probs(lo_np, wtm, 1.0)
                        pw = temp_probs(wl.cpu().numpy(), wtm, 1.0)
                        pm = pm / pm.reshape(pm.shape[0], -1).sum(1)[:, None, None]
                        pw = pw / pw.reshape(pw.shape[0], -1).sum(1)[:, None, None]
                        mx = torch.log(torch.tensor(
                            ((1 - LAM) * pm + LAM * pw).clip(1e-20), device=dev))
                        acc.setdefault((sub, t, 'mixture'), []).append(
                            checked('mixture', mx, tg, aa, True, seed, L))
                    del m
            print(f"  built {sub}/{t}", flush=True)
    return acc


def boot(A, B, N, nb=20000, sd=0):
    n = len(A)
    pt = (A.sum() - B.sum()) / N.sum()
    rng = random.Random(sd)
    d = []
    for _ in range(nb):
        i = [rng.randrange(n) for _ in range(n)]
        d.append(sum(A[j] - B[j] for j in i) / max(sum(N[j] for j in i), 1e-9))
    d.sort()
    return pt, d[int(.025 * nb)], d[int(.975 * nb)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trees', nargs='+', default=PANEL)
    args = p.parse_args()
    acc = build(args.trees)

    def vec(sub, t, k):
        v = acc.get((sub, t, k))
        if v is None:
            return None
        return np.mean(v, axis=0) if isinstance(v, list) else v

    for sub in ('all', 'novel'):
        trees = [t for t in args.trees if (sub, t, 'n') in acc]
        if not trees:
            continue
        print(f"\n{'='*104}\n=== {sub.upper()} post-D substitutions -- hit@10% "
              f"(POSITIVE = better)\n{'='*104}")
        cols = BASELINES + ARMS
        hdr = f"{'tree':9s} {'n':>5s} " + ' '.join(f'{c[:9]:>9s}' for c in cols)
        print(hdr); print('-' * len(hdr))
        for t in trees:
            N = acc[(sub, t, 'n')]
            row = ' '.join(f"{vec(sub,t,c).sum()/N.sum():9.3f}" for c in cols)
            print(f"{t:9s} {N.sum():5.0f} {row}")
        N = np.concatenate([acc[(sub, t, 'n')] for t in trees])
        tot = ' '.join(
            f"{np.concatenate([vec(sub,t,c) for t in trees]).sum()/N.sum():9.3f}"
            for c in cols)
        print(f"{'POOLED':9s} {N.sum():5.0f} {tot}")
        print(f"{'(uniform = 0.100 by construction)'}")

        for a, b in (('joint_panel', 'single'),
                     ('joint_panel', 'where_times_which'),
                     ('joint_panel', 'frequency'),
                     ('joint_all', 'joint_panel'),
                     ('mixture', 'joint_panel')):
            print(f"\n{a} - {b}  ({sub}, hit@10%, positive = {a} better)")
            pa = []
            for t in trees:
                A, B, Nt = vec(sub, t, a), vec(sub, t, b), acc[(sub, t, 'n')]
                if A is None or B is None:
                    continue
                pa.append((A, B, Nt))
                pt, lo_, hi = boot(A, B, Nt)
                flag = 'better' if lo_ > 0 else ('worse' if hi < 0 else '')
                print(f"  {t:9s} {pt:+7.3f} [{lo_:+.3f}, {hi:+.3f}]  {flag}")
            A = np.concatenate([x[0] for x in pa]); B = np.concatenate([x[1] for x in pa])
            Nn = np.concatenate([x[2] for x in pa])
            pt, lo_, hi = boot(A, B, Nn)
            flag = 'better' if lo_ > 0 else ('worse' if hi < 0 else '')
            print(f"  {'POOLED':9s} {pt:+7.3f} [{lo_:+.3f}, {hi:+.3f}]  {flag}")


if __name__ == '__main__':
    main()
