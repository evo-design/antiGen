#!/usr/bin/env python
"""Recall at top 10% as a function of how many seeds are in the model ensemble.

Averaging three seeds' log-probability maps into ONE predictor was the largest
single effect measured on this arm: de novo recall 0.501 -> 0.566 against the
per-seed convention, and all-substitutions 0.813 -> 0.837. Nothing established that
three is the knee, and more seeds is the only lever found so far that does not
depend on gating -- every mixing rule in `gate_search.py` failed to beat the
ensemble, so the ensemble is the thing to push.

This scores K = 1..8 seeds. For each K it averages over several random subsets of
that size, so the curve reflects "K seeds" rather than "these particular K seeds";
at K = 1 and K = 8 there is only one distinct answer and one subset is used.

`where x which` is held fixed at its own 3-seed ensemble, which is the fair
comparator established earlier -- the baseline's seeded minibatch sampler gives it
a real, if small, ensembling benefit of its own (+0.013 on all substitutions).

Both target subsets use the composition-aware de novo rule, and the canonical stop
convention: masked for the ESM-2 ensemble, unmasked for the baseline.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
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
                                positional_marginal_counts,
                                WhereTimesWhichPredictor)
from antiGen.load import load_examples                                     # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import boot_pooled, seen_cells                 # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402
from fuse_forecast import D, DATA, PANEL, load_joint, logp_model           # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def subsets(seeds, k, n_max, rng):
    """Up to n_max distinct size-k subsets of `seeds`."""
    allc = list(itertools.combinations(seeds, k))
    if len(allc) <= n_max:
        return allc
    idx = rng.choice(len(allc), n_max, replace=False)
    return [allc[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--n-subsets', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(',')]
    rng = np.random.default_rng(0)

    models = {s: load_joint(s) for s in seeds}
    store = {}
    for tree in PANEL:
        with contextlib.redirect_stdout(io.StringIO()):
            pre = load_examples(f'{DATA}/examples_{tree}@{D}_train.txt', AL, None,
                                source=tree)
            post = load_examples(f'{DATA}/examples_{tree}@{D}_test.txt', AL, None,
                                 source=tree)
        if not post:
            continue
        L = len(pre[0].aa_idx)
        aa = torch.stack([e.aa_idx for e in post]).to(dev)
        tg = torch.stack([e.mat for e in post]).to(dev)
        novel = torch.tensor(~seen_cells(pre, L), device=dev)
        pm = positional_marginal_counts(pre, L, E)

        lws = []
        for s in (0, 1, 2):
            fit = fit_substitution_baseline(pre, dev, AL, seed=s)
            fac = WhereTimesWhichPredictor(fit.config, pm, fit).to(dev)
            wl, _, wt = collect(fac, post, dev)
            lws.append(torch.tensor(logp_model(wl, wt), device=dev,
                                    dtype=torch.float32))
            del fit, fac
        lw = torch.stack(lws).mean(0)
        lm = {s: torch.tensor(logp_model(collect(models[s], post, dev)[0], wt),
                              device=dev, dtype=torch.float32) for s in seeds}
        store[tree] = (aa, tg, novel, lm, lw)
        print(f'  {tree}', flush=True)

    def score(mapper, mask_stop):
        per, nvec = {}, {}
        for tree, (aa, tg, novel, lm, lw) in store.items():
            sc = mapper(lm, lw)
            for sub, t in (('all', tg),
                           ('novel', tg * novel.unsqueeze(0).to(tg.dtype))):
                if float(t.sum()) == 0:
                    continue
                per[(tree, sub)] = per_example_hits(sc, t, aa,
                                                    mask_stop).cpu().numpy()
                nvec[(tree, sub)] = t.sum((1, 2)).cpu().numpy()
        return per, nvec

    wper, wnv = score(lambda lm, lw: lw, False)

    def report(per, nvec, sub):
        ks = [k for k in per if k[1] == sub]
        est, _, _ = boot_pooled([(per[k], nvec[k]) for k in ks])
        d, lo, hi = boot_pooled([(per[k] - wper[k], nvec[k]) for k in ks])
        return est, d, lo, hi

    print('\n== model ensemble recall at top 10% vs number of seeds ==')
    print(f'{"K":>2s} {"subsets":>8s} {"ALL":>7s} {"vs wxw ens":>22s} '
          f'{"DE NOVO":>8s} {"vs wxw ens":>22s}')
    rows = []
    for k in range(1, len(seeds) + 1):
        combos = subsets(seeds, k, args.n_subsets, rng)
        acc = {}
        for cb in combos:
            per, nvec = score(
                lambda lm, lw, cb=cb: torch.stack([lm[s] for s in cb]).mean(0), True)
            for sub in ('all', 'novel'):
                for key, val in zip(('est', 'd', 'lo', 'hi'), report(per, nvec, sub)):
                    acc.setdefault((sub, key), []).append(val)
        m = {kk: float(np.mean(v)) for kk, v in acc.items()}
        print(f'{k:2d} {len(combos):8d} {m[("all", "est")]:7.3f} '
              f'{m[("all", "d")]:+7.3f} [{m[("all", "lo")]:+.3f},'
              f'{m[("all", "hi")]:+.3f}] {m[("novel", "est")]:8.3f} '
              f'{m[("novel", "d")]:+7.3f} [{m[("novel", "lo")]:+.3f},'
              f'{m[("novel", "hi")]:+.3f}]')
        rows.append(dict(k=k, n_subsets=len(combos),
                         **{f'{a}_{b}': m[(a, b)]
                            for a in ('all', 'novel')
                            for b in ('est', 'd', 'lo', 'hi')}))

    ks = [k for k in wper if k[1] == 'all']
    print(f'\n  wxw (3-seed ens)  ALL '
          f'{boot_pooled([(wper[k], wnv[k]) for k in ks])[0]:.3f}   DE NOVO '
          f'{boot_pooled([(wper[k], wnv[k]) for k in wper if k[1] == "novel"])[0]:.3f}')
    with open(f'{args.out}/seed_curve.json', 'w') as fh:
        json.dump(rows, fh, indent=1)


if __name__ == '__main__':
    main()
