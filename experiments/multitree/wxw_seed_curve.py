#!/usr/bin/env python
"""Does `where x which` keep improving with more seeds, like the model does?

Fairness question. `fit_substitution_baseline` draws minibatches from a seeded
generator, so the 25-scalar substitution fit is seed-dependent and the baseline can
be ensembled exactly as the model is. Going 1 -> 3 seeds was already worth +0.013 on
all substitutions. The model is now being pushed to 8 and 16 seeds, so the baseline
has to be given at least as many before any contrast is quoted -- otherwise part of
the model's margin is just the ensembling the baseline was denied.

This sweeps the baseline's own K with the model held fixed, and reports where it
saturates. Only the substitution matrix varies with seed; the positional marginal is
a pure count and is identical across seeds, so the saturation point is expected to
be early -- but that is a measurement, not an assumption.

Scored with the canonical convention: the baseline's stop column is NOT masked.
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
from fuse_forecast import D, DATA, PANEL, logp_model                       # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--max-seeds', type=int, default=16)
    ap.add_argument('--n-subsets', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.max_seeds))
    rng = np.random.default_rng(0)

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
        lws = {}
        for s in seeds:
            fit = fit_substitution_baseline(pre, dev, AL, seed=s)
            fac = WhereTimesWhichPredictor(fit.config, pm, fit).to(dev)
            wl, _, wt = collect(fac, post, dev)
            lws[s] = torch.tensor(logp_model(wl, wt), device=dev, dtype=torch.float32)
            del fit, fac
        store[tree] = (aa, tg, novel, lws)
        print(f'  {tree}', flush=True)

    print('\n== `where x which` recall at top 10% vs number of seeds ==')
    print(f'{"K":>2s} {"subsets":>8s} {"ALL":>7s} {"DE NOVO":>9s}')
    rows = []
    for k in range(1, len(seeds) + 1):
        allc = list(itertools.combinations(seeds, k))
        if len(allc) > args.n_subsets:
            allc = [allc[i] for i in
                    rng.choice(len(allc), args.n_subsets, replace=False)]
        acc = {}
        for cb in allc:
            per, nvec = {}, {}
            for tree, (aa, tg, novel, lws) in store.items():
                sc = torch.stack([lws[s] for s in cb]).mean(0)
                for sub, t in (('all', tg),
                               ('novel', tg * novel.unsqueeze(0).to(tg.dtype))):
                    if float(t.sum()) == 0:
                        continue
                    per[(tree, sub)] = per_example_hits(sc, t, aa,
                                                        False).cpu().numpy()
                    nvec[(tree, sub)] = t.sum((1, 2)).cpu().numpy()
            for sub in ('all', 'novel'):
                ks = [x for x in per if x[1] == sub]
                acc.setdefault(sub, []).append(
                    boot_pooled([(per[x], nvec[x]) for x in ks])[0])
        m = {s: float(np.mean(v)) for s, v in acc.items()}
        print(f'{k:2d} {len(allc):8d} {m["all"]:7.3f} {m["novel"]:9.3f}')
        rows.append(dict(k=k, n_subsets=len(allc), **m))
    with open(f'{args.out}/wxw_seed_curve.json', 'w') as fh:
        json.dump(rows, fh, indent=1)


if __name__ == '__main__':
    main()
