#!/usr/bin/env python
"""Two questions that decide whether a learned per-cell gate is worth building.

1. **Is the `where x which` comparison fair?** The model rung is now a 3-seed
   ensemble, and `fit_substitution_baseline` carries a seeded minibatch sampler that
   every caller leaves at seed=0 -- so the baseline has been fit once while the model
   gets three runs averaged. This refits it at seeds 0,1,2 and ensembles it the same
   way (mean log-prob), so any ensembling benefit is given to both sides. If the
   baseline's seed variance is negligible the contrast was already fair, but that is
   a measurement, not an assumption.

2. **How much is a gate worth at the honest budget?** `rank_agreement.py` found the
   union of the two top-10% lists reaches 0.660 on de novo -- but it spends 1.46x
   the budget, so it is not a rule anyone could deploy. The deployable ceilings are:

     per-genotype oracle   for each genotype, take whichever predictor does better
                           on it. The ceiling for a gate that picks ONE predictor
                           per query. Upper bound: it uses the answer to choose.
     per-cell oracle       the union bound, reported for scale with its realized
                           budget attached.

   If the per-genotype oracle sits near the ensemble alone, there is nothing for a
   gate to learn and mixing is finished. If it sits well above, the complementarity
   is addressable and a learned gate has room.

Every oracle here is an UPPER BOUND and may not be quoted as a result.
"""
from __future__ import annotations

import argparse
import contextlib
import io
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
from fuse_forecast import D, DATA, PANEL, SEEDS, load_joint, logp_model    # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = {s: load_joint(s) for s in SEEDS}
    per, nvec = {}, {}

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

        # --- the baseline, refit at three seeds and ensembled the same way
        lws = []
        for s in SEEDS:
            fit = fit_substitution_baseline(pre, dev, AL, seed=s)
            fac = WhereTimesWhichPredictor(fit.config, pm, fit).to(dev)
            wl, _, wt = collect(fac, post, dev)
            lws.append(torch.tensor(logp_model(wl, wt), device=dev,
                                    dtype=torch.float32))
            del fit, fac
        lw_ens = torch.stack(lws).mean(0)

        lms = [torch.tensor(logp_model(collect(models[s], post, dev)[0], wt),
                            device=dev, dtype=torch.float32) for s in SEEDS]
        lm_ens = torch.stack(lms).mean(0)

        rungs = {'wxw seed 0': (lws[0], False), 'wxw (3-seed ens)': (lw_ens, False),
                 'model (3-seed ens)': (lm_ens, True)}
        for sub, t in (('all', tg),
                       ('novel', tg * novel.unsqueeze(0).to(tg.dtype))):
            if float(t.sum()) == 0:
                continue
            nvec[(tree, sub)] = t.sum((1, 2)).cpu().numpy()
            for nm, (sc, ms) in rungs.items():
                per[(nm, tree, sub)] = per_example_hits(sc, t, aa, ms).cpu().numpy()
            hm = per[('model (3-seed ens)', tree, sub)]
            hw = per[('wxw (3-seed ens)', tree, sub)]
            # ORACLE: per genotype, whichever predictor did better on it
            per[('ORACLE per-genotype gate', tree, sub)] = np.maximum(hm, hw)
            # ORACLE: the union of the two top-10% lists (1.46x budget)
            per[('ORACLE union (1.46x budget)', tree, sub)] = np.minimum(
                hm + hw, nvec[(tree, sub)])
        del aa, tg, lms, lws, lm_ens, lw_ens
        print(f'  {tree}', flush=True)

    order = ['wxw seed 0', 'wxw (3-seed ens)', 'model (3-seed ens)',
             'ORACLE per-genotype gate', 'ORACLE union (1.46x budget)']
    print(f'\n{"rung":30s} {"ALL":>7s} {"vs wxw ens":>22s} '
          f'{"DE NOVO":>8s} {"vs wxw ens":>22s}')
    for rung in order:
        line = f'{rung:30s}'
        for sub in ('all', 'novel'):
            trees = [t for t in PANEL if (rung, t, sub) in per]
            est, _, _ = boot_pooled([(per[(rung, t, sub)], nvec[(t, sub)])
                                     for t in trees])
            d = [(per[(rung, t, sub)] - per[('wxw (3-seed ens)', t, sub)],
                  nvec[(t, sub)]) for t in trees]
            de, lo, hi = boot_pooled(d)
            line += f' {est:7.3f} {de:+7.3f} [{lo:+.3f},{hi:+.3f}]'
        print(line)

    print(f'\n{"":30s} {"ALL vs model ens":>24s} {"DE NOVO vs model ens":>24s}')
    for rung in order:
        line = f'{rung:30s}'
        for sub in ('all', 'novel'):
            trees = [t for t in PANEL if (rung, t, sub) in per]
            d = [(per[(rung, t, sub)] - per[('model (3-seed ens)', t, sub)],
                  nvec[(t, sub)]) for t in trees]
            de, lo, hi = boot_pooled(d)
            line += f' {de:+8.3f} [{lo:+.3f},{hi:+.3f}]'
        print(line)

    np.savez_compressed(f'{args.out}/gate_ceiling.npz',
                        **{'|'.join(k): v for k, v in per.items()},
                        **{'n|' + '|'.join(k): v for k, v in nvec.items()})


if __name__ == '__main__':
    main()
