#!/usr/bin/env python
"""Paired bootstrap intervals for the candidate rules from `fuse_forecast.py`.

That script sweeps weights and reports point estimates only. This one takes the
handful of rules worth quoting, keeps their PER-EXAMPLE hit counts, and runs the
same pooled paired bootstrap the figure uses (resample examples within each tree,
then pool), so the contrasts carry intervals.

The rules:
  model (per seed)  the figure's convention -- each seed scored, the three recalls
                    averaged. This is `joint` in ladder.csv.
  model (ensemble)  ONE predictor built from the three seeds by averaging their
                    log-probability maps (a geometric mean of the three). All three
                    arms trained on pre-D data only, so this is causally legal; it is
                    simply a better predictor than any one of them.
  loglin/rank ...   the fused rules whose weights `fuse_forecast.py` chose on the
                    pre-D split, re-evaluated here so they get intervals too.

Weights are hard-coded from that script's causal selection, NOT refitted here.
"""
from __future__ import annotations

import argparse
import contextlib
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

from antiGen.load import load_examples                                     # noqa: E402
from calibrate_forecast import AL                                          # noqa: E402
from collect_figure_metrics import boot_pooled                             # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402
from fuse_forecast import (D, DATA, PANEL, SEEDS, load_joint, mid_ranks,   # noqa: E402
                           prep)

# (name, kind, w_seen, w_unseen) -- weights chosen on the pre-D split by fuse_forecast
RULES = [
    ('loglin gated (sel: all)',   'loglin', 0.80, 0.00),
    ('loglin global (sel: both)', 'loglin', 0.45, 0.45),
    ('rank gated (sel: both)',    'rank',   0.20, 0.20),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {s: load_joint(s) for s in SEEDS}
    per = {}                      # (rung, tree, subset) -> per-example hit array
    nvec = {}                     # (tree, subset) -> per-example target counts

    for tree in PANEL:
        with contextlib.redirect_stdout(io.StringIO()):
            pre = load_examples(f'{DATA}/examples_{tree}@{D}_train.txt', AL, None,
                                source=tree)
            post = load_examples(f'{DATA}/examples_{tree}@{D}_test.txt', AL, None,
                                 source=tree)
        if not post:
            continue
        L = len(pre[0].aa_idx)
        aa, tg, lm, lw, seen = prep(models, post, pre, L)
        novel = ~seen
        subs = {'all': tg, 'novel': tg * novel.unsqueeze(0).to(tg.dtype)}
        subs = {k: v for k, v in subs.items() if float(v.sum()) > 0}
        for sub, t in subs.items():
            nvec[(tree, sub)] = t.sum((1, 2)).cpu().numpy()

        scored = {'wxw': (lw, False), 'model (ensemble)': (lm['ens'], True)}
        for s in SEEDS:
            scored[f'model seed {s}'] = (lm[s], True)
        rm, rw = mid_ranks(lm['ens']), mid_ranks(lw)
        sv = seen.unsqueeze(0)
        for name, kind, a, b in RULES:
            w = torch.where(sv, torch.full_like(lm['ens'], a),
                            torch.full_like(lm['ens'], b))
            sc = ((1 - w) * rm + w * rw) if kind == 'rank' \
                else ((1 - w) * lm['ens'] + w * lw)
            scored[name] = (sc, True)

        for rung, (sc, ms) in scored.items():
            for sub, t in subs.items():
                per[(rung, tree, sub)] = per_example_hits(sc, t, aa, ms).cpu().numpy()
        del aa, tg, lm, lw, seen, rm, rw, scored
        print(f'  {tree}', flush=True)

    # the figure's convention: three per-seed predictors, their recalls averaged
    for tree in PANEL:
        for sub in ('all', 'novel'):
            ks = [(f'model seed {s}', tree, sub) for s in SEEDS]
            if all(k in per for k in ks):
                per[('model (per seed)', tree, sub)] = np.mean([per[k] for k in ks],
                                                               axis=0)

    np.savez_compressed(f'{args.out}/fuse_ci_per_example.npz',
                        **{'|'.join(k): v for k, v in per.items()},
                        **{'n|' + '|'.join(k): v for k, v in nvec.items()})

    rungs = ['wxw', 'model (per seed)', 'model (ensemble)'] + [r[0] for r in RULES]
    out = []
    print(f'\n{"rung":28s} {"ALL":>7s} {"vs wxw":>22s} {"DE NOVO":>8s} {"vs wxw":>22s}')
    for rung in rungs:
        line = f'{rung:28s}'
        rec = {}
        for sub in ('all', 'novel'):
            trees = [t for t in PANEL if (rung, t, sub) in per]
            parts = [(per[(rung, t, sub)], nvec[(t, sub)]) for t in trees]
            est, _, _ = boot_pooled(parts)
            d = [(per[(rung, t, sub)] - per[('wxw', t, sub)], nvec[(t, sub)])
                 for t in trees]
            de, lo, hi = boot_pooled(d)
            line += f' {est:7.3f} {de:+7.3f} [{lo:+.3f},{hi:+.3f}]'
            rec[sub] = dict(recall=est, vs_wxw=de, lo=lo, hi=hi)
        print(line)
        out.append(dict(rung=rung, **rec))

    print(f'\n{"rung":28s} {"ALL vs model(ens)":>24s} {"DE NOVO vs model(ens)":>24s}')
    for rung in rungs:
        line = f'{rung:28s}'
        for sub in ('all', 'novel'):
            trees = [t for t in PANEL if (rung, t, sub) in per]
            d = [(per[(rung, t, sub)] - per[('model (ensemble)', t, sub)],
                  nvec[(t, sub)]) for t in trees]
            de, lo, hi = boot_pooled(d)
            line += f' {de:+8.3f} [{lo:+.3f},{hi:+.3f}]'
        print(line)

    with open(f'{args.out}/fuse_ci.json', 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
