#!/usr/bin/env python
"""Why does the model win on NLL zero-shot when it loses on NLL de novo?

NOTES s14 reports the model beating the panel-transferred `substitution` matrix
on NLL (8.096 against 8.488) while losing to it on recall. On the de novo subset
of the temporal holdout -- the closest analogue, since both deny the model any
position-specific memorization -- the sign is the other way round and large:
9.524 against 8.878. Two regimes that should behave alike do not, so one of the
two numbers is measuring something other than what it says.

Two candidate explanations, both about how scores become probabilities rather
than about the model:

**1. `zeroshot_eval.py` does not mask wild-type cells for every rung.** It turns
scores into probabilities with a plain softmax over the whole (L, E) map:

    p = softmax(sc.reshape(B, -1))

For the ESM-2 and codon rungs `sc` is already a `logp_model` map, which zeroed
the wild-type cells before normalizing, so they carry ~1e-8 each and the softmax
is a no-op. The `substitution` rung is `sub_mat[aa]` -- the raw score matrix,
including its DIAGONAL, which is the largest entry in its row (BLOSUM scores a
residue against itself at +4 to +11). So that rung spends a share of its
probability on cells that are wild-type by construction and can never be a
target, and is charged for it. Recall is unaffected: `per_example_hits` NaNs the
wild-type cells itself.

**2. The 1e-8 floor is applied to some rungs and not others.** `logp_model` goes
through `temp_probs`, which clamps the post-softmax map to [1e-8, 1-1e-8]. s11
measured that floor at ~1.06 nats to the model on the de novo subset and ~0 to
`where x which`, because it caps what a single confidently-wrong cell can cost
and the model is far peakier than the baselines. In `zeroshot_eval.py` the ESM-2
and codon rungs carry it; `substitution`, built from the raw matrix, does not.

Both asymmetries push the same way -- they flatter the model against
`substitution` -- which is exactly the direction of the anomaly.

This script rescores the zero-shot panel under three conventions applied
IDENTICALLY to every rung, and reports what the floor is worth to each:

    as-is        reproduces zeroshot_eval.py, to confirm the published numbers
    masked       wild-type cells zeroed, renormalized over the rest, NO floor
    masked+floor the figure's convention: the same, then clamped at 1e-8

and sweeps recall over q rather than reading 10% alone, since a predictor can
be better at the head of the list and worse further down.

Usage:
    python experiments/multitree/zeroshot_probe.py [--out DIR]
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

from antiGen.config import (AlphabetConfig, CodonConfig, ModelConfig,      # noqa: E402
                            ModelType)
from antiGen.eval_utils import load_checkpoint                             # noqa: E402
from antiGen.load import load_examples                                     # noqa: E402
from antiGen.models.factory import create_model                            # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import boot_pooled                             # noqa: E402
from forecast_hit_summary import STOP                                      # noqa: E402
from zeroshot_eval import GENES, ZS, FORECAST, FORECAST_PT, panel_substitution_matrix  # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FLOOR = 1e-8
QS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def load_any(path):
    cp, cfg = load_checkpoint(sorted(glob.glob(path + '/*_best.pt'))[-1],
                              map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def to_logp(logits, wt, mask_wt, floor):
    """Raw scores -> log-probabilities, under one explicit convention.

    `mask_wt` zeroes the wild-type cell at each position before renormalizing, so
    the distribution is over substitutions rather than over "what residue is
    here". `floor` clamps the post-softmax map, capping what one confidently
    wrong cell can cost. Both are applied to every rung or to none; the point of
    this script is that mixing them across rungs is what produced the anomaly.
    """
    B = logits.shape[0]
    x = logits - logits.reshape(B, -1).max(axis=1)[:, None, None]
    p = np.exp(x)
    p = p / p.reshape(B, -1).sum(axis=1)[:, None, None]
    if floor:
        p = np.clip(p * wt, floor, 1 - floor) if mask_wt else np.clip(p, floor,
                                                                     1 - floor)
        p = p / p.reshape(B, -1).sum(axis=1)[:, None, None]
        return np.log(np.clip(p, 1e-300, None))
    # Unfloored: normalize in LOG space. In probability space the denominator
    # underflows to zero whenever one cell holds nearly all the mass -- the
    # ESM-2 arms put 99.2% of theirs on wild-type cells -- and the whole example
    # comes back NaN.
    keep = wt if mask_wt else np.ones_like(wt)
    lp = np.where(keep, np.log(np.clip(p, 1e-300, None)), -np.inf)
    mx = lp.reshape(B, -1).max(axis=1)[:, None, None]
    z = mx + np.log(np.exp(lp - mx).reshape(B, -1).sum(axis=1))[:, None, None]
    return np.clip(lp - z, -700.0, None)


def wt_mass(logits, wt):
    """Share of a plain softmax's mass that lands on wild-type cells."""
    x = logits - logits.reshape(logits.shape[0], -1).max(axis=1)[:, None, None]
    p = np.exp(x)
    p = p / p.reshape(p.shape[0], -1).sum(axis=1)[:, None, None]
    return float((p * ~wt).sum() / p.shape[0])


def hits_at_q(scores, tg, aa, mask_stop, q):
    """compute_hit_rate's fractional tie-breaking at an arbitrary budget q."""
    s = scores.clone()
    if mask_stop:
        s[:, :, STOP] = float('nan')
    s.scatter_(2, aa.unsqueeze(-1), float('nan'))
    s = torch.where(torch.isneginf(s), torch.full_like(s, float('nan')), s)
    B = s.shape[0]
    fl = s.reshape(B, -1).float()
    n_valid = (~torch.isnan(fl)).sum(1).clamp(min=1).float()
    vb = torch.nanquantile(fl, 1.0 - q, dim=1, interpolation='lower').view(B, 1, 1)
    above, at = s > vb, s == vb
    n_above = above.reshape(B, -1).sum(1).float()
    n_at = at.reshape(B, -1).sum(1).float().clamp(min=1)
    frac = ((q * n_valid - n_above) / n_at).clamp(0.0, 1.0)
    return (above * tg).reshape(B, -1).sum(1) + frac * (at * tg).reshape(B, -1).sum(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--joint-seeds', default='0,1,2,3,4,5,6,7')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jseeds = [int(x) for x in args.joint_seeds.split(',')]

    sub_mat = panel_substitution_matrix()
    joint = [load_any(f'{FORECAST_PT}/joint_panel_s{s}') for s in jseeds]
    jall = [load_any(f'{FORECAST}/joint_all_s{s}') for s in (0, 1, 2)]
    print(f'models loaded (joint K={len(joint)})', flush=True)

    CONV = [('as-is (zeroshot_eval)', None, None),
            ('masked, no floor', True, None),
            ('masked + 1e-8 floor', True, FLOOR)]
    nll, hits, nvec, diag = {}, {}, {}, {}

    for gene in GENES:
        with contextlib.redirect_stdout(io.StringIO()):
            ex = load_examples(f'{ZS}/examples_bdbv_{gene}.txt', AL, None,
                               source=gene)
        if not ex:
            continue
        L = len(ex[0].aa_idx)
        aa = torch.stack([e.aa_idx for e in ex]).to(dev)
        tg = torch.stack([e.mat for e in ex]).to(dev)
        nvec[gene] = tg.sum((1, 2)).cpu().numpy()

        cm = create_model(ModelConfig(
            model_type=ModelType.CODON, seq_len=L,
            alphabet=AlphabetConfig(amino_acids=AL), codon=CodonConfig()), dev)
        codon_lo, _, wt = collect(cm, ex, dev)
        del cm

        # Raw, unnormalized scores for the non-ensemble rungs.
        raw = {'uniform': (np.zeros((len(ex), L, E), dtype=np.float32), False),
               'codon': (codon_lo, False),
               'substitution (panel)': (sub_mat[aa].float().cpu().numpy(), False)}
        # Per-seed raw logits for the ensembles; they are averaged in log-prob
        # space, which needs the convention applied FIRST, so they are kept raw
        # here and combined inside the convention loop below.
        ens_raw = {'joint_panel (8-seed ens)': [collect(m, ex, dev)[0] for m in joint],
                   'joint_all (3-seed ens)': [collect(m, ex, dev)[0] for m in jall]}

        for nm, (lo, _) in raw.items():
            diag[(nm, gene, 'wt_mass')] = wt_mass(lo, wt)
        for nm, los in ens_raw.items():
            diag[(nm, gene, 'wt_mass')] = float(np.mean([wt_mass(l, wt) for l in los]))

        for cname, mask, floor in CONV:
            maps = {}
            ## `as-is` is not a convention so much as the absence of one: it
            ## reproduces zeroshot_eval.py, where the ESM-2 and codon rungs had
            ## already been through logp_model (masked AND floored) and the
            ## substitution rung had not. Reproducing it exactly is what lets the
            ## published numbers be checked against the fixed ones.
            for nm, (lo, _) in raw.items():
                if cname.startswith('as-is'):
                    maps[nm] = (to_logp(lo, wt, True, FLOOR) if nm == 'codon'
                                else to_logp(lo, wt, False, None))
                else:
                    maps[nm] = to_logp(lo, wt, mask, floor)
            for nm, los in ens_raw.items():
                m_, f_ = (True, FLOOR) if cname.startswith('as-is') else (mask, floor)
                lps = np.stack([to_logp(l, wt, m_, f_) for l in los]).mean(0)
                # the mean of log-probs is not normalized; renormalize so NLL is
                # a proper density and the ensemble is comparable to the rungs
                z = np.log(np.exp(lps).reshape(lps.shape[0], -1).sum(axis=1))
                maps[nm] = lps - z[:, None, None]

            for nm, lp in maps.items():
                t = torch.tensor(lp, device=dev, dtype=torch.float32)
                nll[(cname, nm, gene)] = (
                    -(tg * t).sum((1, 2)).cpu().numpy())
                if cname == 'masked, no floor':
                    ms = nm.startswith('joint')
                    for q in QS:
                        hits[(nm, gene, q)] = hits_at_q(t, tg, aa, ms,
                                                        q).cpu().numpy()
                    # how much of the map the floor would collapse into one tie
                    fl = to_logp(lp, wt, True, FLOOR)
                    diag[(nm, gene, 'frac_at_floor')] = float(
                        (np.abs(fl - fl.min()) < 1e-6).mean())
                del t
        print(f'  {gene}  L={L}  n_mut={nvec[gene].sum():.0f}', flush=True)

    genes = sorted(nvec)
    rungs = ['uniform', 'codon', 'substitution (panel)',
             'joint_all (3-seed ens)', 'joint_panel (8-seed ens)']

    def pooled(d, key):
        ks = [g for g in genes if key(g) in d]
        return boot_pooled([(d[key(g)], nvec[g]) for g in ks])

    print(f'\n== NLL per mutation, same three rungs, three conventions '
          f'(n={sum(nvec[g].sum() for g in genes):.0f}) ==')
    print(f'{"rung":30s}' + ''.join(f'{c:>24s}' for c, _, _ in CONV))
    for nm in rungs:
        row = ''
        for cname, _, _ in CONV:
            est, lo, hi = pooled(nll, lambda g, c=cname, n=nm: (c, n, g))
            row += f'{est:9.3f} [{lo:.2f},{hi:.2f}]'
        print(f'{nm:30s}{row}')

    print('\n== paired contrast vs `substitution (panel)` '
          '(negative = model better) ==')
    for cname, _, _ in CONV:
        print(f'  {cname}:')
        for nm in ('joint_panel (8-seed ens)', 'joint_all (3-seed ens)'):
            d, lo, hi = boot_pooled(
                [(nll[(cname, nm, g)] - nll[(cname, 'substitution (panel)', g)],
                  nvec[g]) for g in genes])
            star = '' if lo <= 0 <= hi else '  *'
            print(f'    {nm:30s} {d:+7.3f} [{lo:+.3f},{hi:+.3f}]{star}')

    print('\n== what the 1e-8 floor is worth (nats, positive = floor helps) ==')
    for nm in rungs:
        a = pooled(nll, lambda g, n=nm: ('masked, no floor', n, g))[0]
        b = pooled(nll, lambda g, n=nm: ('masked + 1e-8 floor', n, g))[0]
        fr = np.mean([diag[(nm, g, 'frac_at_floor')] for g in genes
                      if (nm, g, 'frac_at_floor') in diag])
        wm = np.mean([diag[(nm, g, 'wt_mass')] for g in genes])
        print(f'{nm:30s} unfloored {a:7.3f}  floored {b:7.3f}  '
              f'worth {a - b:+6.3f}   cells at floor {fr:5.1%}   '
              f'softmax mass on WT {wm:5.1%}')

    print('\n== recall vs budget q (masked, no floor; chance = q) ==')
    print(f'{"rung":30s}' + ''.join(f'{q:>9.0%}' for q in QS))
    rows = []
    for nm in rungs:
        vals = []
        for q in QS:
            est, _, _ = pooled(hits, lambda g, n=nm, qq=q: (n, g, qq))
            vals.append(est)
        print(f'{nm:30s}' + ''.join(f'{v:9.3f}' for v in vals))
        rows.append(dict(rung=nm, **{f'q{q}': v for q, v in zip(QS, vals)}))

    print('\n== recall lift over chance, model minus substitution, by q ==')
    print(f'{"":30s}' + ''.join(f'{q:>9.0%}' for q in QS))
    for nm in ('joint_panel (8-seed ens)', 'joint_all (3-seed ens)'):
        cells = ''
        for q in QS:
            d, lo, hi = boot_pooled(
                [(hits[(nm, g, q)] - hits[('substitution (panel)', g, q)], nvec[g])
                 for g in genes])
            cells += f'{d:+7.3f}{"*" if not lo <= 0 <= hi else " "} '
        print(f'{nm:30s}{cells}')

    with open(f'{args.out}/zeroshot_probe.json', 'w') as fh:
        json.dump({'recall_vs_q': rows,
                   'nll': {f'{c}|{n}': pooled(nll, lambda g, cc=c, nn=n: (cc, nn, g))[0]
                           for c, _, _ in CONV for n in rungs}}, fh, indent=1)


if __name__ == '__main__':
    main()
