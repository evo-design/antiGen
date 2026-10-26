#!/usr/bin/env python
"""Score the held-out post-cutoff tree: the wide corpus against its panel control.

`eval_tree.json` is the ongoing 2026 Bundibugyo ebolavirus outbreak (DRC & Uganda,
1048 nodes, 813 tips). **Every tip dates 2026.330-2026.697**, i.e. after the
D = 2026-05-01 cutoff by construction -- the earliest is one day past it -- so nothing
in it can have reached training through either inclusion route. Only GP is scored: the
surface glycoprotein, the antigenic analogue of the corpus's HA/spike/F/H/E, 676 aa so
it stays off ESM-2's >1022 sliding-window path (which NOTES s19 now shows is where mpox
at 1881 fails outright).

**What this comparison is, and what it is not.** `joint_wide` trained on 43 trees,
three of which are ebolavirus GP -- ebola/all-outbreaks and ebola/ebov-2013 (both Zaire
ebolavirus) and ebola/sudv (Sudan ebolavirus). `joint_panel` trained on nine trees, none
of them a filovirus. So a `joint_wide` win here is NOT "more data helps" in the abstract:
the wide corpus contains the evaluation virus's genus, at the same gene, and the control
does not. That is a fair consequence of widening the corpus -- a wider corpus is more
likely to contain a relative -- but it is not the same claim, and NOTES s14's framing of
this tree as "a virus family the panel has never seen" survives only for `joint_panel`.
The training trees remain causally clean: all three ebola builds are pre-cutoff (two
archived at the cutoff, sudv's live build with every tip <= 2025.05).

**The ladder** is the one NOTES s14 established and s16 corrected, which is as much of
the temporal-split ladder as this setting admits. `where_only`, `pssm`, `frequency` and
`where x which` all need position-specific counts from the tree being scored; in the
temporal split those came from the tree's own pre-D half, and here there is no pre-D
half -- every tip is post-cutoff. So they cannot appear in the primary table at all. The
second table gives them half of this tree to fit on and scores everything on the other
half, a deliberately unfair comparison that says what the model's transfer is worth
against predictors that did get to see the virus.

**Every rung goes through `logp_model`**, which zeroes the wild-type cell before
normalizing (s16: routing only some rungs through it put ~48% of the substitution
matrix's probability on cells that are wild type by construction and reversed that
script's headline NLL result).

Usage:
    python experiments/multitree/eval_wide.py [--gene GP] [--seeds 0,1,2]
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
from antiGen.eval_utils import (fit_substitution_baseline,                 # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts,
                                where_only_baseline_scores,
                                WhereTimesWhichPredictor)
from antiGen.load import load_examples                                     # noqa: E402
from antiGen.models.factory import create_model                            # noqa: E402
from antiGen.train import split_examples                                   # noqa: E402
from calibrate_forecast import collect, AL, E                              # noqa: E402
from collect_figure_metrics import boot_pooled                             # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402
from fuse_forecast import logp_model                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, 'data', 'eval2026')
WIDE = os.path.join(HERE, 'data', 'wide')
TRAIN = '/large_storage/hielab/ispecht/multitree/wide_train'
MANIFEST = os.path.join(WIDE, 'corpus@2026-05-01.json')
PANEL = ['ncov', 'rsv_a_f', 'rsv_b_f', 'measles_h', 'denv1', 'denv2', 'denv3',
         'denv4', 'h5nx_ha']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_arm(arm, seed):
    return sorted(glob.glob(f'{TRAIN}/{arm}_s{seed}/*_best.pt'))


def load_any(arm, seed):
    paths = [p for p in load_arm(arm, seed) if p.endswith('_best.pt')
             and '_best_' not in os.path.basename(p)]
    cp, cfg = load_checkpoint(paths[-1], map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def substitution_matrix(trees):
    """Position-free 22x22 score matrix, fitted per tree and averaged.

    `fit_substitution_baseline` takes one sequence length per call, so trees cannot be
    pooled into a single fit -- but the fitted object has no positional part, so the
    mean of the per-tree matrices IS the pooled baseline, and it applies at any length.
    """
    manifest = json.load(open(MANIFEST))
    mats = []
    for short in trees:
        with contextlib.redirect_stdout(io.StringIO()):
            ex = load_examples(manifest[short]['path'], AL, None, source=short)
        fit = fit_substitution_baseline(ex, dev, AL, seed=0)
        mats.append(fit.score_matrix().detach())
        del fit
    return torch.stack(mats).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gene', default='GP')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    os.makedirs(args.out, exist_ok=True)

    manifest = json.load(open(MANIFEST))
    wide_trees = sorted(manifest)
    print(f'corpus: {len(wide_trees)} trees; panel control: {len(PANEL)}', flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        ex = load_examples(f'{EVAL}/examples_evaltree_{args.gene}.txt', AL, None,
                           source=args.gene)
    L = len(ex[0].aa_idx)
    aa = torch.stack([e.aa_idx for e in ex]).to(dev)
    tg = torch.stack([e.mat for e in ex]).to(dev)
    n_mut = float(tg.sum().item())
    print(f'{args.gene}: {len(ex)} examples, L={L}, {n_mut:.0f} target mutations',
          flush=True)

    sub_panel = substitution_matrix(PANEL)
    sub_wide = substitution_matrix(wide_trees)
    print('substitution matrices built', flush=True)

    arms = {f'joint_wide ({len(seeds)}-seed ens)': [load_any('joint_wide', s)
                                                    for s in seeds],
            f'joint_panel ({len(seeds)}-seed ens)': [load_any('joint_panel', s)
                                                     for s in seeds]}
    print('models loaded', flush=True)

    def ens(models, exx):
        _, _, wt = collect(models[0], exx, dev)
        return torch.stack([
            torch.tensor(logp_model(collect(m, exx, dev)[0], wt), device=dev,
                         dtype=torch.float32) for m in models]).mean(0)

    cm = create_model(ModelConfig(
        model_type=ModelType.CODON, seq_len=L,
        alphabet=AlphabetConfig(amino_acids=AL), codon=CodonConfig()), dev)
    _, _, wt = collect(cm, ex, dev)

    rungs = {
        'uniform': (torch.tensor(
            logp_model(np.zeros((len(ex), L, E), dtype=np.float32), wt),
            device=dev, dtype=torch.float32), False),
        'codon': (torch.tensor(logp_model(collect(cm, ex, dev)[0], wt),
                               device=dev, dtype=torch.float32), False),
        'substitution (panel-transferred)': (
            torch.tensor(logp_model(sub_panel[aa].float().cpu().numpy(), wt),
                         device=dev, dtype=torch.float32), False),
        'substitution (wide-transferred)': (
            torch.tensor(logp_model(sub_wide[aa].float().cpu().numpy(), wt),
                         device=dev, dtype=torch.float32), False),
    }
    for nm, models in arms.items():
        rungs[nm] = (ens(models, ex), True)

    hits, nlls, per_ex = {}, {}, {}
    for nm, (sc, ms) in rungs.items():
        h = per_example_hits(sc, tg, aa, ms).cpu().numpy()
        p = torch.softmax(sc.reshape(len(ex), -1).float(), 1).reshape(sc.shape)
        nll = -(tg * torch.log(p.clamp(min=1e-30))).sum((1, 2)).cpu().numpy()
        hits[nm], nlls[nm], per_ex[nm] = h, nll, (h, nll)

    nvec = tg.sum((1, 2)).cpu().numpy()
    print(f'\n== zero-shot ladder, {args.gene} only '
          f'(n={n_mut:.0f} mutations; chance recall@10% = 0.100) ==')
    print(f'{"rung":36s} {"recall@10%":>22s} {"NLL/mut":>9s}')
    for nm in rungs:
        est, lo, hi = boot_pooled([(hits[nm], nvec)])
        print(f'{nm:36s} {est:8.3f} [{lo:.3f}, {hi:.3f}] {nlls[nm].sum()/n_mut:9.3f}')

    # paired bootstrap, wide - panel, resampling examples
    w = f'joint_wide ({len(seeds)}-seed ens)'
    p_ = f'joint_panel ({len(seeds)}-seed ens)'
    rng = np.random.default_rng(0)
    dh, dn = [], []
    for _ in range(10000):
        idx = rng.integers(0, len(ex), len(ex))
        n = nvec[idx].sum()
        if n == 0:
            continue
        dh.append((hits[w][idx].sum() - hits[p_][idx].sum()) / n)
        dn.append((nlls[w][idx].sum() - nlls[p_][idx].sum()) / n)
    dh, dn = np.array(dh), np.array(dn)
    obs_h = (hits[w].sum() - hits[p_].sum()) / nvec.sum()
    obs_n = (nlls[w].sum() - nlls[p_].sum()) / n_mut
    print(f'\nwide - panel, paired over examples (10k resamples):')
    print(f'  recall@10%  {obs_h:+.3f} [{np.percentile(dh,2.5):+.3f}, '
          f'{np.percentile(dh,97.5):+.3f}]   (positive = wide better)')
    print(f'  NLL/mut     {obs_n:+.3f} [{np.percentile(dn,2.5):+.3f}, '
          f'{np.percentile(dn,97.5):+.3f}]   (negative = wide better)')

    # ---- second table: the positional baselines get half this tree
    with contextlib.redirect_stdout(io.StringIO()):
        trb, vab = split_examples({args.gene: ex}, 'random', 0.5, 0)
    tr, te = trb[args.gene], vab[args.gene]
    if tr and te:
        aa2 = torch.stack([e.aa_idx for e in te]).to(dev)
        tg2 = torch.stack([e.mat for e in te]).to(dev)
        n2 = float(tg2.sum().item())
        fit = fit_substitution_baseline(tr, dev, AL, seed=0)
        fac = WhereTimesWhichPredictor(
            fit.config, positional_marginal_counts(tr, L, E), fit).to(dev)
        w2, _, wt2 = collect(fac, te, dev)
        pssm = np.zeros((L, E), dtype=np.float32)
        for e in tr:
            pssm[np.arange(L), e.aa_idx.numpy()] += 1.0
        r2 = {
            'where x which (half this tree)': (
                torch.tensor(logp_model(w2, wt2), device=dev,
                             dtype=torch.float32), False),
            'frequency (half this tree)': (
                torch.log(frequency_baseline_scores(tr, L, E).to(dev)
                          ).unsqueeze(0).expand(len(te), -1, -1), False),
            'where_only (half this tree)': (
                torch.log(where_only_baseline_scores(tr, L, E).to(dev)
                          ).unsqueeze(0).expand(len(te), -1, -1), False),
            'pssm (half this tree)': (
                torch.log1p(torch.tensor(pssm, device=dev)
                            ).unsqueeze(0).expand(len(te), -1, -1), False),
            'substitution (wide-transferred)': (
                torch.tensor(logp_model(sub_wide[aa2].float().cpu().numpy(), wt2),
                             device=dev, dtype=torch.float32), False),
        }
        for nm, models in arms.items():
            r2[nm + ' ZERO-SHOT'] = (ens(models, te), True)
        nvec2 = tg2.sum((1, 2)).cpu().numpy()
        print(f'\n== the positional baselines get half this tree; scored on the other '
              f'half (n={n2:.0f}) ==')
        print(f'{"rung":36s} {"recall@10%":>22s} {"NLL/mut":>9s}')
        out2 = {}
        for nm, (sc, ms) in r2.items():
            h = per_example_hits(sc, tg2, aa2, ms).cpu().numpy()
            p = torch.softmax(sc.reshape(len(te), -1).float(), 1).reshape(sc.shape)
            nll = -(tg2 * torch.log(p.clamp(min=1e-30))).sum((1, 2)).cpu().numpy()
            est, lo, hi = boot_pooled([(h, nvec2)])
            out2[nm] = dict(recall=float(est), lo=float(lo), hi=float(hi),
                            nll=float(nll.sum() / n2))
            print(f'{nm:36s} {est:8.3f} [{lo:.3f}, {hi:.3f}] {nll.sum()/n2:9.3f}')
    else:
        out2 = {}

    result = dict(
        gene=args.gene, n_examples=len(ex), seq_len=L, n_mutations=n_mut,
        seeds=seeds,
        zero_shot={nm: dict(recall=float(boot_pooled([(hits[nm], nvec)])[0]),
                            lo=float(boot_pooled([(hits[nm], nvec)])[1]),
                            hi=float(boot_pooled([(hits[nm], nvec)])[2]),
                            nll=float(nlls[nm].sum() / n_mut)) for nm in rungs},
        wide_minus_panel=dict(
            recall=float(obs_h), recall_lo=float(np.percentile(dh, 2.5)),
            recall_hi=float(np.percentile(dh, 97.5)), nll=float(obs_n),
            nll_lo=float(np.percentile(dn, 2.5)),
            nll_hi=float(np.percentile(dn, 97.5))),
        half_fit=out2)
    path = os.path.join(args.out, f'eval_wide_{args.gene}.json')
    with open(path, 'w') as fh:
        json.dump(result, fh, indent=1)
    print(f'\n-> {path}')


if __name__ == '__main__':
    main()
