#!/usr/bin/env python
"""Zero-shot transfer to a virus family the panel has never seen.

`recent.json` is the ongoing 2026 Bundibugyo ebolavirus outbreak (DRC & Uganda,
762 nodes, 597 tips). It is new on both axes that matter:

  * **New family.** The training panel is coronavirus, pneumovirus (RSV A/B),
    orthomyxovirus (H5Nx, and the frozen seasonal-flu trees in `joint_all`),
    paramyxovirus (measles) and flavivirus (dengue 1-4). A filovirus shares no
    protein with any of them, so nothing positional can transfer -- only whatever
    general substitution structure the model has absorbed.
  * **After D.** Every tip dates 2025.94-2026.63, entirely after the
    D = 2025-01-01 snapshot the arms were trained on, so this is causally clean
    without any further argument.

**What is and is not available zero-shot.** This is the whole design problem, and
the ladder used elsewhere cannot be carried over intact:

  available   uniform and codon (parameter-free); `substitution` (25 position-free
              scalars, fitted on the panel and transferred -- it indexes a 22x22
              score matrix by residue, with no dependence on sequence length);
              and every ESM-2 arm.
  NOT         `where_only`, `pssm`, `frequency` and `where x which` all need
              position-specific counts FROM THIS TREE. There is no such thing as a
              zero-shot positional prior for a protein never seen, so these cannot
              appear in a zero-shot ladder at all.

So the primary table is the zero-shot ladder on every mutation. A second table then
gives the tree-specific baselines a random half of this tree to fit on and scores
everything on the other half -- a deliberately unfair comparison, run to show what
the model's transfer is worth against predictors that did get to see the virus.

Genes: GP (the surface glycoprotein -- the antigenic analogue of the panel's
HA/spike/F/H/E, and 677 aa so it stays off ESM-2's >1022 sliding-window path),
plus NP, VP35, VP30, VP40, VP24 pooled for sample size. L is excluded: 2211 aa puts
it on the two-pass stitching branch that NOTES s9 flags as the suspected cause of
the mpox failure, and that is a confound, not a result.
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
from fuse_forecast import D, DATA, PANEL, logp_model                       # noqa: E402

ZS = 'experiments/multitree/data/zeroshot'
GENES = ['GP', 'NP', 'VP35', 'VP30', 'VP40', 'VP24']
FORECAST = '/large_storage/hielab/ispecht/multitree/forecast'
FORECAST_PT = '/large_storage/hielab/ispecht/multitree/forecast_pertree'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_any(path):
    cp, cfg = load_checkpoint(sorted(glob.glob(path + '/*_best.pt'))[-1],
                              map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def panel_substitution_matrix():
    """A transferable substitution baseline: fit per panel tree, average.

    `fit_substitution_baseline` needs one sequence length per call, so the nine
    trees cannot be pooled into a single fit. But the fitted object is a 22x22
    score matrix with no positional part, so averaging the nine matrices IS the
    pooled baseline, and it applies to any length.
    """
    mats = []
    for tree in PANEL:
        with contextlib.redirect_stdout(io.StringIO()):
            pre = load_examples(f'{DATA}/examples_{tree}@{D}_train.txt', AL, None,
                                source=tree)
        fit = fit_substitution_baseline(pre, dev, AL, seed=0)
        mats.append(fit.score_matrix().detach())
        del fit
    return torch.stack(mats).mean(0)                     # (E, E)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='experiments/multitree/out')
    ap.add_argument('--joint-seeds', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--single-seeds', default='0,1,2')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jseeds = [int(x) for x in args.joint_seeds.split(',')]
    sseeds = [int(x) for x in args.single_seeds.split(',')]

    sub_mat = panel_substitution_matrix()
    print('panel substitution matrix built', flush=True)

    joint = [load_any(f'{FORECAST_PT}/joint_panel_s{s}') for s in jseeds]
    jall = [load_any(f'{FORECAST}/joint_all_s{s}') for s in (0, 1, 2)]
    singles = {t: [load_any(f'{FORECAST}/single_{t}_s{s}') for s in sseeds]
               for t in PANEL}
    print('models loaded', flush=True)

    per, nvec, per2, nvec2 = {}, {}, {}, {}
    for gene in GENES:
        with contextlib.redirect_stdout(io.StringIO()):
            ex = load_examples(f'{ZS}/examples_bdbv_{gene}.txt', AL, None,
                               source=gene)
        if not ex:
            continue
        L = len(ex[0].aa_idx)
        aa = torch.stack([e.aa_idx for e in ex]).to(dev)
        tg = torch.stack([e.mat for e in ex]).to(dev)

        def ens(models, exx, aaa):
            _, _, wt = collect(models[0], exx, dev)
            return torch.stack([
                torch.tensor(logp_model(collect(m, exx, dev)[0], wt), device=dev,
                             dtype=torch.float32) for m in models]).mean(0)

        cm = create_model(ModelConfig(
            model_type=ModelType.CODON, seq_len=L,
            alphabet=AlphabetConfig(amino_acids=AL), codon=CodonConfig()), dev)
        _, _, wt = collect(cm, ex, dev)

        ## EVERY rung goes through `logp_model`, which zeroes the wild-type cell
        ## before normalizing. Routing only some of them through it is a real
        ## error with a large effect: `sub_mat` includes its DIAGONAL, the
        ## largest entry in each row, so an unmasked substitution rung spends
        ## ~48% of its probability on cells that are wild type by construction
        ## and can never be a target -- and is charged 0.655 nats for it. That
        ## alone reversed this script's headline NLL result (the model read
        ## -0.392 against the substitution matrix and actually loses by +0.263).
        ## Recall never saw the bug, because `per_example_hits` NaNs the
        ## wild-type cells itself, which is why the two metrics disagreed.
        rungs = {
            'uniform': (torch.tensor(
                logp_model(np.zeros((len(ex), L, E), dtype=np.float32), wt),
                device=dev, dtype=torch.float32), False),
            'codon': (torch.tensor(logp_model(collect(cm, ex, dev)[0], wt),
                                   device=dev, dtype=torch.float32), False),
            'substitution (panel-transferred)': (
                torch.tensor(logp_model(sub_mat[aa].float().cpu().numpy(), wt),
                             device=dev, dtype=torch.float32), False),
            'joint_panel (8-seed ens)': (ens(joint, ex, aa), True),
            'joint_all (3-seed ens)': (ens(jall, ex, aa), True),
        }
        for t in PANEL:
            rungs[f'single_{t} ({len(sseeds)}-seed ens)'] = (ens(singles[t], ex, aa),
                                                             True)
        for nm, (sc, ms) in rungs.items():
            per[(nm, gene)] = per_example_hits(sc, tg, aa, ms).cpu().numpy()
            p = torch.softmax(sc.reshape(len(ex), -1).float(), 1).reshape(sc.shape)
            per[('NLL|' + nm, gene)] = (
                -(tg * torch.log(p.clamp(min=1e-30))).sum((1, 2)).cpu().numpy())
        nvec[gene] = tg.sum((1, 2)).cpu().numpy()

        # ---- second table: give the tree-specific baselines half of this tree
        with contextlib.redirect_stdout(io.StringIO()):
            trb, vab = split_examples({gene: ex}, 'random', 0.5, 0)
        tr, te = trb[gene], vab[gene]
        if te and tr:
            aa2 = torch.stack([e.aa_idx for e in te]).to(dev)
            tg2 = torch.stack([e.mat for e in te]).to(dev)
            fit = fit_substitution_baseline(tr, dev, AL, seed=0)
            fac = WhereTimesWhichPredictor(
                fit.config, positional_marginal_counts(tr, L, E), fit).to(dev)
            w2, _, wt2 = collect(fac, te, dev)
            pssm = np.zeros((L, E), dtype=np.float32)
            for e in tr:
                pssm[np.arange(L), e.aa_idx.numpy()] += 1.0
            r2 = {
                'where x which (fit on half this tree)': (
                    torch.tensor(logp_model(w2, wt2), device=dev,
                                 dtype=torch.float32), False),
                'frequency (fit on half this tree)': (
                    torch.log(frequency_baseline_scores(tr, L, E).to(dev)
                              ).unsqueeze(0).expand(len(te), -1, -1), False),
                'where_only (fit on half this tree)': (
                    torch.log(where_only_baseline_scores(tr, L, E).to(dev)
                              ).unsqueeze(0).expand(len(te), -1, -1), False),
                'pssm (fit on half this tree)': (
                    torch.log1p(torch.tensor(pssm, device=dev)
                                ).unsqueeze(0).expand(len(te), -1, -1), False),
                'joint_panel ZERO-SHOT': (ens(joint, te, aa2), True),
                'substitution (panel-transferred)  ': (
                    torch.tensor(logp_model(sub_mat[aa2].float().cpu().numpy(), wt2),
                                 device=dev, dtype=torch.float32), False),
            }
            for nm, (sc, ms) in r2.items():
                per2[(nm, gene)] = per_example_hits(sc, tg2, aa2, ms).cpu().numpy()
            nvec2[gene] = tg2.sum((1, 2)).cpu().numpy()
            del fit, fac
        print(f'  {gene}  L={L}  n_mut={nvec[gene].sum():.0f}', flush=True)

    def table(per, nvec, rungs, title, genes):
        print(f'\n== {title} ==')
        tot = sum(nvec[g].sum() for g in genes)
        print(f'{"rung":38s} {"recall@10%":>18s} {"GP only":>9s}   (n={tot:.0f} '
              f'mutations; chance = 0.100)')
        rows = []
        for nm in rungs:
            ks = [g for g in genes if (nm, g) in per]
            if not ks:
                continue
            est, lo, hi = boot_pooled([(per[(nm, g)], nvec[g]) for g in ks])
            gp = (boot_pooled([(per[(nm, 'GP')], nvec['GP'])])[0]
                  if (nm, 'GP') in per else float('nan'))
            print(f'{nm:38s} {est:6.3f} [{lo:.3f},{hi:.3f}] {gp:9.3f}')
            rows.append(dict(rung=nm, recall=est, lo=lo, hi=hi, gp=gp))
        return rows

    genes = [g for g in GENES if g in nvec]
    order = (['uniform', 'codon', 'substitution (panel-transferred)']
             + [f'single_{t} ({len(sseeds)}-seed ens)' for t in PANEL]
             + ['joint_all (3-seed ens)', 'joint_panel (8-seed ens)'])
    r1 = table(per, nvec, order, 'ZERO-SHOT on Bundibugyo ebolavirus, all genes '
                                 'pooled', genes)

    print('\n== the same rungs by NLL per mutation (lower better) ==')
    for nm in order:
        ks = [g for g in genes if ('NLL|' + nm, g) in per]
        if not ks:
            continue
        est, lo, hi = boot_pooled([(per[('NLL|' + nm, g)], nvec[g]) for g in ks])
        print(f'{nm:38s} {est:7.3f} [{lo:.3f},{hi:.3f}]')

    g2 = [g for g in GENES if g in nvec2]
    r2 = table(per2, nvec2, list(dict.fromkeys(
        ['joint_panel ZERO-SHOT', 'substitution (panel-transferred)  ',
         'where x which (fit on half this tree)',
         'frequency (fit on half this tree)', 'pssm (fit on half this tree)',
         'where_only (fit on half this tree)'])),
        'NOT a fair comparison: the baselines were fitted on a random half of this '
        'tree,\n   the model still has never seen any of it. Scored on the other '
        'half', g2)

    # Marginal intervals on two rungs cannot settle which is better -- they score
    # the SAME mutations, so the comparison has to be paired.
    for base in ('joint_panel (8-seed ens)', 'substitution (panel-transferred)'):
        print(f'\n== paired contrast vs `{base}` (positive = row is better) ==')
        for nm in order:
            if nm == base:
                continue
            ks = [g for g in genes if (nm, g) in per and (base, g) in per]
            if not ks:
                continue
            d, lo, hi = boot_pooled([(per[(nm, g)] - per[(base, g)], nvec[g])
                                     for g in ks])
            star = '' if lo <= 0 <= hi else '  *'
            print(f'{nm:38s} {d:+6.3f} [{lo:+.3f},{hi:+.3f}]{star}')

    np.savez_compressed(f'{args.out}/zeroshot_per_example.npz',
                        **{'|'.join(k): v for k, v in per.items()},
                        **{'n|' + g: v for g, v in nvec.items()})
    with open(f'{args.out}/zeroshot.json', 'w') as fh:
        json.dump({'zero_shot': r1, 'half_fit': r2}, fh, indent=1)


if __name__ == '__main__':
    main()
