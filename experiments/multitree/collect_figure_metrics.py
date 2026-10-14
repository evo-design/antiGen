#!/usr/bin/env python
"""Score both evaluation regimes on the same nine trees, two metrics, two subsets.

Writes the tidy CSVs behind Figure 5. Everything is computed here rather than read
back from the per-run JSONs, so the two regimes go through byte-identical scoring
code and the same definition of "novel".

Regimes (same nine trees, same evaluation trees):
  random    each tree's random held-out split. Not forecast-valid -- a held-out node's
            parent is usually in the training half.
  temporal  the D = 2025-01-01 holdout: trained on the build as published at D,
            scored on substitutions that did not exist then.
  What is deliberately NOT matched: the random arm trains on today's full builds and
  the temporal arm on the D snapshot. That difference is the comparison, not a confound.

Subsets: `all` post-split substitutions, and `novel` -- those whose (position, alt)
cell never occurs among the training examples the model actually saw. The same rule is
applied in both regimes, so the two columns mean the same thing. (For the temporal
regime this is very slightly different from the published sidecar count, which keyed
novelty off the snapshot *tree* rather than the generated examples; the script prints
both so the gap is visible.)

Metrics: NLL per mutation (lower better) and hit@10% (higher better), the latter with
`compute_hit_rate`'s fractional tie-breaking and the stop column NaN-masked for
predictors whose OUTPUTS_LEGITIMATE_STOP is False.

Epoch rules, both reported because the choice materially changes the effect size:
  pooled    the epoch with the best POOLED validation NLL -- one checkpoint for every
            tree, which is what --save-best ships and what you would actually deploy.
            Conservative for the joint model, which must compromise across trees.
  per-tree  each tree scored at ITS OWN best validation epoch (--save-best-per-source).
            Only the joint arm is affected: a single-tree run has one source, so its
            pooled best epoch already is its per-tree best, and one checkpoint serves
            both rules. Baselines do not depend on the epoch at all.
Selection uses validation data only under both rules, in both regimes.

Usage:
    python experiments/multitree/collect_figure_metrics.py [--out DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import os
import random
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
from calibrate_forecast import collect, temp_probs, AL, E                  # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402

PANEL = ['ncov', 'rsv_a', 'rsv_b', 'h5nx', 'measles',
         'denv1', 'denv2', 'denv3', 'denv4']
SEEDS = [0, 1, 2]
D = '2025-01-01'
DATA = 'experiments/multitree/data'
MATCHED = '/large_storage/hielab/ispecht/multitree/matched'
FORECAST = '/large_storage/hielab/ispecht/multitree/forecast'
# joint_panel retrained with --save-best-per-source; the original forecast/
# runs are left untouched so the already-reported numbers still stand.
FORECAST_PT = '/large_storage/hielab/ispecht/multitree/forecast_pertree'
RUNGS = ['uniform', 'codon', 'where_only', 'pssm', 'substitution', 'frequency',
         'where_times_which', 'single', 'joint']
RULES = ['pooled', 'per-tree']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_dir(tree, seed):
    """Random-split single-tree control, three-way split.

    Only matched/ qualifies: the decomp/ controls used the two-way split, so their
    validation examples overlap this arm's test block and reusing them would leak.
    """
    p = f'{MATCHED}/single_{tree}_s{seed}'
    if not glob.glob(p + '/*_best.pt'):
        raise SystemExit(f'no three-way-split single run for {tree} seed {seed} ({p})')
    return p


def load_model(path):
    cp, cfg = load_checkpoint(sorted(glob.glob(path + '/*_best.pt'))[-1],
                              map_location=dev)
    m = create_model(cfg, dev)
    m.load_state_dict(cp['model_state_dict'], strict=False)
    m.eval()
    return m


def split_random(tree, seed):
    """(seen pool, test) for the random regime.

    Three-way split: the model trained on `train`, chose its epoch on `val`, and is
    reported on `test`, which nothing in training or tuning touched. Baselines are fit
    on train+val -- everything available before the test block -- which mirrors the
    temporal regime, where they are fit on the whole pre-D snapshot file.
    """
    from antiGen.train import held_out_split
    with contextlib.redirect_stdout(io.StringIO()):
        ex = load_examples(f'{DATA}/examples_{tree}_nocutoff.txt', AL, None, source=tree)
        tr, va, te = held_out_split({tree: ex}, 0.1, 0.1, seed)
    return tr[tree] + va[tree], te[tree]


def split_temporal(tree, _seed):
    with contextlib.redirect_stdout(io.StringIO()):
        tr = load_examples(f'{DATA}/forecast/examples_{tree}@{D}_train.txt', AL,
                           None, source=tree)
        te = load_examples(f'{DATA}/forecast/examples_{tree}@{D}_test.txt', AL,
                           None, source=tree)
    return tr, te


def seen_cells(train_ex, L):
    """Cells already present in training, by EITHER route.

    A cell counts as seen if some training branch mutated to that residue there,
    *or* if any training sequence simply carries that residue there.

    The second clause is not redundant. The consensus residue at a position is
    inherited from the root, not mutated into, so it leaves no event -- and a
    post-D reversion from a rare derived state back to consensus would then be
    labeled de novo despite being the single most predictable cell at that
    position. On the D = 2025-01-01 panel that was 12 of 157 event-novel
    targets, every one a reversion to a residue carried by 65-99.5% of the
    training sequences, and a pure composition baseline called all 12 correctly.
    De novo has to mean "this residue has not been seen here", not merely "no
    branch in the training window happened to mutate into it".
    """
    m = np.zeros((L, E), dtype=bool)
    idx = np.arange(L)
    for ex in train_ex:
        m[idx, ex.aa_idx.numpy()] = True                 # carried by this sequence
        for pos, alt in (ex.mat > 0).nonzero().tolist():
            m[pos, alt] = True                           # mutated to on this branch
    return m


def score_tree(model_scores, tg, aa, mask_stop, novel):
    """(loss, hits, n) per example for the `all` and `novel` target subsets."""
    out = {}
    for subset in ('all', 'novel'):
        t = tg if subset == 'all' else tg * torch.tensor(
            novel, device=tg.device, dtype=tg.dtype).unsqueeze(0)
        if float(t.sum()) == 0:
            continue
        probs = model_scores['probs']
        loss = -(t * torch.log(probs.clamp(min=1e-300))).sum((1, 2))
        hits = per_example_hits(model_scores['logits'], t, aa, mask_stop)
        out[subset] = (loss.cpu().numpy(), hits.cpu().numpy(),
                       t.sum((1, 2)).cpu().numpy())
    return out


def _resample(parts, nb, seed):
    """Pooled bootstrap: resample examples within each tree, then pool.

    `parts` is a list of (numerator, denominator) arrays, one per tree. Vectorized
    because the ladder alone needs 128 of these and the pure-Python version was the
    slowest step in the script by an order of magnitude.
    """
    rng = np.random.default_rng(seed)
    num = np.zeros(nb)
    den = np.zeros(nb)
    for a, n in parts:
        idx = rng.integers(0, len(a), size=(nb, len(a)))
        num += a[idx].sum(axis=1)
        den += n[idx].sum(axis=1)
    return num / np.maximum(den, 1e-9)


def boot(A, B, N, nb=10000, sd=0):
    """95% interval for a paired difference, pooled over one tree or many."""
    pt = (A.sum() - B.sum()) / max(N.sum(), 1e-9)
    d = _resample([(A - B, N)], nb, sd)
    return pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_pooled(parts, nb=10000, sd=0):
    """Same, resampling within each tree before pooling across them."""
    num = sum(float(a.sum()) for a, _ in parts)
    den = sum(float(n.sum()) for _, n in parts)
    d = _resample(parts, nb, sd)
    return num / max(den, 1e-9), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='antiGen-figures/data/multitree')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # acc[(regime, tree, subset, rung, metric)] = list over seeds of per-example array
    acc, nvec = {}, {}
    for regime, splitter, joint_root in (
            ('Random holdout', split_random, f'{MATCHED}/joint9'),
            ('Temporal holdout', split_temporal, f'{FORECAST_PT}/joint_panel')):
        for tree in PANEL:
            for seed in SEEDS:
                train_ex, eval_ex = splitter(tree, seed)
                if not eval_ex:
                    continue
                L = len(eval_ex[0].aa_idx)
                aa = torch.stack([e.aa_idx for e in eval_ex]).to(dev)
                tg = torch.stack([e.mat for e in eval_ex]).to(dev)
                novel = ~seen_cells(train_ex, L)

                scores = {}
                fit = fit_substitution_baseline(train_ex, dev, AL)
                fac = WhereTimesWhichPredictor(
                    fit.config, positional_marginal_counts(train_ex, L, E), fit).to(dev)
                cm = create_model(ModelConfig(
                    model_type=ModelType.CODON, seq_len=L,
                    alphabet=AlphabetConfig(amino_acids=AL), codon=CodonConfig()), dev)
                for nm, mdl in (('where_times_which', fac), ('substitution', fit),
                                ('codon', cm)):
                    scores[nm] = (torch.tensor(collect(mdl, eval_ex, dev)[0],
                                               device=dev), False)
                for nm, fn in (('frequency', frequency_baseline_scores),
                               ('where_only', where_only_baseline_scores)):
                    scores[nm] = (torch.log(fn(train_ex, L, E).to(dev)
                                            ).unsqueeze(0).expand(len(eval_ex), -1, -1),
                                  False)
                ## PSSM: residue composition of the training sequences,
                ## log(count + 1) -- the same construction as the repo's
                ## pssm_root baseline, but counted here rather than read from
                ## <data_dir>/pssms/root.csv. That artifact exists only for the
                ## paper's four viruses and, being precomputed over a fixed
                ## training window, has no notion of this arm's train/val/test
                ## split, so reusing it would leak the test block.
                ##
                ## Note what it counts: SEQUENCES carrying each residue, not
                ## substitution EVENTS. A mutation founding a large clade is
                ## counted once by `frequency` and once per descendant here.
                pssm = np.zeros((L, E), dtype=np.float32)
                for ex in train_ex:
                    pssm[np.arange(L), ex.aa_idx.numpy()] += 1.0
                scores['pssm'] = (
                    torch.log1p(torch.tensor(pssm, device=dev)
                                ).unsqueeze(0).expand(len(eval_ex), -1, -1), False)
                scores['uniform'] = (torch.zeros_like(scores['codon'][0]), False)
                del fit, fac, cm

                joint_root_s = f'{joint_root}_s{seed}'
                single_path = (single_dir(tree, seed) if regime.startswith('Random')
                               else f'{FORECAST}/single_{tree}_s{seed}')
                # A single-tree run has one source, so its pooled best epoch IS its
                # per-tree best; one checkpoint serves both rules.
                m = load_model(single_path)
                scores['single'] = (torch.tensor(collect(m, eval_ex, dev)[0],
                                                 device=dev), True)
                del m
                for rule, pat in (('pooled', '*_best.pt'),
                                  ('per-tree', f'*_best_{tree}.pt')):
                    hits = sorted(glob.glob(os.path.join(joint_root_s, pat)))
                    if not hits:
                        raise SystemExit(
                            f'missing {pat} in {joint_root_s} -- was it trained with '
                            f'--save-best-per-source?')
                    cp, cfg = load_checkpoint(hits[-1], map_location=dev)
                    m = create_model(cfg, dev)
                    m.load_state_dict(cp['model_state_dict'], strict=False)
                    m.eval()
                    lo, _, wtm = collect(m, eval_ex, dev)
                    scores[f'joint@{rule}'] = (torch.tensor(lo, device=dev), True)
                    del m

                for rung, (sc, mask_stop) in scores.items():
                    pr = torch.tensor(temp_probs(sc.cpu().numpy(), wtm, 1.0), device=dev)
                    pr = pr / pr.reshape(pr.shape[0], -1).sum(1).clamp(min=1e-30
                                                                      ).view(-1, 1, 1)
                    got = score_tree({'probs': pr, 'logits': sc}, tg, aa, mask_stop,
                                     novel)
                    base, _, rule_of = rung.partition('@')
                    rules = [rule_of] if rule_of else RULES
                    for subset, (loss, hits, n) in got.items():
                        for rl in rules:
                            acc.setdefault((regime, rl, tree, subset, base, 'nll'),
                                           []).append(loss)
                            acc.setdefault((regime, rl, tree, subset, base, 'hit'),
                                           []).append(hits)
                        nvec[(regime, tree, subset)] = n
            print(f"  {regime} / {tree}", flush=True)

    mean = {k: np.mean(v, axis=0) for k, v in acc.items()}

    # ---- ladder.csv
    with open(f'{args.out}/ladder.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval', 'rule', 'subset', 'metric', 'rung', 'value',
                    'lo', 'hi', 'n_mutations'])
        for regime in ('Random holdout', 'Temporal holdout'):
            for rule in RULES:
                for subset in ('all', 'novel'):
                    trees = [t for t in PANEL if (regime, t, subset) in nvec]
                    N = sum(nvec[(regime, t, subset)].sum() for t in trees)
                    if not N:
                        continue
                    for rung in RUNGS:
                        for metric in ('nll', 'hit'):
                            ks = [(regime, rule, t, subset, rung, metric)
                                  for t in trees]
                            if any(k not in mean for k in ks):
                                continue
                            ## NLL is reported as nats better than uniform. That
                            ## is a PAIRED difference -- uniform and the rung score
                            ## the same examples -- so the interval must be taken
                            ## on the difference. Bootstrapping the raw NLL and
                            ## then subtracting uniform's point estimate instead
                            ## carries uniform's own sampling variance, which gives
                            ## the uniform rung a wide interval around a quantity
                            ## that is identically zero.
                            if metric == 'nll':
                                parts = [
                                    (mean[(regime, rule, t, subset, 'uniform',
                                           metric)] - mean[k],
                                     nvec[(regime, t, subset)])
                                    for k, t in zip(ks, trees)]
                            else:
                                parts = [(mean[k], nvec[(regime, t, subset)])
                                         for k, t in zip(ks, trees)]
                            est, lo, hi = boot_pooled(parts)
                            w.writerow([regime, rule, subset, metric, rung,
                                        f'{est:.6f}', f'{lo:.6f}', f'{hi:.6f}',
                                        f'{N:.0f}'])

    # ---- deltas.csv
    with open(f'{args.out}/deltas.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval', 'rule', 'subset', 'metric', 'contrast', 'tree',
                    'est', 'lo', 'hi', 'pooled'])
        for regime in ('Random holdout', 'Temporal holdout'):
          for rule in RULES:
            for subset in ('all', 'novel'):
                for metric in ('nll', 'hit'):
                    for contrast, other in (('vs single-tree model', 'single'),
                                            ('vs where x which', 'where_times_which')):
                        A, B, N = [], [], []
                        for tree in PANEL:
                            ka = (regime, rule, tree, subset, 'joint', metric)
                            kb = (regime, rule, tree, subset, other, metric)
                            if ka not in mean or kb not in mean:
                                continue
                            a, b = mean[ka], mean[kb]
                            n = nvec[(regime, tree, subset)]
                            est, lo, hi = boot(a, b, n)
                            w.writerow([regime, rule, subset, metric, contrast, tree,
                                        f'{est:.6f}', f'{lo:.6f}', f'{hi:.6f}', 0])
                            A.append(a); B.append(b); N.append(n)
                        if not A:
                            continue
                        est, lo, hi = boot_pooled(
                            [(a - b, n) for a, b, n in zip(A, B, N)])
                        w.writerow([regime, rule, subset, metric, contrast, 'POOLED',
                                    f'{est:.6f}', f'{lo:.6f}', f'{hi:.6f}', 1])
    np.savez_compressed(
        f'{args.out}/per_example.npz',
        **{'|'.join(k): v for k, v in mean.items()},
        **{'n|' + '|'.join(k): v for k, v in nvec.items()})
    print(f"wrote {args.out}/ladder.csv, {args.out}/deltas.csv and per_example.npz")


if __name__ == '__main__':
    main()
