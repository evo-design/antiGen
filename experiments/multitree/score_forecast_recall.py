#!/usr/bin/env python
"""Recall-only scoring of a forecast arm against the sealed post-D test set.

Same construction as Ivan's score_forecast.py (baselines fit on the pre-D snapshot half
only, scored through the same wild-type-masked machinery), but reports ONLY the quantile
hit-rate == next-mutation recall (compute_hit_rate, fractional tie-break) at q = 1/5/10/25%,
for the model and every baseline. No NLL. Records recall for baselines too (score_forecast
kept only their NLL), so model-vs-baseline is a recall comparison end to end.

Runs against Ivan's feature-complete antiGen (see run_train_fc.py).

Usage:
    python score_forecast_recall.py <run_dir> --data <forecast_dir> --date 2025-01-01 \
        --trees h5nx ncov ... [--subset all novel] [--output scores_recall.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# In-repo vendored antiGen (gap/stop masking lives here), not Ivan's checkout.

import transformers.utils.import_utils as _iu  # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import antiGen  # noqa: E402

from antiGen.config import (AlphabetConfig, CodonConfig, ModelConfig,  # noqa: E402
                            ModelType, SubstitutionConfig)
from antiGen.eval_utils import (fit_substitution_baseline, fixed_map_metrics,  # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts, validate_per_source,
                                where_only_baseline_scores, WhereTimesWhichPredictor)
from antiGen.load import load_examples  # noqa: E402
from antiGen.models.factory import create_model  # noqa: E402
from antiGen.train import MutationDataset  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
QTILES = (0.01, 0.05, 0.10, 0.25)
BASES = ['codon', 'where_only', 'substitution', 'where_times_which', 'frequency']


def loader_for(examples, bs=8):
    return DataLoader(MutationDataset(examples), batch_size=bs, shuffle=False)


def recall(d, q):
    hr = d.get('hit_rate', {})
    return hr.get(f"{q:g}", float('nan')) if isinstance(hr, dict) else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--data', default='experiments/multitree/data/forecast')
    p.add_argument('--date', default='2025-01-01')
    p.add_argument('--trees', nargs='+', required=True)
    p.add_argument('--subset', nargs='+', default=['all', 'novel'], choices=['all', 'novel'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', default=None)
    p.add_argument('--checkpoint', default=None,
                   help='Explicit checkpoint path (overrides the *_best.pt glob). Used to '
                        'score a joint run with its per-source best (fitness_<prefix>_best_'
                        '<tree>.pt), matching how a single-tree run is scored at its own '
                        'best epoch -- the fair joint-vs-single comparison.')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpts = sorted(p2 for p2 in glob.glob(os.path.join(args.run_dir, "*_best.pt"))
                       if '_best_' not in os.path.basename(p2))
        if not ckpts:
            raise SystemExit(f"No *_best.pt under {args.run_dir} (train with --save-best).")
        ckpt_path = ckpts[-1]
    checkpoint, config = load_checkpoint(ckpt_path, map_location=device)
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"checkpoint: {os.path.basename(ckpt_path)} (epoch {checkpoint.get('epoch')})")
    print(f"forecast date: {args.date}   device: {device}   recall @ {QTILES}\n")

    results = {}
    for subset in args.subset:
        suffix = 'test' if subset == 'all' else 'novel'
        qcols = " ".join(f"r@{int(q*100)}".rjust(7) for q in QTILES)
        print(f"=== {subset.upper()} post-D substitutions  (recall; model vs best baseline)")
        print(f"{'tree':11}{'n':>5}  {'who':>10} {qcols}")
        print('-' * (28 + 8 * len(QTILES)))
        for tree in args.trees:
            test_path = os.path.join(args.data, f"examples_{tree}@{args.date}_{suffix}.txt")
            train_path = os.path.join(args.data, f"examples_{tree}@{args.date}_train.txt")
            if not os.path.exists(test_path) or os.path.getsize(test_path) == 0:
                continue
            test_ex = load_examples(test_path, ALPHABET, None, source=tree)
            train_ex = load_examples(train_path, ALPHABET, None, source=tree)
            if not test_ex:
                continue
            seq_len = len(test_ex[0].aa_idx)
            E = len(ALPHABET)
            loader = loader_for(test_ex, args.batch_size)
            alpha_cfg = AlphabetConfig(amino_acids=ALPHABET)

            refs = {}
            refs['frequency'] = fixed_map_metrics(
                frequency_baseline_scores(train_ex, seq_len, E), loader, device, qtiles=QTILES, mask_gap_stop=True)
            refs['where_only'] = fixed_map_metrics(
                where_only_baseline_scores(train_ex, seq_len, E), loader, device, qtiles=QTILES, mask_gap_stop=True)
            fitted = fit_substitution_baseline(train_ex, device, ALPHABET)
            refs['substitution'] = validate_per_source(
                fitted, {tree: loader}, device, qtiles=QTILES, mask_gap_stop=True)['per_source'][tree]
            factored = WhereTimesWhichPredictor(
                fitted.config, positional_marginal_counts(train_ex, seq_len, E), fitted).to(device)
            refs['where_times_which'] = validate_per_source(
                factored, {tree: loader}, device, qtiles=QTILES, mask_gap_stop=True)['per_source'][tree]
            codon_model = create_model(
                ModelConfig(model_type=ModelType.CODON, seq_len=seq_len,
                            alphabet=alpha_cfg, codon=CodonConfig()), device)
            refs['codon'] = validate_per_source(
                codon_model, {tree: loader}, device, qtiles=QTILES, mask_gap_stop=True)['per_source'][tree]
            del fitted, factored, codon_model

            scored = validate_per_source(model, {tree: loader}, device,
                                         qtiles=QTILES, mask_gap_stop=True)['per_source'][tree]
            n = scored['n_mutations']

            # best baseline per quantile (max recall over baselines)
            best_base = {q: max(recall(refs[k], q) for k in BASES) for q in QTILES}
            results.setdefault(subset, {})[tree] = {
                'n_mutations': n,
                'model': {f"{q:g}": recall(scored, q) for q in QTILES},
                'best_baseline': {f"{q:g}": best_base[q] for q in QTILES},
                'baselines': {k: {f"{q:g}": recall(refs[k], q) for q in QTILES} for k in BASES},
            }
            mrow = " ".join(f"{recall(scored,q):7.3f}" for q in QTILES)
            brow = " ".join(f"{best_base[q]:7.3f}" for q in QTILES)
            print(f"{tree:11}{n:5.0f}  {'model':>10} {mrow}")
            print(f"{'':11}{'':>5}  {'bestBase':>10} {brow}")
        print()

    if args.output:
        json.dump({'run_dir': args.run_dir, 'checkpoint': ckpt_path, 'date': args.date,
                   'qtiles': QTILES, 'results': results}, open(args.output, 'w'), indent=2)
        print(f"wrote {args.output}")


if __name__ == '__main__':
    main()
