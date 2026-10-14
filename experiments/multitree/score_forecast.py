#!/usr/bin/env python
"""Score a trained forecast arm against the sealed post-D test set.

This is the only script that reads the post-D examples. Everything upstream -- training,
epoch selection, arm choice -- sees snapshot data alone, so opening the test set here is
the single moment the future is consulted.

Every baseline is fitted on the **snapshot training half only**, through the same
wild-type-masked machinery as the model, so the rungs are directly comparable:

    uniform            nothing
    codon              the genetic code, data-free
    where_only         which sites mutate, AA axis flattened
    substitution       which residue appears, position-blind
    where_times_which  both, assumed independent
    frequency          the empirical joint from pre-D counts
    model              --

Two subsets are reported. *all* post-D substitutions, and *novel* ones -- absent from the
entire snapshot tree. The novel subset is the forecast that matters: a count-based
baseline cannot score a cell it has never seen above its smoothing floor, so that is
where a sequence model should separate from one.

Usage:
    python experiments/multitree/score_forecast.py <run_dir> \
        --data experiments/multitree/data/forecast --date 2025-01-01 \
        [--trees h5nx ncov ...] [--subset all novel]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

import transformers.utils.import_utils as _import_utils       # noqa: E402
_import_utils._torchvision_available = False                  # see run_train.py
_import_utils._torchvision_version = "N/A"

from antiGen.config import (AlphabetConfig, CodonConfig, ModelConfig,      # noqa: E402
                            ModelType, SubstitutionConfig)
from antiGen.eval_utils import (fit_substitution_baseline, fixed_map_metrics,  # noqa: E402
                                frequency_baseline_scores, load_checkpoint,
                                positional_marginal_counts, validate_per_source,
                                where_only_baseline_scores, WhereTimesWhichPredictor)
from antiGen.load import load_examples                        # noqa: E402
from antiGen.models.factory import create_model               # noqa: E402
from antiGen.train import MutationDataset                     # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
PANEL = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'denv1', 'denv2', 'denv3', 'denv4']


def loader_for(examples, batch_size=8):
    return DataLoader(MutationDataset(examples), batch_size=batch_size, shuffle=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--data', default='experiments/multitree/data/forecast')
    p.add_argument('--date', default='2025-01-01')
    p.add_argument('--trees', nargs='+', default=PANEL)
    p.add_argument('--subset', nargs='+', default=['all', 'novel'],
                   choices=['all', 'novel'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "*_best.pt")))
    if not ckpts:
        raise SystemExit(f"No *_best.pt under {args.run_dir} -- was it trained with "
                         f"--save-best? The final-epoch checkpoint is the overfit model "
                         f"and should not be used here.")
    checkpoint, config = load_checkpoint(ckpts[-1], map_location=device)
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"checkpoint: {ckpts[-1]} (epoch {checkpoint.get('epoch')})")
    print(f"forecast date: {args.date}   device: {device}\n")

    results = {}
    for subset in args.subset:
        suffix = 'test' if subset == 'all' else 'novel'
        header = (f"{'tree':9s} {'n':>5s} {'unif':>7s} {'codon':>7s} {'where':>7s} "
                  f"{'subst':>7s} {'wxw':>7s} {'freq':>7s} {'model':>7s} | {'vs best':>8s}")
        print(f"=== {subset.upper()} post-D substitutions")
        print(header)
        print('-' * len(header))

        for tree in args.trees:
            test_path = os.path.join(args.data,
                                     f"examples_{tree}@{args.date}_{suffix}.txt")
            train_path = os.path.join(args.data,
                                      f"examples_{tree}@{args.date}_train.txt")
            if not os.path.exists(test_path) or os.path.getsize(test_path) == 0:
                continue
            test_ex = load_examples(test_path, ALPHABET, None, source=tree)
            train_ex = load_examples(train_path, ALPHABET, None, source=tree)
            if not test_ex:
                continue
            seq_len = len(test_ex[0].aa_idx)
            E = len(ALPHABET)
            loader = loader_for(test_ex, args.batch_size)

            # Baselines: pre-D counts only.
            refs = {}
            refs['frequency'] = fixed_map_metrics(
                frequency_baseline_scores(train_ex, seq_len, E), loader, device)
            refs['where_only'] = fixed_map_metrics(
                where_only_baseline_scores(train_ex, seq_len, E), loader, device)

            alphabet_cfg = AlphabetConfig(amino_acids=ALPHABET)
            fitted = fit_substitution_baseline(train_ex, device, ALPHABET)
            refs['substitution'] = validate_per_source(
                fitted, {tree: loader}, device)['per_source'][tree]
            factored = WhereTimesWhichPredictor(
                fitted.config, positional_marginal_counts(train_ex, seq_len, E),
                fitted).to(device)
            refs['where_times_which'] = validate_per_source(
                factored, {tree: loader}, device)['per_source'][tree]
            codon_model = create_model(
                ModelConfig(model_type=ModelType.CODON, seq_len=seq_len,
                            alphabet=alphabet_cfg, codon=CodonConfig()), device)
            refs['codon'] = validate_per_source(
                codon_model, {tree: loader}, device)['per_source'][tree]
            del fitted, factored, codon_model

            scored = validate_per_source(model, {tree: loader}, device)['per_source'][tree]

            def nll(key):
                return refs[key].get('nll_per_mutation', float('nan'))

            best_base = min(nll(k) for k in
                            ('codon', 'where_only', 'substitution',
                             'where_times_which', 'frequency'))
            m = scored['nll_per_mutation']
            print(f"{tree:9s} {scored['n_mutations']:5.0f} "
                  f"{scored['uniform_nll_per_mutation']:7.3f} {nll('codon'):7.3f} "
                  f"{nll('where_only'):7.3f} {nll('substitution'):7.3f} "
                  f"{nll('where_times_which'):7.3f} {nll('frequency'):7.3f} "
                  f"{m:7.3f} | {m - best_base:+8.3f}")
            results.setdefault(subset, {})[tree] = {
                'n_mutations': scored['n_mutations'],
                'model': m, 'uniform': scored['uniform_nll_per_mutation'],
                **{k: nll(k) for k in refs},
                'hit_rate': scored['hit_rate'],
                # The test loader is shuffle=False over a fixed file, so index i is the
                # same example in every arm. Kept so arms can be compared with the
                # paired bootstrap rather than only as aggregates.
                'per_example': {
                    'model': scored['per_example'],
                    **{k: refs[k].get('per_example') for k in refs},
                },
            }
        print()

    if args.output:
        with open(args.output, 'w') as handle:
            json.dump({'run_dir': args.run_dir, 'checkpoint': ckpts[-1],
                       'date': args.date, 'results': results}, handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == '__main__':
    main()
