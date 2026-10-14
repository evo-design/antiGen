#!/usr/bin/env python
"""The baseline ladder: how much of the model's skill is where, which, or neither.

Every rung is scored through the same wild-type-masked, renormalized machinery as the
model, so the numbers are directly comparable as NLL per held-out mutation:

    uniform          nothing
    where_only       which sites mutate, AA axis flattened to uniform
    substitution     which residue appears, position-blind (codon x BLOSUM62)
    where_times_which  both, assumed independent  <- the reference that matters
    frequency        the empirical joint from the train half (memorization ceiling)
    model            --

The reading:

* model worse than or equal to where_times_which -> it has learned no position-specific
  substitution preference. Where and which, combined independently, is all it has.
* model between where_times_which and frequency -> some joint structure, but still below
  simply memorizing the training counts.
* model better than frequency -> it is generalizing past the training counts.

Usage:
    python experiments/multitree/baseline_ladder.py <run_dir> [--epoch-rule per-tree|overall]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

RUNGS = [
    ('uniform', 'uniform_nll_per_mutation'),
    ('where_only', None),
    ('substitution', None),
    ('where_x_which', None),
    ('frequency', None),
]

# Key in the `baselines.<tree>` block for each rung that has one.
BASELINE_KEY = {
    'where_only': 'where_only',
    'substitution': 'substitution',
    'where_x_which': 'where_times_which',
    'frequency': 'frequency',
}


def load_records(run_dir):
    matches = sorted(glob.glob(os.path.join(run_dir, "*_val_metrics_by_epoch.json")))
    if not matches:
        raise SystemExit(f"No *_val_metrics_by_epoch.json under {run_dir}")
    with open(matches[-1]) as handle:
        return json.load(handle)


def scored_epoch(records, source, rule):
    """The epoch this run is read at for one source: that tree's best, or the pooled best."""
    scored = [r for r in records if r['epoch'] >= 0
              and source in r['val']['per_source']]
    if not scored:
        return None
    if rule == 'overall':
        best = min(scored, key=lambda r: r['val']['overall']['nll_per_mutation'])
    else:
        best = min(scored,
                   key=lambda r: r['val']['per_source'][source]['nll_per_mutation'])
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir')
    parser.add_argument('--epoch-rule', choices=['per-tree', 'overall'], default='per-tree')
    args = parser.parse_args()

    records = load_records(args.run_dir)
    sources = sorted({s for r in records for s in r['val']['per_source']})

    header = (f"{'tree':11s} {'L':>5s} {'unif':>7s} {'where':>7s} {'subst':>7s} "
              f"{'wxw':>7s} {'freq':>7s} {'model':>7s} | {'vs wxw':>7s} {'vs freq':>8s}  verdict")
    print(f"run: {args.run_dir}")
    print(f"epoch rule: {args.epoch_rule}   (NLL per held-out mutation; lower is better)")
    print(header)
    print('-' * len(header))

    missing = False
    for source in sources:
        rec = scored_epoch(records, source, args.epoch_rule)
        if rec is None:
            continue
        stats = rec['val']['per_source'][source]
        refs = rec.get('baselines', {}).get(source, {})
        if 'where_times_which' not in refs:
            missing = True

        def ref(key):
            return refs.get(key, {}).get('nll_per_mutation', float('nan'))

        model = stats['nll_per_mutation']
        wxw, freq = ref('where_times_which'), ref('frequency')
        d_wxw, d_freq = model - wxw, model - freq
        if d_wxw >= 0:
            verdict = 'no joint structure'
        elif d_freq >= 0:
            verdict = 'below memorization'
        else:
            verdict = 'generalizes'

        print(f"{source:11s} {stats['seq_len']:5d} "
              f"{stats['uniform_nll_per_mutation']:7.3f} "
              f"{ref('where_only'):7.3f} {ref('substitution'):7.3f} "
              f"{wxw:7.3f} {freq:7.3f} {model:7.3f} | "
              f"{d_wxw:+7.3f} {d_freq:+8.3f}  {verdict}")

    if missing:
        print("\nNOTE: some trees have no where_times_which rung — that run predates the "
              "baseline. Re-run training to populate it.")


if __name__ == '__main__':
    main()
