#!/usr/bin/env python
"""Print the per-tree validation curve written by a multi-tree training run.

Usage:
    python experiments/multitree/summarize_run.py <output_dir_or_metrics.json>

Reads ``<run_prefix>_val_metrics_by_epoch.json`` and prints, per source tree, the
validation NLL per observed mutation and top-q hit rate against the run's reference
points (uniform / codon accessibility / per-position frequency). Epoch -1 is the
zero-shot row, recorded before the first optimizer step.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load_records(target):
    if os.path.isdir(target):
        matches = sorted(glob.glob(os.path.join(target, "*_val_metrics_by_epoch.json")))
        if not matches:
            raise SystemExit(f"No *_val_metrics_by_epoch.json under {target}")
        target = matches[-1]
    with open(target) as handle:
        return target, json.load(handle)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__.strip())
    path, records = load_records(argv[0])
    print(f"{path}\n")

    sources = list(records[0]['val']['per_source'])
    baselines = records[0].get('baselines', {})
    quantile = sorted(records[0]['val']['per_source'][sources[0]]['hit_rate'])[0]

    for source in sources:
        stats0 = records[0]['val']['per_source'][source]
        refs = baselines.get(source, {})
        print(f"── {source}  (L={stats0['seq_len']}, "
              f"{stats0['n_mutations']:.0f} held-out mutations)")
        print(f"   reference  nll/mut   hit@{quantile}")
        print(f"   uniform    {stats0['uniform_nll_per_mutation']:7.4f}   "
              f"{'—':>7}")
        for label in ('codon', 'frequency'):
            ref = refs.get(label)
            if ref:
                print(f"   {label:<9}  {ref['nll_per_mutation']:7.4f}   "
                      f"{ref['hit_rate'][quantile]:7.4f}")
        print(f"   {'epoch':<9}  {'nll/mut':>7}   {'hit@' + quantile:>7}   train_loss")
        for record in records:
            stats = record['val']['per_source'].get(source)
            if stats is None:
                continue
            epoch = 'zero' if record['epoch'] < 0 else str(record['epoch'])
            train_loss = record.get('avg_train_loss')
            train_str = f"{train_loss:.4f}" if train_loss is not None else "—"
            print(f"   {epoch:<9}  {stats['nll_per_mutation']:7.4f}   "
                  f"{stats['hit_rate'][quantile]:7.4f}   {train_str}")
        print()

    best = min(records, key=lambda r: r['val']['overall']['nll_per_mutation'])
    print(f"Best overall val nll/mut: {best['val']['overall']['nll_per_mutation']:.4f} "
          f"at epoch {best['epoch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
