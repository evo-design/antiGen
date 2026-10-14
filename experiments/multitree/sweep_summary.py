#!/usr/bin/env python
"""Aggregate the multi-tree sweep across replicate seeds.

Usage:
    python experiments/multitree/sweep_summary.py <sweep_dir> [--baseline joint3]
        [--arm joint5] [--epoch-rule per-tree|overall]

The sweep directory holds one subdirectory per (configuration, seed), named
``<config>_s<seed>`` — see train_sweep_esm2.sh. For every tree this prints the per-seed
delta in validation NLL per held-out mutation between the arm and its comparator, plus a
delta pooled over seeds with a paired bootstrap CI.

Two comparisons are available:

* ``--baseline single`` (default): arm vs that tree's own single-tree run. Answers
  "does training across trees beat a per-tree model".
* ``--baseline joint3``: arm vs the three-tree joint run, on the trees both saw. Answers
  "did adding the extra trees help the original ones", holding the evaluation set fixed
  and varying only what else was in training.

Within a seed every configuration used the same ``--split-seed``, so each tree's held-out
set is identical across configurations and examples can be differenced one-to-one. Across
seeds the held-out sets differ, so the pooled figure sums numerators and denominators and
the bootstrap resamples examples within each seed.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys


def load(directory):
    matches = sorted(glob.glob(os.path.join(directory, "*_val_metrics_by_epoch.json")))
    if not matches:
        return None
    with open(matches[-1]) as handle:
        return json.load(handle)


def scored_epoch(records, source, rule):
    rows = [r for r in records if r['epoch'] >= 0 and source in r['val']['per_source']]
    if not rows:
        return None
    if rule == 'overall':
        chosen = min(rows, key=lambda r: r['val']['overall']['nll_per_mutation'])
    else:
        chosen = min(rows, key=lambda r: r['val']['per_source'][source]['nll_per_mutation'])
    stats = chosen['val']['per_source'][source]
    return {'epoch': chosen['epoch'], 'nll': stats['nll_per_mutation'],
            'per_example': stats.get('per_example'),
            'n_mutations': stats['n_mutations']}


def pooled_bootstrap(pairs, n_boot=10000, seed=0):
    """Paired bootstrap over (arm, comparator) per-example arrays from several seeds."""
    usable = [(a, b) for a, b in pairs
              if a and b and len(a['loss']) == len(b['loss'])]
    if not usable:
        return None
    num = sum(sum(a['loss']) - sum(b['loss']) for a, b in usable)
    den = sum(sum(a['n_mutations']) for a, b in usable)
    point = num / max(den, 1e-9)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        total_num = total_den = 0.0
        for a, b in usable:
            n = len(a['loss'])
            idx = [rng.randrange(n) for _ in range(n)]
            total_num += sum(a['loss'][i] - b['loss'][i] for i in idx)
            total_den += sum(a['n_mutations'][i] for i in idx)
        draws.append(total_num / max(total_den, 1e-9))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        raise SystemExit(__doc__.strip())

    def take(flag, default):
        if flag in argv:
            i = argv.index(flag)
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        return default

    baseline = take('--baseline', 'single')
    arm = take('--arm', 'joint5')
    rule = take('--epoch-rule', 'per-tree')
    sweep = argv[0]

    seeds = sorted({os.path.basename(d.rstrip('/')).rsplit('_s', 1)[1]
                    for d in glob.glob(os.path.join(sweep, f"{arm}_s*"))})
    if not seeds:
        raise SystemExit(f"No {arm}_s* directories under {sweep}")

    arm_runs = {s: load(os.path.join(sweep, f"{arm}_s{s}")) for s in seeds}
    sources = sorted({src for recs in arm_runs.values() if recs
                      for src in recs[0]['val']['per_source']})

    print(f"arm={arm}  baseline={baseline}  epoch-rule={rule}  seeds={','.join(seeds)}\n")
    print(f"{'tree':<10} " + " ".join(f"{'s' + s:>16}" for s in seeds)
          + f"   {'pooled delta':>14}  95% CI")
    print("-" * (10 + 17 * len(seeds) + 34))

    all_pairs, cell_signs = [], []
    for source in sources:
        cells, pairs = [], []
        for s in seeds:
            recs_arm = arm_runs.get(s)
            comparator = (os.path.join(sweep, f"single_{source}_s{s}")
                          if baseline == 'single' else os.path.join(sweep, f"{baseline}_s{s}"))
            recs_base = load(comparator)
            if not recs_arm or not recs_base:
                cells.append(f"{'—':>16}")
                continue
            a = scored_epoch(recs_arm, source, rule)
            b = scored_epoch(recs_base, source, rule)
            if not a or not b:
                cells.append(f"{'—':>16}")
                continue
            cells.append(f"{a['nll']:7.3f}/{b['nll']:<7.3f}"[:16].rjust(16))
            pairs.append((a['per_example'], b['per_example']))

        all_pairs.extend(pairs)
        for a, b in pairs:
            if a and b and len(a['loss']) == len(b['loss']):
                cell_signs.append(
                    (sum(a['loss']) - sum(b['loss'])) / max(sum(a['n_mutations']), 1e-9))
        interval = pooled_bootstrap(pairs)
        if interval:
            point, low, high = interval
            verdict = ("arm better" if high < 0 else
                       "baseline better" if low > 0 else "indistinguishable")
            print(f"{source:<10} " + " ".join(cells)
                  + f"   {point:+14.3f}  [{low:+.3f}, {high:+.3f}] {verdict}")
        else:
            print(f"{source:<10} " + " ".join(cells) + f"   {'pending':>14}")

    # Each tree alone is underpowered at 100-350 held-out mutations. Pooling the paired
    # differences over all trees and seeds tests the shared effect with everything at once.
    combined = pooled_bootstrap(all_pairs)
    if combined:
        point, low, high = combined
        negative = sum(1 for d in cell_signs if d < 0)
        verdict = ("arm better" if high < 0 else
                   "baseline better" if low > 0 else "indistinguishable")
        print("-" * (10 + 17 * len(seeds) + 34))
        print(f"{'POOLED':<10} " + " ".join(f"{'':>16}" for _ in seeds)
              + f"   {point:+14.3f}  [{low:+.3f}, {high:+.3f}] {verdict}")
        print(f"{'':<10} {negative}/{len(cell_signs)} (tree, seed) cells favor {arm}")

    print(f"\ncells are arm/baseline validation NLL per held-out mutation at each run's "
          f"scored epoch; lower is better.\npooled delta = arm - baseline over all seeds; "
          f"negative favors {arm}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
