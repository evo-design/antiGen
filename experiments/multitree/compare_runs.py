#!/usr/bin/env python
"""Compare a joint multi-tree run against its single-tree control runs.

Usage:
    python experiments/multitree/compare_runs.py <joint_dir> <single_dir>[ ...]
        [--epoch-rule per-tree|overall]

Each directory is a training `--output-dir` holding a `*_val_metrics_by_epoch.json`.
Single-tree directories are matched to trees by the source name recorded inside their
metrics file, so the order of the arguments does not matter.

For every tree it prints the best validation NLL per held-out mutation reached by the
joint model and by that tree's own single-tree model, alongside the run's reference
points. The joint run earns its keep only where `joint - single` is negative.

The comparison is only valid when both runs used the same `--split-seed` and the same
source NAME, which makes `split_examples` draw the identical held-out set; this script
checks the held-out mutation counts agree and warns if they do not.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load_records(directory):
    matches = sorted(glob.glob(os.path.join(directory, "*_val_metrics_by_epoch.json")))
    if not matches:
        raise SystemExit(f"No *_val_metrics_by_epoch.json under {directory}")
    with open(matches[-1]) as handle:
        return json.load(handle)


def best_for(records, source, quantile, rule='per-tree'):
    """The epoch this run is scored at for one source, plus its hit rate and zero-shot row.

    rule='per-tree': that tree's own best epoch. The same selection is applied to every
        (run, tree) pair, so the selection bias is of the same kind on both sides — but
        with flat, noisy curves it still rewards whichever run fluctuates lowest.
    rule='overall': the epoch with the best *pooled* validation NLL, i.e. the one
        checkpoint you would actually ship. Note this is not symmetric: a single-tree run
        has one curve, so its pooled best IS its per-tree best, while a joint run must
        compromise across trees. It is the conservative rule for the joint model.
    """
    rows = [(r['epoch'], r['val']['per_source'][source]) for r in records
            if source in r['val']['per_source'] and r['epoch'] >= 0]
    if not rows:
        return None
    if rule == 'overall':
        pooled = min((r for r in records if r['epoch'] >= 0),
                     key=lambda r: r['val']['overall']['nll_per_mutation'])
        epoch, stats = pooled['epoch'], pooled['val']['per_source'][source]
    else:
        epoch, stats = min(rows, key=lambda row: row[1]['nll_per_mutation'])
    zero = next((r['val']['per_source'][source] for r in records if r['epoch'] < 0), None)
    return {
        'epoch': epoch,
        'nll': stats['nll_per_mutation'],
        'hit': stats['hit_rate'][quantile],
        'n_mutations': stats['n_mutations'],
        'per_example': stats.get('per_example'),
        'zero_shot_nll': zero['nll_per_mutation'] if zero else float('nan'),
        'n_epochs': max(r['epoch'] for r in records),
    }


def paired_interval(joint, single, n_boot=10000, seed=0):
    """Bootstrap 95% CI for (joint - single) NLL per mutation, paired by example.

    Both runs score the same validation examples in the same order (shuffle=False over a
    split drawn from the same per-source RNG), so resampling examples jointly removes the
    example-to-example variance that swamps an unpaired comparison.
    """
    import random as _random

    if not (joint and single):
        return None
    jl, jn = joint['loss'], joint['n_mutations']
    sl, sn = single['loss'], single['n_mutations']
    if len(jl) != len(sl) or jn != sn:
        return None            # different held-out sets; not comparable

    n = len(jl)
    point = (sum(jl) - sum(sl)) / max(sum(jn), 1e-9)
    rng = _random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        num = sum(jl[i] - sl[i] for i in idx)
        den = sum(jn[i] for i in idx)
        draws.append(num / max(den, 1e-9))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit(__doc__.strip())

    rule = 'per-tree'
    if '--epoch-rule' in argv:
        i = argv.index('--epoch-rule')
        rule = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    joint = load_records(argv[0])
    quantile = sorted(next(iter(joint[0]['val']['per_source'].values()))['hit_rate'])[0]
    sources = list(joint[0]['val']['per_source'])

    singles = {}
    for directory in argv[1:]:
        records = load_records(directory)
        for source in records[0]['val']['per_source']:
            singles[source] = records

    baselines = joint[0].get('baselines', {})
    header = (f"{'tree':<10} {'muts':>5} {'zero':>7} {'codon':>7} {'freq':>7} "
              f"{'single':>8} {'joint':>8} {'delta':>7}   hit@" + quantile + " (single -> joint)")
    print(header)
    print("-" * len(header))

    deltas = []
    for source in sources:
        joint_best = best_for(joint, source, quantile, rule)
        single_best = (best_for(singles[source], source, quantile, rule)
                       if source in singles else None)
        refs = baselines.get(source, {})
        codon = refs.get('codon', {}).get('nll_per_mutation', float('nan'))
        freq = refs.get('frequency', {}).get('nll_per_mutation', float('nan'))

        if single_best is None:
            print(f"{source:<10} {joint_best['n_mutations']:5.0f} "
                  f"{joint_best['zero_shot_nll']:7.3f} {codon:7.3f} {freq:7.3f} "
                  f"{'pending':>8} {joint_best['nll']:8.3f} {'—':>7}")
            continue
        if single_best['n_mutations'] != joint_best['n_mutations']:
            print(f"  !! {source}: held-out sets differ "
                  f"({single_best['n_mutations']:.0f} vs {joint_best['n_mutations']:.0f} "
                  f"mutations) — check --split-seed and the source name")

        delta = joint_best['nll'] - single_best['nll']
        deltas.append(delta)
        print(f"{source:<10} {joint_best['n_mutations']:5.0f} "
              f"{joint_best['zero_shot_nll']:7.3f} {codon:7.3f} {freq:7.3f} "
              f"{single_best['nll']:8.3f} {joint_best['nll']:8.3f} {delta:+7.3f}   "
              f"{single_best['hit']:.3f} -> {joint_best['hit']:.3f}  "
              f"(epochs {single_best['epoch']} / {joint_best['epoch']})")
        interval = paired_interval(joint_best.get('per_example'),
                                   single_best.get('per_example'))
        if interval:
            point, low, high = interval
            verdict = ("joint better" if high < 0 else
                       "single better" if low > 0 else "indistinguishable")
            print(f"{'':<10} {'':>5} paired delta {point:+.3f} nats/mutation, "
                  f"95% CI [{low:+.3f}, {high:+.3f}] -> {verdict}")

    print("\nnll = validation NLL per held-out mutation at each run's best epoch; "
          "lower is better.")
    print("delta = joint - single: negative means training on all three trees helped.")
    if deltas:
        better = sum(1 for d in deltas if d < 0)
        print(f"\nJoint model is better on {better}/{len(deltas)} trees "
              f"(mean delta {sum(deltas) / len(deltas):+.3f} nats).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
