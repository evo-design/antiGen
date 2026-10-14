#!/usr/bin/env python
"""Rebuild a run's validation set and join it to the per-example losses in its metrics.

The per-example NLLs written to ``<run_prefix>_val_metrics_by_epoch.json`` carry no
identifier — only their array position. But the split is deterministic, so the examples
themselves can be regenerated and lined up with those positions, which turns the existing
metrics files into a per-mutation record without re-running anything.

This matters because ``per_example.loss[i]`` is the *summed* NLL over all of example
``i``'s observed child mutations. Only for the single-mutation examples is it the NLL of
one known ``(position, alt_aa)`` cell; ``mutation_table`` keeps the count so callers can
filter on that.

Usage:
    from val_index import val_examples, mutation_table
    ex = val_examples('ncov', split_seed=0)
    tab = mutation_table(ex)      # (example_idx, position, alt_aa, count) rows

Run directly to check the replay against a run's logged counts:
    python experiments/multitree/val_index.py <run_dir> [--split-seed 0]
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def examples_path(source):
    return os.path.join(DATA_DIR, f"examples_{source}_nocutoff.txt")


def val_examples(source, split_seed=0, val_frac=0.1, path=None):
    """The validation ``Example`` list for one source, in the order the loader saw it.

    Delegates the split to ``antiGen.train.split_examples`` rather than reimplementing
    the RNG, so this cannot drift from what training actually did. The loaders are
    ``shuffle=False``, so loader order is this list's order.
    """
    from antiGen.load import load_examples
    from antiGen.train import split_examples

    path = path or examples_path(source)
    # Both are chatty on stdout; callers here only want the examples.
    with contextlib.redirect_stdout(io.StringIO()):
        examples = load_examples(path, ALPHABET, None, source=source)
        _, val_by = split_examples({source: examples}, 'random', val_frac, split_seed)
    return val_by[source]


def mutation_table(examples):
    """Rows of ``(example_idx, position, alt_aa_idx, count)`` over a list of examples.

    ``example_idx`` indexes into ``examples``, which is also the index into the metrics
    file's ``per_example`` arrays.
    """
    rows = []
    for idx, example in enumerate(examples):
        nz = (example.mat > 0).nonzero()
        for pos, alt in nz.tolist():
            rows.append((idx, pos, alt, float(example.mat[pos, alt])))
    return rows


def load_metrics(run_dir):
    matches = sorted(glob.glob(os.path.join(run_dir, "*_val_metrics_by_epoch.json")))
    if not matches:
        raise SystemExit(f"No *_val_metrics_by_epoch.json under {run_dir}")
    with open(matches[-1]) as handle:
        return json.load(handle)


def check(run_dir, split_seed, val_frac):
    """Verify the replay reproduces every source's logged example and mutation counts."""
    records = load_metrics(run_dir)
    scored = [r for r in records if r['epoch'] >= 0]
    if not scored:
        raise SystemExit(f"No scored epochs in {run_dir}")
    per_source = scored[0]['val']['per_source']

    failures = 0
    for source, logged in sorted(per_source.items()):
        if not os.path.exists(examples_path(source)):
            print(f"  {source:12s} SKIP (no examples file)")
            continue
        examples = val_examples(source, split_seed, val_frac)
        n_ex = len(examples)
        n_mut = sum(float(e.mat.sum()) for e in examples)
        want_ex = len(logged['per_example']['loss'])
        want_mut = logged['n_mutations']
        ok = (n_ex == want_ex) and abs(n_mut - want_mut) < 1e-6
        failures += not ok
        print(f"  {source:12s} examples {n_ex:5d}/{want_ex:<5d} "
              f"mutations {n_mut:7.0f}/{want_mut:<7.0f} {'ok' if ok else 'MISMATCH'}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--val-frac', type=float, default=0.1)
    args = parser.parse_args()

    print(f"Replaying {args.run_dir} (split_seed={args.split_seed}, "
          f"val_frac={args.val_frac})")
    failures = check(args.run_dir, args.split_seed, args.val_frac)
    if failures:
        raise SystemExit(f"{failures} source(s) did not match")
    print("All sources match.")


if __name__ == '__main__':
    main()
