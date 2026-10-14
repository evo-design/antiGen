#!/usr/bin/env python
"""Keep only the newest run's artifacts in each output directory.

Re-running an arm into an existing --output-dir leaves the previous attempt's files
beside the new ones; they differ only by the timestamp in their filename. Readers here
take sorted(glob(...))[-1], which picks the newest, but that is a fragile invariant to
rely on when two attempts used *different splits* -- the matched/ arms were re-run from
a two-way to a three-way split, so a stale checkpoint there is not merely old, it was
trained on data that is now part of the test block.

Keeps the lexicographically greatest timestamp prefix per directory (timestamps sort
chronologically) and only if that run looks complete: metrics JSON plus a pooled
best checkpoint. Dry-run by default.

Usage:
    python experiments/multitree/prune_stale_runs.py <dir> [<dir> ...] [--apply]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

STAMP = re.compile(r'_(\d{8}_\d{6})')


def prefixes(files):
    out = {}
    for f in files:
        m = STAMP.search(f)
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', nargs='+')
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()

    freed = 0
    for d in args.dirs:
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        by = prefixes(files)
        if len(by) <= 1:
            continue
        newest = max(by)
        keep = by[newest]
        if not any(f.endswith('_val_metrics_by_epoch.json') for f in keep) or \
           not any(f.endswith('_best.pt') for f in keep):
            print(f"  SKIP {d}: newest run {newest} looks incomplete")
            continue
        for stamp, group in sorted(by.items()):
            if stamp == newest:
                continue
            for f in group:
                path = os.path.join(d, f)
                freed += os.path.getsize(path)
                if args.apply:
                    os.remove(path)
        print(f"  {'removed' if args.apply else 'would remove'} "
              f"{sum(len(g) for s, g in by.items() if s != newest):3d} stale file(s) "
              f"from {d} (keeping {newest})")
    print(f"{'Freed' if args.apply else 'Would free'} {freed / 1e9:.1f} GB")
    if not args.apply:
        print("Dry run; pass --apply to delete.")


if __name__ == '__main__':
    main()
