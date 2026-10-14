#!/usr/bin/env python
"""Build a nested pre-D split so the EPOCH is chosen on a forecast, not on interpolation.

The forecast arms pick their epoch on a random 10% of the pre-D snapshot examples. That
validation set is an interpolation task -- a held-out node's parent is usually in the
training half -- while the reported task is extrapolation past D. The best interpolating
epoch need not be the best forecasting one.

This builds the nested alternative, entirely inside pre-D data:

    train  branches of the D' snapshot          (everything before D')
    val    substitutions crossing D', read off  (D' -> D, still pre-D)
           the D snapshot
    test   substitutions crossing D             (unchanged, opened only at scoring)

Train examples are written TRAINING and validation examples TESTING into a single file
per tree, so `antiGen-train --val-mode file` consumes it directly -- that mode exists for
exactly this, a temporal cutoff rather than a random split.

Only trees with a D' snapshot qualify. RSV A and B are 404 at 2024-07-01 (Charon only
retains them from 2025-01), so the nested panel is seven trees, not nine.

Usage:
    python experiments/multitree/make_nested_split.py \
        --inner-date 2024-07-01 --outer-date 2025-01-01
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_trees import CATALOGUE                                   # noqa: E402
from generate_all import AMBIGUOUS_REF_OK, EXTRA, SHORT, STRIP_STOP  # noqa: E402

SNAPS = os.path.normpath(os.path.join(HERE, "..", "..", "nextstrain", "snapshots"))
# Nine-tree forecast panel minus RSV, which has no 2024-07 snapshot.
PANEL = ["ncov_open_global_all-time", "flu_avian_h5nx_ha", "measles_genome",
         "dengue_denv1", "dengue_denv2", "dengue_denv3", "dengue_denv4"]


def genes_for(name):
    return dict(list(CATALOGUE.items()) + list(EXTRA.items()))[name]


def flags_for(name, tree_dir, stem):
    out = []
    sidecar = os.path.join(tree_dir, f"{stem}_root-sequence.json")
    if os.path.exists(sidecar):
        out += ["--ref", sidecar]
    if name in STRIP_STOP:
        out.append("--strip-trailing-stop")
    if name in AMBIGUOUS_REF_OK:
        out += ["--ambiguous-ref", "gap"]
    return out


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("    FAILED:", (r.stderr.strip().splitlines() or ["?"])[-1][:120])
    return r


def count_examples(path):
    return sum(1 for line in open(path) if line.startswith("EXAMPLE_ID:"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inner-date", default="2024-07-01")
    p.add_argument("--outer-date", default="2025-01-01")
    p.add_argument("--out", default=os.path.join(HERE, "data", "nested"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"{'tree':9s} {'train < Dprime':>15s} {'val Dprime->D':>15s}")
    for name in PANEL:
        short = SHORT[name]
        inner = os.path.join(SNAPS, f"{name}@{args.inner_date}.json")
        outer = os.path.join(SNAPS, f"{name}@{args.outer_date}.json")
        if not (os.path.exists(inner) and os.path.exists(outer)):
            print(f"{short:9s} missing snapshot, skipped")
            continue
        genes = genes_for(name)

        # train: every branch of the D' snapshot
        tr = os.path.join(args.out, f"examples_{short}@{args.inner_date}_train.txt")
        r = run([sys.executable, os.path.join(HERE, "make_tree_examples.py"),
                 "--tree", inner, "--genes", *genes, "--out", tr]
                + flags_for(name, SNAPS, f"{name}@{args.inner_date}"))
        if r.returncode != 0:
            continue

        # val: substitutions crossing D', queried on the D snapshot. Pre-D throughout,
        # so nothing here touches the reported post-D set.
        va = os.path.join(args.out, f"examples_{short}@{args.inner_date}_val.txt")
        r = run([sys.executable, os.path.join(HERE, "make_forecast_testset.py"),
                 "--snapshot", inner, "--current", outer, "--genes", *genes,
                 "--date", args.inner_date, "--out", va]
                + [a for a in flags_for(name, SNAPS, f"{name}@{args.inner_date}")
                   if a not in ("--ref",)
                   and not a.endswith("_root-sequence.json")])
        if r.returncode != 0 or not os.path.exists(va):
            continue

        # One file: TRAINING rows then TESTING rows, for --val-mode file. Example ids
        # may collide across the two halves; nothing resolves parent_id, so that is
        # inert (load.py stores it and no consumer dereferences it).
        combined = os.path.join(args.out, f"examples_{short}@nested.txt")
        with open(combined, "w") as out:
            for src in (tr, va):
                with open(src) as fh:
                    out.write(fh.read())
        print(f"{short:9s} {count_examples(tr):15d} {count_examples(va):15d}")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
