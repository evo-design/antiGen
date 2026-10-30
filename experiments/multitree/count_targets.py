#!/usr/bin/env python
"""How many target mutations each endemic panel actually rests on.

A recall is a fraction, and a fraction says nothing without its denominator. The
backbone test sets are small enough that this stops being pedantry: h3n2's de novo
panel has **five** target mutations, so one of them is twenty points of recall and the
ordering of the bars is noise. The figure cannot be read honestly without the count,
so the count is written here and annotated onto the panel.

The de novo cut is the eval's: a cell is de novo when the training-window root counts
have never seen it, which is why --root-counts is read here rather than assumed.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from antiGen.load import load_examples, load_counts_matrix  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
VIRUSES = (("h3n2", "2022-10-01"), ("rsv-a", "2023-10-01"), ("denv2", "2019-10-01"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/home/ivan.specht/antiGen-data-revised/output")
    ap.add_argument("--testset", default="test",
                    help="directory suffix: 'test' or 'test-backbone'")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    for v, d in VIRUSES:
        root = load_counts_matrix(f"{args.data}/{v}-{d}/train/counts_{d}_cur.tsv", AA)
        seen = root > 0
        ex = load_examples(f"{args.data}/{v}-{args.testset}/examples_all_cur.txt",
                           AA, None, split_date=d)
        test = [e for e in ex if not e.is_training]
        tot = sum(float(e.mat.sum()) for e in test)
        dn = sum(float(e.mat[~seen].sum()) for e in test)
        rows.append({"virus": v, "metric": "all", "n_queries": len(test),
                     "n_targets": int(round(tot))})
        rows.append({"virus": v, "metric": "unseen", "n_queries": len(test),
                     "n_targets": int(round(dn))})
        print(f"{v:6s} queries={len(test):5d}  all={tot:7.0f}  de novo={dn:6.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["virus", "metric", "n_queries", "n_targets"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
