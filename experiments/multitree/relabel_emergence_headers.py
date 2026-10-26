#!/usr/bin/env python
"""Rewrite the emergence CSVs' t_* column headers from the grid's own dates.

WHY THIS EXISTS RATHER THAN A RE-RUN. `eval.py` labelled the emergence grid's columns
`(i + 1) * 0.1`, the step INDEX times a tenth of a year. That was right for exactly one
grid -- the published arm's `pssms/<variant>/time_<t>.csv`, whose filenames ARE t and
which the pipeline writes at 0.1-year spacing. `--variant-counts-dir` is a different
grid: dated files at whatever spacing the corpus was built with. The revised arm's is
QUARTERLY over 2021-01-01 .. 2026-01-01, so a five-year span was headed t_0.1 .. t_2.1,
an axis wrong by a factor of 2.5 -- and panel L's `x >= 2` filter, which means "from two
years in", silently became "the last two points" (NOTES s72).

`emergence_step_labels(n_steps, dates=)` fixed the writer. The CSVs already on disk
carry correct VALUES under wrong headers, so this rewrites the header line and nothing
else: an 11-hour eval does not need repeating to correct a label.

THE LABELLER IS IMPORTED, NOT RESTATED. A second copy of the convention is a mismatch
waiting to happen, and the figure asserts that the `_all-models` and `_ensemble` CSVs
have identical headers -- so the scorer that writes the second must be using the same
function as the script that rewrites the first.

IDEMPOTENT, AND IT CHECKS RATHER THAN ASSUMES. A file already carrying the right
headers is left alone and reported. A file whose column count does not match the grid
is REFUSED, not truncated: that is the signature of a CSV from a different run, and
silently rewriting its headers would put confident wrong dates on real numbers.
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from antiGen.eval import emergence_step_labels  # noqa: E402


def counts_dates(counts_dir, since=None):
    """The grid's dates, in the order `load_counts_series` reads them: sorted by name."""
    dates = []
    for path in sorted(glob.glob(os.path.join(counts_dir, "counts_*.tsv"))):
        m = re.search(r"counts_(\d{4}-\d{2}-\d{2})\.tsv$", os.path.basename(path))
        if not m:
            raise SystemExit(f"cannot read a date out of {path}")
        dates.append(m.group(1))
    if since:
        dates = [d for d in dates if d >= since]
    return dates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True,
                    help="directory holding variant_emergence_*.csv")
    ap.add_argument("--variant-counts-dir", required=True)
    ap.add_argument("--counts-since", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dates = counts_dates(args.variant_counts_dir, args.counts_since)
    want = emergence_step_labels(len(dates), dates=dates)
    print(f"{len(dates)} grid points, {dates[0]} .. {dates[-1]}")
    print(f"headers {want[0]} .. {want[-1]}")

    paths = sorted(glob.glob(os.path.join(args.eval_dir, "variant_emergence_*.csv")))
    if not paths:
        raise SystemExit(f"no variant_emergence_*.csv in {args.eval_dir}")

    n_done = n_same = 0
    for path in paths:
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        # Everything before the first t_ column is an identifier column; the published
        # shape is (model, top_quantile) but it is located rather than assumed.
        lead = [i for i, c in enumerate(header) if not c.startswith("t_")]
        if lead != list(range(len(lead))):
            raise SystemExit(f"{path}: t_ columns are not contiguous at the end")
        n_t = len(header) - len(lead)
        if n_t != len(want):
            raise SystemExit(
                f"{path}: has {n_t} t_ columns but the grid has {len(want)}. "
                f"This CSV is from a different run; refusing to relabel it.")
        if header[len(lead):] == want:
            print(f"  [same] {os.path.basename(path)}")
            n_same += 1
            continue
        rows[0] = header[:len(lead)] + want
        if not args.dry_run:
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerows(rows)
        print(f"  [{'would' if args.dry_run else 'done'}] {os.path.basename(path)}: "
              f"{header[len(lead)]} .. {header[-1]}  ->  {want[0]} .. {want[-1]}")
        n_done += 1
    print(f"{n_done} rewritten, {n_same} already correct")


if __name__ == "__main__":
    main()
