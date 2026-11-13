#!/usr/bin/env python
"""Bloom & Neher's amino-acid fitness estimates as an antiGen baseline score map.

Converts `aa_fitness.csv` from jbloomlab/SARS2-mut-fitness (Virus Evolution 2023,
vead055) into the (1273 x 22) spike CSV that `load_baselines` reads, one row per
residue in Wuhan-Hu-1 numbering and one column per token of the 22-token frame.

**THE VALUES ARE SHIFTED, AND THEY HAVE TO BE.** `load_baselines` ends with
`tensor[tensor < 0] = nan`: every negative entry is a sentinel meaning "this baseline
abstains here". Bloom & Neher's fitness is a log ratio of actual to expected counts, so
negative means DELETERIOUS -- 72% of spike cells. Written raw, the baseline would
silently abstain on nearly three quarters of its own estimates, and the quarter left
would be the tolerated mutations, which is a bias toward exactly the cells most likely
to be observed. A constant is therefore added to put every real estimate above zero.
Adding a constant is strictly monotone, and the hit-rate metrics read within-query
RANKS only, so this changes no number the figure reports.

**ONLY ESTIMATES THEY WOULD SHOW.** `min_expected_count` is a display cutoff in their
config, not a filter on the released file, which carries cells with expected counts
below 1. Those are the ones their own README calls least accurate. Cells under the
threshold are written as the sentinel, so the baseline abstains where the authors would
not quote a number.

**WRITTEN AT FULL PRECISION.** Rounding the shifted values introduced ties that are
not in the source -- two estimates differing in the seventh decimal became one number,
and a tie is a real thing in these metrics (the fractional top-q convention spreads it
across the quantile boundary). The source's own precision is what gets written.

**USE A DATE-MATCHED SNAPSHOT.** Their default release is built from the newest tree.
For a train/test comparison, pass the `results_public_<date>/` copy whose date equals
the cutoff, or the baseline has counted mutations from inside the test window.
"""
from __future__ import annotations

import argparse
import csv
import os

TOKENS = list("ACDEFGHIKLMNPQRSTVWY") + ["*", "-"]
SENTINEL = -1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aa-fitness", required=True, help="their aa_fitness.csv")
    ap.add_argument("--gene", default="S")
    ap.add_argument("--seq-len", type=int, default=1273)
    ap.add_argument("--min-expected-count", type=float, default=10.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.aa_fitness)) if r["gene"] == args.gene]
    if not rows:
        raise SystemExit(f"no rows with gene == {args.gene!r} in {args.aa_fitness}")

    kept, dropped_thresh, out_of_range = {}, 0, 0
    lo = None
    for r in rows:
        if r["aa"] not in TOKENS:
            continue
        site, fit = int(r["aa_site"]), float(r["fitness"])
        if not 1 <= site <= args.seq_len:
            out_of_range += 1
            continue
        if float(r["expected_count"]) < args.min_expected_count:
            dropped_thresh += 1
            continue
        kept[(site, r["aa"])] = fit
        lo = fit if lo is None else min(lo, fit)

    if not kept:
        raise SystemExit("nothing survived the expected-count threshold")
    # Smallest shift putting every estimate strictly above the sentinel band, rounded
    # up to a whole number so the file is readable and the choice is obvious.
    shift = float(int(abs(min(lo, 0.0))) + 1)
    vals = [v + shift for v in kept.values()]
    if min(vals) <= 0:
        raise SystemExit(f"shift {shift} left non-positive values (min {min(vals)})")

    grid = [[SENTINEL] * len(TOKENS) for _ in range(args.seq_len)]
    for (site, aa), fit in kept.items():
        grid[site - 1][TOKENS.index(aa)] = repr(fit + shift)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TOKENS)
        w.writerows(grid)

    # COUNTED OVER THE 20 RESIDUES, because that is the alphabet the eval loads with:
    # `load_baselines` drops the '*' and '-' columns, so counting them here reports a
    # coverage the figure will never have.
    aa20 = set("ACDEFGHIKLMNPQRSTVWY")
    scored = sum(1 for (_s, a) in kept if a in aa20)
    rbd = sum(1 for (s, a) in kept if a in aa20 and 331 <= s <= 531)
    print(f"wrote {args.output}")
    print(f"  {args.seq_len} rows x {len(TOKENS)} columns")
    print(f"  scored cells (20-residue alphabet): {scored:,}  (RBD 331-531: {rbd:,} "
          f"of 3,819 = {100*rbd/3819:.1f}% of the panels' pool)")
    print(f"  dropped below expected_count {args.min_expected_count:g}: {dropped_thresh:,}")
    if out_of_range:
        print(f"  outside 1..{args.seq_len}: {out_of_range:,}")
    print(f"  shift applied: +{shift:g}  (raw range {lo:.3f} .. "
          f"{max(kept.values()):.3f} -> {min(vals):.3f} .. {max(vals):.3f})")


if __name__ == "__main__":
    main()
