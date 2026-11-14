#!/usr/bin/env python
"""Emit the combined --examples spec for training the fc13 joint model on the 13 eval
panels' own pre-cutoff _train.txt PLUS the extra core-virus panels (cut at the same
D=2025-01-01). Prints space-separated NAME=PATH tokens for run_fc.py / antiGen-train.

  --mode single|surface   which core corpus to add (trees_core_<mode>.json)
  --panels-only           print only the added core panels (omit the 13 eval trains)

The 13 eval panels are temporally clean already; the core panels were generated with the
same cutoff (snapshot where retained, else pruned) by build_core_corpus.py, so the whole
training pool is leak-free against the post-2025-01-01 test set.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NA = os.path.join(HERE, "data", "forecast_noattach")
DATE = "2025-01-01"

# The 13 fc13 eval panels (their pre-cutoff training files already exist in NA/).
FC13 = ["denv1", "denv2", "denv3", "denv4", "h1n1pdm_ha", "h3n2_ha", "h5nx",
        "hmpv_a_g", "measles", "ncov", "rsv_a", "rsv_b", "vic_ha"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "surface"], required=True)
    ap.add_argument("--panels-only", action="store_true")
    ap.add_argument("--route", choices=["all", "snapshot", "prune"], default="all",
                    help="restrict the added core panels by cutoff route "
                         "(snapshot = strictly leak-free @D builds only)")
    args = ap.parse_args()

    specs = []
    if not args.panels_only:
        for t in FC13:
            f = os.path.join(NA, f"examples_{t}@{DATE}_train.txt")
            if not os.path.isfile(f):
                raise SystemExit(f"missing fc13 train file: {f}")
            specs.append(f"{t}={f}")

    manifest = json.load(open(os.path.join(HERE, "data", f"trees_core_{args.mode}.json")))
    for panel in sorted(manifest):
        if args.route != "all" and manifest[panel].get("route") != args.route:
            continue
        specs.append(f"{panel}={manifest[panel]['path']}")

    print(" ".join(specs))


if __name__ == "__main__":
    main()
