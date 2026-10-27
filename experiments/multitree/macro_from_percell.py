#!/usr/bin/env python
"""Cell-weighted (macro) recall from `list_macro_percell.csv`, at any event threshold.

THE POINT OF READING THE PER-CELL FILE rather than the summary: `list_macro.py`'s
`summarize` averages the per-cell rate over cells with ANY observed event
(`cell_events > 0`), and that threshold is baked into the number it writes. The
per-cell file keeps `k_events` for every cell, so every threshold is a different
aggregation of one run. Asking for "cells seen at least 100 times" therefore costs no
GPU at all -- it is the same pass, summed differently.

    macro(t) = mean over cells with k_events >= t of (hits / k_events)

`hits` is already the number of queries carrying that cell as a descendant mutation in
which it landed in the top q, so `rate` is that share and the mean over cells is the
cell-weighted recall. `k_events` is how many queries carry it at all.

TWO THINGS THIS CHECKS RATHER THAN ASSUMES:

* **The micro column is the gate.** Summed the other way -- total hits over total
  events -- the same file must reproduce the EVENT-weighted recall that
  `quantile_hit_rates_*_all-models.csv` carries, which was written by `eval.py` through
  a different accumulator. `--gate-against` does that comparison and refuses on
  disagreement. Without it the macro number has nothing holding it to the figure it
  will sit on.
* **A threshold that empties a metric is an error, not an empty panel.** A bar that
  silently rests on three cells is worse than a render that stops.
"""
import argparse
import csv
import os
from collections import defaultdict

# The per-cell variant that matches each of the figure's recall metrics. The pool is
# the whole eval range in both ("fullpool"); what differs is which cells count as
# targets. With every cell listed, `list_allcells` and `list_allcells_fullpool` are
# the same object, and the fullpool spelling is the one that says why.
VARIANT_FOR_METRIC = {
    "quantile_hit_rates_all":    "list_{label}_fullpool",
    "quantile_hit_rates_unseen": "list_{label}_unseen_fullpool",
}


def read_percell(path, label, qtile):
    """(metric, model) -> list of (k_events, hits)."""
    want = {v.format(label=label): k for k, v in VARIANT_FOR_METRIC.items()}
    out = defaultdict(list)
    seen_variants = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            seen_variants.add(row["variant"])
            metric = want.get(row["variant"])
            if metric is None or abs(float(row["qtile"]) - qtile) > 1e-9:
                continue
            out[(metric, row["model"])].append(
                (float(row["k_events"]), float(row["hits"])))
    if not out:
        raise SystemExit(
            f"{path} carries no rows for {sorted(want)} at qtile {qtile}. "
            f"Variants present: {sorted(seen_variants)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--percell", required=True)
    ap.add_argument("--label", default="allcells")
    ap.add_argument("--qtile", type=float, default=0.1)
    ap.add_argument("--min-events", type=float, default=1.0)
    ap.add_argument("--gate-against", default=None, metavar="EVAL_DIR",
                    help="directory holding quantile_hit_rates_*_all-models.csv")
    ap.add_argument("--gate-tol", type=float, default=1e-6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cells = read_percell(args.percell, args.label, args.qtile)

    if args.gate_against:
        col = f"top_{100 * args.qtile:.1f}%"
        worst, n = 0.0, 0
        for metric in sorted(VARIANT_FOR_METRIC):
            path = os.path.join(args.gate_against, f"{metric}_all-models.csv")
            if not os.path.exists(path):
                raise SystemExit(f"gate: no {path}")
            published = {}
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if col not in row:
                        raise SystemExit(f"gate: {path} has no column {col!r}")
                    published[row["model"]] = float(row[col])
            for (m, model), vals in cells.items():
                if m != metric or model not in published:
                    continue
                ev = sum(k for k, _ in vals)
                micro = sum(h for _, h in vals) / ev if ev else float("nan")
                worst = max(worst, abs(micro - published[model]))
                n += 1
        if n == 0:
            raise SystemExit("gate: no model matched between the two files")
        print(f"[gate] micro vs eval.py's own row, worst |delta| = {worst:.3e} "
              f"over {n} (metric, model) pair(s)")
        if worst > args.gate_tol:
            raise SystemExit(
                f"gate FAILED: {worst:.3e} > {args.gate_tol:.1e}. The per-cell file "
                f"and the published recall table disagree about the same quantity, so "
                f"the macro column beside them is not to be read.")

    rows = []
    for (metric, model), vals in sorted(cells.items()):
        kept = [(k, h) for k, h in vals if k >= args.min_events]
        if not kept:
            raise SystemExit(
                f"--min-events {args.min_events:g} leaves no cell for {model} in "
                f"{metric}. An empty panel is not a result; lower the threshold.")
        rows.append((model, metric, sum(h / k for k, h in kept) / len(kept), len(kept)))

    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "metric", "value", "n_cells"])
        w.writerows((m, mt, f"{v:.8f}", n) for m, mt, v, n in rows)

    # The cell count is per (metric, model) and SHOULD be one number per metric: the
    # threshold selects cells by how often they were observed, which has nothing to do
    # with which model is scoring them. Printing the set rather than the first value
    # makes a disagreement visible instead of averaging over it.
    per_metric = {}
    for _m, mt, _v, n in rows:
        per_metric.setdefault(mt, set()).add(n)
    for mt in sorted(per_metric):
        ns = sorted(per_metric[mt])
        shown = ns[0] if len(ns) == 1 else f"{ns} (DIFFERS BY MODEL)"
        print(f"  {mt:34s} {shown} cell(s) at >= {args.min_events:g} events")
    print(f"wrote {len(rows)} row(s) -> {args.output}")


if __name__ == "__main__":
    main()
