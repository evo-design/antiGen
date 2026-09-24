#!/usr/bin/env python
"""Re-score clade frequency predictions with the clade overlap taken seriously.

Above any frequency cutoff, the clades of a season are a nested chain: at cutoff 0.25 the
largest mutually non-nested subset is *one clade per season* in every season of the H3N2
data, and the frequencies of the selected clades sum to about six per season, so each
season's population is counted six times over. Ten clades on one 2013 lineage with X_t
running 1.00, 0.94, 0.90 ... 0.84 hold nearly the same strains and therefore nearly the
same W: get that lineage's direction right and ten "observations" are correct together.

Pooling direction accuracy over clades therefore reports an n that does not exist. This
module recomputes the same numbers with the season as the unit:

  per-season      each season contributes one growth rate and one decline rate, so a
                  chain of near-identical clades counts once rather than ten times;
  clustered CI    the bootstrap resamples *seasons*, not clades, which is the level at
                  which the observations are independent;
  antichain       a second reading that keeps literally one clade per season -- the
                  largest mutually non-nested subset -- chosen by a rule fixed in advance
                  (largest X_t) so there is no room to pick the flattering one.

Łuksza & Lässig use nested clades as prediction units deliberately, and Fig. 2a pools over
all 188 the same way. They also define a non-overlapping construction (eqs. 29-33: strip
from each clade every strain held by a proper subset clade, giving a partition of the
season) and use it for the information-gain ranking, so the problem is one they recognised
and the remedy is theirs; it is simply not what Fig. 2a does.

Usage:
    python -m antiGen.clade_overlap --predictions out/pred_c0.25_1se.json \
        --clade-table path/to/clades_seasons.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MODELS = ("null", "size", "fitness", "fitness (mean-f)", "fitness+size")


def ancestors(table: pd.DataFrame, season: int):
    """clade_id -> set of its ancestor clade_ids, within one season."""
    g = table[table.season == season].set_index("clade_id")
    out = {}
    for cid in g.index:
        seen, p = set(), g.loc[cid, "parent_clade_id"]
        while not pd.isna(p):
            p = int(p)
            seen.add(p)
            if p not in g.index:
                break
            p = g.loc[p, "parent_clade_id"]
        out[cid] = seen
    return out


def antichain(rows, anc, max_X=0.9):
    """One clade per season: the largest mutually non-nested subset.

    Clades above `max_X` are skipped first. The near-total clade of a season is a
    degenerate unit for this metric -- it *is* the population, so its observed W is pinned
    within a few per cent of 1 and its predicted W is 1 by construction, leaving a coin
    flip on the coverage gap. Taking simply "the largest" therefore selects the one clade
    per season that carries no information about which lineage outcompetes which. Ordering
    is by descending frequency among what remains, fixed here rather than per result."""
    pool = [r for r in rows if r["X_t"] <= max_X] or list(rows)
    keep = []
    for r in sorted(pool, key=lambda r: -r["X_t"]):
        cid = r["clade_id"]
        if all((cid not in anc.get(o["clade_id"], set())) and
               (o["clade_id"] not in anc.get(cid, set())) for o in keep):
            keep.append(r)
    return keep


TIE = 1e-6      # |log What| below this is no prediction at all, not a direction


def _rates(rows, tie=TIE):
    """(growth rate, decline rate) for one set of clades; nan where a class is absent.

    A prediction of no change scores half. The tie test is a tolerance on log What, not
    exact equality with 1: a clade holding nearly the whole season has
    <e^f>_clade ~ <e^f>_pop by construction, so its predicted ratio is 1 to every decimal
    that means anything and only the last bits of the float decide the comparison. Exact
    equality never fires on those and silently turns numerical noise into an 11-for-11
    winning streak."""
    g_hit = g_n = d_hit = d_n = 0.0
    for r in rows:
        lw = float(r["pred_logW"])
        is_tie = abs(lw) < tie
        if r["W"] > 1.0:
            g_n += 1
            g_hit += 0.5 if is_tie else float(lw > 0)
        elif r["W"] < 1.0:
            d_n += 1
            d_hit += 0.5 if is_tie else float(lw < 0)
    return (g_hit / g_n if g_n else np.nan, d_hit / d_n if d_n else np.nan, g_n, d_n)


def season_stats(rows):
    """One row per season: its growth rate, decline rate and balanced accuracy."""
    out = {}
    for s in sorted({r["season"] for r in rows}):
        sub = [r for r in rows if r["season"] == s]
        g, d, gn, dn = _rates(sub)
        bal = np.nanmean([g, d]) if (gn or dn) else np.nan
        out[s] = dict(growth=g, decline=d, balanced=bal, n=len(sub), n_grow=gn, n_decline=dn)
    return out


def cluster_bootstrap(rows, n_boot=4000, seed=0):
    """Resample seasons with replacement; report the season-mean balanced accuracy and
    the pooled r(W), each with a 95% interval. Seasons are the independent unit."""
    rng = np.random.default_rng(seed)
    by = {s: [r for r in rows if r["season"] == s] for s in {r["season"] for r in rows}}
    seasons = sorted(by)
    stats = season_stats(rows)
    point = float(np.nanmean([stats[s]["balanced"] for s in seasons]))
    W = np.asarray([r["W"] for r in rows])
    P = np.exp(np.asarray([r["pred_logW"] for r in rows]))
    point_r = float(np.corrcoef(P, W)[0, 1]) if len(rows) > 2 and P.std() > 0 else np.nan
    bal, cor = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(seasons), len(seasons), replace=True)
        draw = [r for i in pick for r in by[seasons[i]]]
        st = season_stats(draw)
        vals = [st[s]["balanced"] for s in st]
        if np.all(np.isnan(vals)):
            continue
        bal.append(float(np.nanmean(vals)))
        w = np.asarray([r["W"] for r in draw])
        p = np.exp(np.asarray([r["pred_logW"] for r in draw]))
        if len(draw) > 2 and p.std() > 0 and w.std() > 0:
            cor.append(float(np.corrcoef(p, w)[0, 1]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
    ties = sum(1 for r in rows if abs(float(r["pred_logW"])) < TIE)
    near = sum(1 for r in rows if r["X_t"] > 0.9)
    return dict(n_seasons=len(seasons), n_clades=len(rows), balanced=point,
                balanced_ci=q(bal), pearson=point_r, pearson_ci=q(cor),
                p_not_better=float(np.mean([b <= 0.5 for b in bal])) if bal else np.nan,
                n_ties=ties, n_near_total=near)


def report(path, table, models, n_boot, seed):
    payload = json.load(open(path))
    res = payload["results"][0]
    label = os.path.basename(path)[:-5]
    cutoff = res.get("cutoff")
    print(f"=== {label}  (cutoff {cutoff}, rule {res.get('lambda_rule')}) ===")
    anc = {}
    out = {}
    for name in models:
        rows = res.get("predictions", {}).get(name)
        if not rows:
            continue
        for s in {r["season"] for r in rows}:
            anc.setdefault(s, ancestors(table, s))
        pooled = cluster_bootstrap(rows, n_boot, seed)
        anti_rows = []
        for s in sorted({r["season"] for r in rows}):
            # clade_id is numbered within a season, not globally, so the nesting map has
            # to be the one for *this* season. Merging the per-season maps into one dict
            # silently keeps whichever season was iterated last and makes clades of every
            # other season look mutually non-nested, which turns the antichain back into
            # the full pooled set.
            anti_rows += antichain([r for r in rows if r["season"] == s], anc[s])
        anti = cluster_bootstrap(anti_rows, n_boot, seed)
        out[name] = dict(all_clades=pooled, antichain=anti)
        print(f"  {name:>17}  all clades: {pooled['n_clades']:>4} over "
              f"{pooled['n_seasons']} seasons  season-mean balanced "
              f"{pooled['balanced']:.3f} [{pooled['balanced_ci'][0]:.3f},"
              f"{pooled['balanced_ci'][1]:.3f}]  r(W) {pooled['pearson']:+.3f} "
              f"[{pooled['pearson_ci'][0]:+.3f},{pooled['pearson_ci'][1]:+.3f}]  "
              f"P(<=0.5) {pooled['p_not_better']:.3f}  "
              f"[{pooled['n_ties']} ties, {pooled['n_near_total']} clades over 0.9]")
        print(f"  {'':>17}  antichain:  {anti['n_clades']:>4} over "
              f"{anti['n_seasons']} seasons  season-mean balanced "
              f"{anti['balanced']:.3f} [{anti['balanced_ci'][0]:.3f},"
              f"{anti['balanced_ci'][1]:.3f}]  r(W) {anti['pearson']:+.3f} "
              f"[{anti['pearson_ci'][0]:+.3f},{anti['pearson_ci'][1]:+.3f}]  "
              f"P(<=0.5) {anti['p_not_better']:.3f}  "
              f"[{anti['n_ties']} ties, {anti['n_near_total']} over 0.9]")
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True,
                    help="clade_fitness JSONs carrying a 'predictions' block")
    ap.add_argument("--clade-table", required=True,
                    help="clades_<tag>.csv, for the parent_clade_id nesting")
    ap.add_argument("--models", nargs="+", default=["fitness", "size", "null"])
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    table = pd.read_csv(args.clade_table)
    paths = [p for pat in args.predictions for p in sorted(glob.glob(pat))]
    everything = {}
    for p in paths:
        everything[os.path.basename(p)[:-5]] = report(p, table, args.models,
                                                      args.n_boot, args.seed)
    print("Season-mean balanced accuracy weights every season equally, so a nested chain")
    print("counts once. The interval and P(<=0.5) come from resampling seasons.")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(everything, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
