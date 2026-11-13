"""Re-pool background_contrast's per-cell output from CELL-weighted to EVENT-weighted.

background_contrast.py reports "the unweighted mean over cells" (its own words): every
(site, AA) pair counts once, however many backbone queries host it. This re-weights each
cell by its host count, so a mutation hosted k times counts k times -- the convention
Figure 2's recall panels call event-weighted.

NO RE-RUN. background_percell.csv already carries the per-cell lift/auc/delta and the
host count, so this is arithmetic over a file that exists. It is ~hours of GPU to
recompute and seconds to re-pool.

THE GATE. Run with --weight none and the output must reproduce background_summary.csv
exactly, because an unweighted mean is what that file already holds. --check does that
comparison and refuses to write if it fails.

The bootstrap matches background_contrast.py's: percentile interval, same default
n_boot and seed. For the weighted estimate it resamples CELLS with probability
proportional to n_hosts and takes an unweighted mean of the draw, whose expectation is
the host-weighted mean -- so the interval is weighted the same way as the point it
brackets, rather than being a cell-weighted interval around an event-weighted dot.
"""
import argparse, csv, os, sys
import numpy as np

SUMMARY = "background_summary.csv"
PERCELL = "background_percell.csv"
CELLSETS = "background_cellsets.csv"
HDR = ["model", "cellset", "lift", "lift_lo", "lift_hi", "auc", "auc_lo", "auc_hi",
       "delta_pct", "n_cells", "n_pairs"]


def boot_ci(vals, w, n_boot, seed, alpha=0.05):
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=np.float64)
    if w is None:
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    else:
        # Inverse-CDF sampling, not rng.choice(p=...): choice with explicit
        # probabilities is orders of magnitude slower here (10k draws x ~1.2k cells
        # x 360 intervals), and searchsorted on a cumulative weight vector draws the
        # same distribution.
        cw = np.cumsum(np.asarray(w, dtype=np.float64))
        cw /= cw[-1]
        idx = np.searchsorted(cw, rng.random((n_boot, len(vals))), side="right")
        np.clip(idx, 0, len(vals) - 1, out=idx)
    means = vals[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--weight", choices=("hosts", "none"), default="hosts")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="compare against background_summary.csv and refuse to write "
                         "on disagreement (meaningful with --weight none)")
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()

    sets = {}
    with open(os.path.join(a.background_dir, CELLSETS)) as fh:
        for r in csv.DictReader(fh):
            sets.setdefault(r["cellset"], set()).add((int(r["pos1"]), r["alt"]))

    per = {}
    with open(os.path.join(a.background_dir, PERCELL)) as fh:
        for r in csv.DictReader(fh):
            per.setdefault(r["model"], {})[(int(r["pos1"]), r["alt"])] = (
                float(r["lift"]), float(r["auc"]), float(r["delta_pct"]),
                int(r["n_hosts"]), int(r["n_pairs"]))

    pub = {}
    sp = os.path.join(a.background_dir, SUMMARY)
    if os.path.exists(sp):
        with open(sp) as fh:
            for r in csv.DictReader(fh):
                pub[(r["model"], r["cellset"])] = (float(r["lift"]), float(r["auc"]))

    rows, worst, n_cmp = [], 0.0, 0
    for model in sorted(per):
        for cs in sorted(sets):
            got = [(c, per[model][c]) for c in sorted(sets[cs]) if c in per[model]]
            if not got:
                continue
            lift = [v[0] for _, v in got]
            auc = [v[1] for _, v in got]
            dlt = [v[2] for _, v in got]
            w = [v[3] for _, v in got] if a.weight == "hosts" else None
            if w is not None and sum(w) == 0:
                continue
            avg = (lambda x: float(np.average(x, weights=w)))
            llo, lhi = boot_ci(lift, w, a.n_boot, a.seed)
            alo, ahi = boot_ci(auc, w, a.n_boot, a.seed)
            rows.append([model, cs, f"{avg(lift):.6f}", f"{llo:.6f}", f"{lhi:.6f}",
                         f"{avg(auc):.6f}", f"{alo:.6f}", f"{ahi:.6f}",
                         f"{avg(dlt):.6f}", len(got), sum(v[4] for _, v in got)])
            if (model, cs) in pub:
                n_cmp += 1
                worst = max(worst, abs(avg(lift) - pub[(model, cs)][0]),
                            abs(avg(auc) - pub[(model, cs)][1]))

    if n_cmp:
        print(f"compared {n_cmp} (model, cellset) pairs against {SUMMARY}: "
              f"worst |delta| = {worst:.2e}")
    if a.check:
        if a.weight != "none":
            print("--check with --weight hosts: a difference is expected, not a failure")
        elif worst > a.tol:
            sys.exit(f"GATE FAILED: worst |delta| {worst:.2e} > {a.tol:.0e}; not writing")
        else:
            print("GATE PASSED: unweighted re-pool reproduces the published summary")

    os.makedirs(a.output_dir, exist_ok=True)
    out = os.path.join(a.output_dir, SUMMARY)
    with open(out, "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(HDR)
        w_.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows, weight={a.weight}")


if __name__ == "__main__":
    main()
