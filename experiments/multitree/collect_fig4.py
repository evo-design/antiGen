#!/usr/bin/env python
"""perevent_fig4.npz -> Figure-4 panel CSV with bootstrap CIs.

For each tree x subset(all/novel) x arm, recall@10% = mean over test events of the
fractional top-10% hit weight, with a B=2000 event-level bootstrap CI (rng seed 0).
Model arms carry (variant, seed); a per-variant 'mean' row averages the 3 seeds'
point recalls (seed spread = min/max). Baselines carry a single arm name.

Writes data/multitree_attach/fig4_panels.csv:
  tree, kind(model|model_mean|baseline), variant, seed, subset, recall, lo, hi, n
"""
import os, sys
import numpy as np
_only = os.environ.get("FIG4_TREES", "").strip()
_suf = f"_{'_'.join(_only.split(','))}" if _only else ""
NPZ = f"/large_storage/hielab/aakarsh/multitree/fig4_single_attach/perevent_fig4{_suf}.npz"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_attach/fig4_panels.csv"
TREES = _only.split(",") if _only else ["denv2", "h3n2", "rsv_a"]
VARIANTS = ["xformer", "esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
SEEDS = [0, 1, 2]
BASELINES = ['zeroshot', 'site_prop', 'cpssm_dir', 'pssm', 'frequency',
             'where_times_which', 'substitution', 'codon_blosum', 'where_only']
B = 2000

def ci(w, rng):
    w = np.asarray(w, float)
    if len(w) == 0: return (np.nan, np.nan, np.nan, 0)
    idx = rng.integers(0, len(w), size=(B, len(w)))
    boot = w[idx].mean(axis=1)
    return (float(w.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), len(w))

def main():
    z = np.load(NPZ)
    rng = np.random.default_rng(0)
    rows = []
    for tree in TREES:
        for sub in ("all", "novel"):
            # models
            for v in VARIANTS:
                pts = []
                for s in SEEDS:
                    k = f"{tree}__{sub}__model__{v}__s{s}__q10"
                    if k not in z.files: continue
                    m, lo, hi, n = ci(z[k], rng)
                    rows.append(["model", tree, v, s, sub, m, lo, hi, n])
                    if not np.isnan(m): pts.append(m)
                if pts:
                    rows.append(["model_mean", tree, v, -1, sub,
                                 float(np.mean(pts)), float(np.min(pts)), float(np.max(pts)), len(pts)])
            # baselines
            for b in BASELINES:
                k = f"{tree}__{sub}__base__{b}__q10"
                if k not in z.files: continue
                m, lo, hi, n = ci(z[k], rng)
                rows.append(["baseline", tree, b, -1, sub, m, lo, hi, n])
            # uniform reference
            rows.append(["baseline", tree, "uniform", -1, sub, 0.10, 0.10, 0.10, 0])

    import csv
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["kind", "tree", "variant", "seed", "subset", "recall", "lo", "hi", "n"])
        wr.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} rows)")
    # quick console summary (all subset, model_mean + key baselines)
    print("\nrecall@10% (all) — model means and baselines:")
    for tree in TREES:
        print(f"  {tree}:")
        for r in rows:
            if r[1] == tree and r[4] == "all" and r[0] in ("model_mean", "baseline") \
               and r[2] in VARIANTS + ["cpssm_dir", "where_times_which", "zeroshot", "codon_blosum"]:
                print(f"    {r[0]:>11} {r[2]:>18}: {r[5]:.3f}  [{r[6]:.3f},{r[7]:.3f}]")

if __name__ == '__main__':
    main()
