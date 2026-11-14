#!/usr/bin/env python
"""J/K forests (49-tree minus single-tree; 49-tree minus S×S–MCM Interp.) under BOTH the
published full pool and the 1nt pool, per virus + POOLED, exactly as build_fig4_j9.forest_J /
forest_K (paired per-event bootstrap; pooled = macro_est_ci_delta). The full-pool rows are a
validation against forest_fig4_j9.csv; the 1nt rows go to forest_fig4_j9_1nt.csv. Reuses
score_j9_1nt's tdata / per-event kernel (joint49 sel9 10-seed, single 10-seed, cpssm_dir)."""
import os, sys, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from score_j9_1nt import build_tdata, score_rung_tree, free, TREES, SEEDS
from score_joint49_sel9 import load_joint49_sel9
from score_HI_j9 import load_ensemble
from build_fig4_j9 import paired_ci, macro_est_ci_delta

J9 = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_j9"

def main():
    td = build_tdata(); W = {}
    j49 = [load_joint49_sel9(s)[0] for s in SEEDS]
    for t in TREES: W[("joint", t)] = score_rung_tree("joint", 'ensemble', t, td, models=j49); print(f"  joint {t}", flush=True)
    free(j49)
    for t in TREES:
        sm, _ = load_ensemble('single', t)
        W[("single", t)] = score_rung_tree("single", 'ensemble', t, td, models=sm); free(sm); print(f"  single {t}", flush=True)
        W[("cpssm_dir", t)] = score_rung_tree("cpssm_dir", 'cpssm_dir', t, td)
    out = {"baseline": [], "onent": []}
    for conv in out:
        for contrast, comp in (("single", "single"), ("smoothed", "cpssm_dir")):
            summ = {}
            for t in TREES:
                for sub in ("all", "novel"):
                    k = (sub, conv)
                    if k not in W[("joint", t)]: continue
                    jv = np.asarray(W[("joint", t)][k]); cv = np.asarray(W[(comp, t)][k])
                    assert len(jv) == len(cv), (t, sub, conv, len(jv), len(cv))
                    d = jv - cv; lo, hi = paired_ci(d)
                    out[conv].append((contrast, t, sub, float(d.mean()), lo, hi, 0))
                    summ.setdefault(sub, []).append((float(d.mean()), lo, hi))
            for sub, rv in summ.items():
                out[conv].append((contrast, "POOLED", sub, *macro_est_ci_delta(rv), 1))
    with open(f"{J9}/forest_fig4_j9_1nt.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["contrast", "tree", "subset", "est", "lo", "hi", "summary"])
        for r in out["onent"]: w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}", r[6]])
    pub = {(r['contrast'], r['tree'], r['subset']): float(r['est']) for r in csv.DictReader(open(f"{J9}/forest_fig4_j9.csv"))}
    d = [abs(r[3] - pub[(r[0], r[1], r[2])]) for r in out["baseline"] if (r[0], r[1], r[2]) in pub]
    print(f"\nVALIDATION full-pool forest vs forest_fig4_j9.csv: {len(d)} rows, max|Δ|={max(d):.6f}")
    for r in out["onent"]:
        if r[1] == "POOLED": print(f"  1nt {r[0]:9}{r[2]:6} {r[3]:+.4f} [{r[4]:+.4f},{r[5]:+.4f}]")

if __name__ == "__main__":
    main()
