#!/usr/bin/env python
"""Report the rank-fusion router: alpha frontier (macro all/seen/de-novo), the in-sample
best alpha, and the LEAVE-ONE-VIRUS-OUT learned-alpha (pick alpha on 12 trees, apply to the
held-out one) -- the honest number for 'learning that scalar'."""
import os, sys
import numpy as np
NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_router_interp.npz"
CSV = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/router_interp.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
ALPHAS = [-1.0, -0.4, -0.25, -0.18, -0.12, -0.08, -0.05, -0.03, -0.015, 0.0,
          0.015, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 1.0]
def akey(a): return f"a{a:+.3f}".replace('.', 'p').replace('+', 'P').replace('-', 'M')
z = np.load(NPZ)

def tree_recall(key, tree, mask="all"):
    w = z[f"{tree}__all__{key}__q10"]; N = z[f"{tree}__all__N"]
    m = np.ones(len(w), bool) if mask == "all" else (N >= 1) if mask == "seen" else (N == 0)
    return w[m].mean() if m.sum() else np.nan

def macro(key, mask="all"):
    v = [tree_recall(key, t, mask) for t in TREES]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v))

ref_ag = macro("joint"); ref_sc = macro("cpssm_dir")
ref_ag_dn = macro("joint", "denovo"); ref_sc_dn = macro("cpssm_dir", "denovo")
ref_ag_s = macro("joint", "seen"); ref_sc_s = macro("cpssm_dir", "seen")
print(f"reference  antiGen(49):   all={ref_ag:.3f}  seen={ref_ag_s:.3f}  denovo={ref_ag_dn:.3f}")
print(f"reference  SmoothedCounts: all={ref_sc:.3f}  seen={ref_sc_s:.3f}  denovo={ref_sc_dn:.3f}\n")

print(f"{'alpha':>7} {'all':>7} {'seen':>7} {'denovo':>7}")
best_a, best_all = None, -1
import csv
with open(CSV, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["alpha", "all", "seen", "denovo"])
    for a in ALPHAS:
        k = akey(a); al = macro(k); sn = macro(k, "seen"); dn = macro(k, "denovo")
        wr.writerow([a, al, sn, dn])
        star = ""
        if al > best_all: best_all, best_a = al, a;
        print(f"{a:>7.3f} {al:7.3f} {sn:7.3f} {dn:7.3f}")
print(f"\nin-sample best: alpha={best_a:+.3f}  all={best_all:.3f}  "
      f"seen={macro(akey(best_a),'seen'):.3f}  denovo={macro(akey(best_a),'denovo'):.3f}")

# LEAVE-ONE-VIRUS-OUT learned alpha: pick alpha maximizing all-recall on the other 12 trees.
loo_all, loo_seen, loo_dn, chosen = [], [], [], []
for held in TREES:
    others = [t for t in TREES if t != held]
    best, ba = -1, None
    for a in ALPHAS:
        m = np.mean([tree_recall(akey(a), t, "all") for t in others])
        if m > best: best, ba = m, a
    chosen.append(ba)
    loo_all.append(tree_recall(akey(ba), held, "all"))
    loo_seen.append(tree_recall(akey(ba), held, "seen"))
    loo_dn.append(tree_recall(akey(ba), held, "denovo"))
import collections
cc = collections.Counter(f"{a:+.3f}" for a in chosen)
print(f"\nLEAVE-ONE-VIRUS-OUT learned alpha (honest):")
print(f"  all={np.mean(loo_all):.3f}  seen={np.nanmean(loo_seen):.3f}  denovo={np.nanmean(loo_dn):.3f}")
print(f"  alpha chosen across folds: {dict(cc)}")
print(f"\nwrote {CSV}")
