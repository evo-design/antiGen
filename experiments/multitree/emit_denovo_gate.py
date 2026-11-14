#!/usr/bin/env python
"""Fit the per-example de-novo gate LEAVE-ONE-VIRUS-OUT and report where it lands between the
single-scalar router (0.900) and the per-event oracle ceiling (0.939).

For each held-out tree:
  - standardize features and fit logistic regression on the OTHER 12 trees' examples
    (target: example's realized event is de-novo), predict p_e for the held tree;
  - tune the alpha map (alpha_lo, alpha_hi) on the 12 training trees by maximizing their
    all-recall under per-example alpha_e = alpha_lo + (alpha_hi-alpha_lo)*p_e (snapped to grid);
  - apply the fitted gate + map to the held tree, read per-event weights at each example's alpha_e.
Reports macro all/seen/denovo, the gate's AUC, and the single-scalar baseline for reference."""
import os, sys
import numpy as np
NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_denovo_gate.npz"
CSV = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/denovo_gate.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
z = np.load(NPZ, allow_pickle=True)
ALPHAS = list(z["__alphas__"]); FEATS = list(z["__feats__"])
def akey(a): return f"a{a:+.3f}".replace('.', 'p').replace('+', 'P').replace('-', 'M')
AK = [akey(a) for a in ALPHAS]
ALO = [-1.0, -0.1, -0.05, -0.03, 0.0]
AHI = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]

def ev_w(tree, a): return z[f"{tree}__ev_w__{akey(a)}"]
def ev_N(tree):    return z[f"{tree}__ev_N"]
def ev_ex(tree):   return z[f"{tree}__ev_ex"]
def feat(tree):    return z[f"{tree}__ex_feat"]
def y(tree):       return z[f"{tree}__ex_y"]

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def fit_logistic(X, yy, l2=1.0, iters=500, lr=0.3):
    """plain-numpy L2 logistic regression (no sklearn dependency)."""
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        p = sigmoid(X @ w + b)
        g = p - yy
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b

def alpha_e(p, alo, ahi):
    a = alo + (ahi - alo) * p
    # snap each to nearest grid alpha
    idx = np.abs(np.array(ALPHAS)[None, :] - a[:, None]).argmin(axis=1)
    return idx  # index into ALPHAS/AK

def tree_recall_gated(tree, aidx, mask="all"):
    """per-event weight using per-example alpha index aidx (len = n_examples of tree)."""
    exj = ev_ex(tree); Nn = ev_N(tree)
    w = np.empty(len(exj), dtype=np.float32)
    for k in range(len(exj)):
        a = ALPHAS[aidx[exj[k]]]
        w[k] = ev_w(tree, a)[k]
    m = np.ones(len(w), bool) if mask == "all" else (Nn >= 1) if mask == "seen" else (Nn < 1)
    return w[m].mean() if m.sum() else np.nan

def fixed_recall(tree, a, mask="all"):
    w = ev_w(tree, a); Nn = ev_N(tree)
    m = np.ones(len(w), bool) if mask == "all" else (Nn >= 1) if mask == "seen" else (Nn < 1)
    return w[m].mean() if m.sum() else np.nan

def auc(score, lab):
    lab = np.asarray(lab);
    if lab.sum() == 0 or lab.sum() == len(lab): return np.nan
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score)+1)
    pos = r[lab == 1].sum(); n1 = lab.sum(); n0 = len(lab) - n1
    return (pos - n1*(n1+1)/2) / (n1*n0)

# ---- single-scalar reference (best fixed alpha over all trees) ----
glob_scores = [(a, np.mean([fixed_recall(t, a) for t in TREES])) for a in ALPHAS]
ga, gv = max(glob_scores, key=lambda x: x[1])
print(f"single global scalar best: alpha={ga:+.3f}  all={gv:.3f}  "
      f"seen={np.mean([fixed_recall(t,ga,'seen') for t in TREES]):.3f}  "
      f"denovo={np.mean([fixed_recall(t,ga,'denovo') for t in TREES]):.3f}\n")

# ---- LOVO gate ----
loo_all, loo_seen, loo_dn, aucs, chosen = [], [], [], [], []
for held in TREES:
    others = [t for t in TREES if t != held]
    Xtr = np.concatenate([feat(t) for t in others]); ytr = np.concatenate([y(t) for t in others])
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    Xn = (Xtr - mu) / sd
    w, b = fit_logistic(Xn, ytr)
    # tune alpha map on the training trees
    best, bpair = -1, (ga, ga)
    for alo in ALO:
        for ahi in AHI:
            if ahi < alo: continue
            tot = []
            for t in others:
                pt = sigmoid(((feat(t) - mu) / sd) @ w + b)
                ai = alpha_e(pt, alo, ahi)
                tot.append(tree_recall_gated(t, ai))
            m = np.mean(tot)
            if m > best: best, bpair = m, (alo, ahi)
    alo, ahi = bpair
    # apply to held
    ph = sigmoid(((feat(held) - mu) / sd) @ w + b)
    ai = alpha_e(ph, alo, ahi)
    loo_all.append(tree_recall_gated(held, ai))
    loo_seen.append(tree_recall_gated(held, ai, "seen"))
    loo_dn.append(tree_recall_gated(held, ai, "denovo"))
    aucs.append(auc(ph, y(held))); chosen.append(bpair)

print("per-virus gate AUC (held-out):")
for t, a in zip(TREES, aucs): print(f"  {t:>11}: {a:.3f}" if not np.isnan(a) else f"  {t:>11}:   n/a")
print(f"\nLEAVE-ONE-VIRUS-OUT gate (honest):")
print(f"  all={np.mean(loo_all):.3f}  seen={np.nanmean(loo_seen):.3f}  denovo={np.nanmean(loo_dn):.3f}")
print(f"  mean held-out AUC = {np.nanmean(aucs):.3f}")
import collections
cc = collections.Counter(f"({a:+.2f},{h:+.2f})" for a, h in chosen)
print(f"  (alpha_lo,alpha_hi) chosen across folds: {dict(cc)}")
print(f"\n  reference: single scalar {gv:.3f} | kappa=5N blend 0.903 | per-event oracle 0.939")

import csv
with open(CSV, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["metric", "value"])
    wr.writerow(["gate_all", np.mean(loo_all)]); wr.writerow(["gate_seen", np.nanmean(loo_seen)])
    wr.writerow(["gate_denovo", np.nanmean(loo_dn)]); wr.writerow(["gate_auc", np.nanmean(aucs)])
    wr.writerow(["single_scalar_all", gv])
print("wrote", CSV)
