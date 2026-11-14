#!/usr/bin/env python
"""Fig-5 method table restricted to NON-de-novo test mutations (the exact (pos,alt)
cell occurred >=1 time in that virus's training tree). Computed from the `all`
per-event arrays masked by the training count N[pos,alt] (same order/machinery as
stratify_antigen_vs_*.py), so all / de-novo / non-de-novo are one clean partition.

Macro recall@10% = per-tree event recall averaged over the 13 trees; CI = tree-level
bootstrap (matching the Fig-5 ladder's pooling). Uniform is 0.10 by construction.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_fig13_sp4e0.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/table_nondenovo.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
# Fig-5 ladder rungs (weak -> strong), with display names.
ARMS = [
    ("pssm",              "PSSM (sequence composition)"),
    ("codon_blosum",      "Codon aware BLOSUM62"),
    ("zeroshot",          "ESM-2 zero-shot"),
    ("frequency",         "Event Counts"),
    ("where_times_which", "Where x which"),
    ("site_prop",         "Site x Substitution"),
    ("cpssm_dir",         "Smoothed Counts"),
    ("single",            "antiGen (single-tree)"),
    ("joint13",           "antiGen (13 trees)"),
    ("joint",             "antiGen (49 trees)"),
]
z = np.load(NPZ)

# N[pos,alt] per test event, in the SAME order as the `all` per-event arrays.
Ncounts = {}
for tree in TREES:
    tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
    L = len(tr[0].aa_idx)
    N = count_baselines.fit(tr, L, E)["N"]
    nc = []
    for ex in load_examples(f"{NA}/examples_{tree}@{D}_test.txt", ALPHA_SET, None, source=tree):
        ev = events_of(ex)
        if not ev:
            continue
        for (pos, a) in ev:
            if a in MASK:
                continue
            nc.append(float(N[pos, a]))
    Ncounts[tree] = np.array(nc)

rng = np.random.default_rng(0); B = 2000
def macro(per):
    per = np.asarray([p for p in per if p == p])
    if len(per) == 0: return (float('nan'),) * 3
    idx = rng.integers(0, len(per), size=(B, len(per))); bs = per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# total event counts per partition (pooled)
allN = np.concatenate([Ncounts[t] for t in TREES])
n_all, n_seen, n_denovo = len(allN), int((allN >= 1).sum()), int((allN == 0).sum())

def arm_macro(key, mask_fn):
    per = []
    for t in TREES:
        a = z[f"{t}__all__{key}"]
        m = mask_fn(Ncounts[t])
        if m.sum() == 0: continue
        per.append(a[m].mean())
    return macro(per)

import csv
rows = []
# uniform row (0.10 by construction for any subset)
rows.append(("Uniform", 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10))
for key, disp in ARMS:
    allm = arm_macro(key, lambda n: np.ones(len(n), bool))
    seen = arm_macro(key, lambda n: n >= 1)
    den  = arm_macro(key, lambda n: n == 0)
    rows.append((disp, *allm, *seen, *den))

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method",
                "all", "all_lo", "all_hi",
                "nondenovo", "nondenovo_lo", "nondenovo_hi",
                "denovo", "denovo_lo", "denovo_hi"])
    w.writerows(rows)

print(f"partition (pooled events): all={n_all}  non-de-novo(N>=1)={n_seen}  de-novo(N=0)={n_denovo}\n")
hdr = f"{'method':30s} {'non-de-novo':>20s} {'(all)':>8s} {'(de novo)':>10s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    disp = r[0]
    allm, seen, den = r[1], r[4], r[7]
    slo, shi = r[5], r[6]
    print(f"{disp:30s} {seen:6.3f} [{slo:.3f},{shi:.3f}] {allm:8.3f} {den:10.3f}")
print("\nwrote", OUT)
