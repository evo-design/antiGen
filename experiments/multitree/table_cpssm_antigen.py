#!/usr/bin/env python
"""Compare Smoothed-Counts-with-antiGen-prior against plain antiGen(49) and plain
Smoothed Counts, at recall@{10,5,1}%, split all / non-de-novo (seen>=1x) / de novo.
Reads perevent_cpssm_antigen.npz keyed <tree>__<sub>__<arm>__q{NN}. Macro over trees.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_cpssm_antigen.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUTDIR = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
ARMS = [
    ("joint",        "antiGen (49 trees)"),
    ("cpssm_dir",    "Smoothed Counts [Site x Sub prior]"),
    ("cpssm_ag_k1",  "Smoothed+antiGen prior (k=N)"),
    ("cpssm_ag_k4",  "Smoothed+antiGen prior (k=4N)"),
    ("cpssm_ag_k5",  "Smoothed+antiGen prior (k=5N)"),
    ("cpssm_ag_k8",  "Smoothed+antiGen prior (k=8N)"),
    ("cpssm_ag_k10", "Smoothed+antiGen prior (k=10N)"),
    ("cpssm_ag_k16", "Smoothed+antiGen prior (k=16N)"),
    ("cpssm_ag_k32", "Smoothed+antiGen prior (k=32N)"),
    ("cpssm_ag_k100","Smoothed+antiGen prior (k=100N)"),
]
z = np.load(NPZ)
Nc = {}
for t in TREES:
    tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
    L = len(tr[0].aa_idx); N = count_baselines.fit(tr, L, E)["N"]
    nc = []
    for ex in load_examples(f"{NA}/examples_{t}@{D}_test.txt", ALPHA_SET, None, source=t):
        ev = events_of(ex)
        if not ev: continue
        for (pos, a) in ev:
            if a in MASK: continue
            nc.append(float(N[pos, a]))
    Nc[t] = np.array(nc)

rng = np.random.default_rng(0); B = 2000
def macro(per):
    per = np.asarray([p for p in per if p == p])
    if len(per) == 0: return (float('nan'),) * 3
    idx = rng.integers(0, len(per), size=(B, len(per))); bs = per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def arm_macro(key, q, mask_fn):
    per = []
    for t in TREES:
        a = z[f"{t}__all__{key}__q{q}"]; m = mask_fn(Nc[t])
        if m.sum() == 0: continue
        per.append(a[m].mean())
    return macro(per)

import csv
for q, qpct in [("10", "10%"), ("05", "5%"), ("01", "1%")]:
    print(f"\n=== recall@{qpct} ===")
    hdr = f"{'method':36s} {'all':>18s} {'non-de-novo':>18s} {'de novo':>18s}"
    print(hdr); print("-" * len(hdr))
    out = f"{OUTDIR}/table_cpssm_antigen_q{q}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method","all","all_lo","all_hi","nondenovo","nondenovo_lo","nondenovo_hi",
                    "denovo","denovo_lo","denovo_hi"])
        for key, disp in ARMS:
            al = arm_macro(key, q, lambda n: np.ones(len(n), bool))
            se = arm_macro(key, q, lambda n: n >= 1)
            dn = arm_macro(key, q, lambda n: n == 0)
            w.writerow([disp, *al, *se, *dn])
            print(f"{disp:36s} {al[0]:.3f}[{al[1]:.3f},{al[2]:.3f}] "
                  f"{se[0]:.3f}[{se[1]:.3f},{se[2]:.3f}] {dn[0]:.3f}[{dn[1]:.3f},{dn[2]:.3f}]")
    print("wrote", out)
