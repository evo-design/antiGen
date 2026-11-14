#!/usr/bin/env python
"""Macro recall@10% (non-de-novo / de novo / all) for each router v2 variant, as a
tidy frontier CSV: family, knob(lambda), seen, denovo, all. Plus joint & cpssm_dir refs.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_router.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/router_frontier_q10.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
TRANSFORMS = ["bin", "sat2", "sat5", "log30"]
LAMBDAS = {"l025": 0.25, "l1": 1.0, "l4": 4.0, "l10": 10.0, "l25": 25.0, "l50": 50.0, "l100": 100.0}
z = np.load(NPZ)
Nc = {}
for t in TREES:
    tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
    L = len(tr[0].aa_idx); N = count_baselines.fit(tr, L, E)["N"]
    nc = []
    for ex in load_examples(f"{NA}/examples_{t}@{D}_test.txt", ALPHA_SET, None, source=t):
        ev = events_of(ex)
        if not ev: continue
        for (p, a) in ev:
            if a in MASK: continue
            nc.append(float(N[p, a]))
    Nc[t] = np.array(nc)

def macro(key, mask_fn):
    per = [z[f"{t}__all__{key}__q10"][mask_fn(Nc[t])].mean() for t in TREES if mask_fn(Nc[t]).sum()]
    return float(np.mean(per))

def row(key):
    return (macro(key, lambda n: n >= 1), macro(key, lambda n: n == 0),
            macro(key, lambda n: np.ones(len(n), bool)))

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["family", "lam", "seen", "denovo", "all"])
    for tn in TRANSFORMS:
        for lk, lam in LAMBDAS.items():
            s, d, a = row(f"{tn}_{lk}")
            w.writerow([tn, lam, s, d, a])
    for key, fam in [("joint", "antigen"), ("cpssm_dir", "smoothed")]:
        s, d, a = row(key)
        w.writerow([fam, "NA", s, d, a])
print("wrote", OUT)
for tn in TRANSFORMS:
    best = max(LAMBDAS, key=lambda lk: row(f"{tn}_{lk}")[2])
    s, d, a = row(f"{tn}_{best}")
    print(f"{tn:6s} best-all lam={LAMBDAS[best]:<4} seen={s:.3f} denovo={d:.3f} all={a:.3f}")
