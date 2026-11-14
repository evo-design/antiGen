#!/usr/bin/env python
"""Macro recall@10% (seen / de novo / all) for each hard-router phi, tidy CSV."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_router_hard.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/router_hard_frontier.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
PHIS = {"p000":0.0,"p002":0.02,"p005":0.05,"p008":0.08,"p012":0.12,
        "p020":0.20,"p035":0.35,"p050":0.50,"p075":0.75,"p100":1.0}
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

def rowfor(key):
    return (macro(key, lambda n: n >= 1), macro(key, lambda n: n == 0),
            macro(key, lambda n: np.ones(len(n), bool)))

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["phi", "seen", "denovo", "all"])
    for pk, phi in PHIS.items():
        s, d, a = rowfor(f"phi{pk}")
        w.writerow([phi, s, d, a])
    for key, fam in [("joint", "antigen"), ("cpssm_dir", "smoothed")]:
        s, d, a = rowfor(key); w.writerow([fam, s, d, a])
print("wrote", OUT)
print(f"{'phi':>5} {'seen':>7} {'denovo':>7} {'all':>7}")
for pk, phi in PHIS.items():
    s, d, a = rowfor(f"phi{pk}")
    print(f"{phi:>5} {s:7.3f} {d:7.3f} {a:7.3f}")
best = max(PHIS, key=lambda pk: rowfor(f"phi{pk}")[2])
print("best all at phi=", PHIS[best], "->", rowfor(f"phi{best}"))
