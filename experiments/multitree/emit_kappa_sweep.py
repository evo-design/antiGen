#!/usr/bin/env python
"""Tidy CSV of the antiGen-prior Smoothed-Counts kappa sweep (recall@10%, macro):
numeric kappa multiplier vs non-de-novo / de novo / all recall, for plotting.
Plus the antiGen(49) and Smoothed-Counts reference levels.
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
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/kappa_sweep_q10.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
KS = [("cpssm_ag_k1",1),("cpssm_ag_k4",4),("cpssm_ag_k5",5),("cpssm_ag_k8",8),
      ("cpssm_ag_k10",10),("cpssm_ag_k16",16),("cpssm_ag_k32",32),("cpssm_ag_k100",100)]
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

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["kappa", "nondenovo", "denovo", "all"])
    for key, km in KS:
        w.writerow([km,
                    macro(key, lambda n: n >= 1),
                    macro(key, lambda n: n == 0),
                    macro(key, lambda n: np.ones(len(n), bool))])
    # references as kappa=NA rows
    for key, lab in [("joint", "antigen"), ("cpssm_dir", "smoothed")]:
        w.writerow([lab,
                    macro(key, lambda n: n >= 1),
                    macro(key, lambda n: n == 0),
                    macro(key, lambda n: np.ones(len(n), bool))])
print("wrote", OUT)
