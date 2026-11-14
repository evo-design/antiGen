#!/usr/bin/env python
"""Beta sweep of the anti-conservatism correction: standalone corrected-antiGen
(seen/denovo/all) and the router de-novo CEILING (phi=0, corrected ranker)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_denovo_correct.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/denovo_correct.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
BETAS = {"b00": 0.0, "b02": 0.2, "b05": 0.5, "b10": 1.0, "b20": 2.0, "b40": 4.0, "b80": 8.0}
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
    per = [z[f"{t}__all__{key}__q10"][mask_fn(Nc[t])].mean() for t in TREES
           if f"{t}__all__{key}__q10" in z.files and mask_fn(Nc[t]).sum()]
    return float(np.mean(per)) if per else float('nan')

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["beta", "seen", "denovo", "all", "router_denovo_ceiling"])
    print(f"{'beta':>5} {'seen':>7} {'denovo':>7} {'all':>7} | {'router de-novo ceiling':>22}")
    for bk, beta in BETAS.items():
        s = macro(f"corr_{bk}", lambda n: n >= 1)
        dn = macro(f"corr_{bk}", lambda n: n == 0)
        al = macro(f"corr_{bk}", lambda n: np.ones(len(n), bool))
        rc = macro(f"rdn_{bk}", lambda n: n == 0)
        w.writerow([beta, s, dn, al, rc])
        print(f"{beta:>5} {s:7.3f} {dn:7.3f} {al:7.3f} | {rc:22.3f}")
print("wrote", OUT)
