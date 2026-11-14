#!/usr/bin/env python
"""Report the hypervariability-gated correction: for each (tau, beta), standalone
corrected-antiGen (seen/denovo/all) and the router de-novo CEILING (phi=0)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_denovo_gated.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/denovo_gated.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
TAUS = {"t0": 0, "t2": 2, "t5": 5, "t10": 10}
BETAS = {"b02": 0.2, "b05": 0.5, "b10": 1.0, "b20": 2.0}
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

seen_m = lambda n: n >= 1
dn_m   = lambda n: n == 0
all_m  = lambda n: np.ones(len(n), bool)

base_dn = macro("corr_base", dn_m); base_rc = macro("rdn_base", dn_m)
base_all = macro("corr_base", all_m); base_seen = macro("corr_base", seen_m)

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["tau","beta","seen","denovo","all","router_denovo_ceiling"])
    print(f"plain antiGen (beta=0):  seen={base_seen:.3f}  denovo={base_dn:.3f}  "
          f"all={base_all:.3f} | router de-novo ceiling={base_rc:.3f}")
    print("gate = boost applies only where positional mutability pm >= tau\n")
    print(f"{'tau':>4} {'beta':>5} {'seen':>7} {'denovo':>7} {'all':>7} | {'router de-novo ceiling':>22}")
    w.writerow([0, 0.0, base_seen, base_dn, base_all, base_rc])
    for tk, tau in TAUS.items():
        for bk, beta in BETAS.items():
            s = macro(f"corr_{tk}_{bk}", seen_m)
            dn = macro(f"corr_{tk}_{bk}", dn_m)
            al = macro(f"corr_{tk}_{bk}", all_m)
            rc = macro(f"rdn_{tk}_{bk}", dn_m)
            star = "  <-- beats 0.802" if rc > base_rc + 1e-6 else ""
            w.writerow([tau, beta, s, dn, al, rc])
            lab = "global" if tau == 0 else f"pm>={tau}"
            print(f"{lab:>4} {beta:>5} {s:7.3f} {dn:7.3f} {al:7.3f} | {rc:22.3f}{star}")
print("\nwrote", OUT)
