#!/usr/bin/env python
"""Non-de-novo (seen >=1x in training) Fig-5 method table at recall@5% and @1%
(and @10% as a reproduction check). Same partition machinery as table_nondenovo.py;
reads the budget npz keyed <tree>__all__<arm>__q{01,05,10}.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_fig13_sp4e0_budgets.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUTDIR = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
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

# training count per test event, same order as the __all__ per-event arrays
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
for q, qpct in [("05", "5%"), ("01", "1%")]:
    rows = [("Uniform", 0.01 if q == "01" else 0.05,) * 1]  # placeholder, filled below
    rows = []
    unif = 0.01 if q == "01" else 0.05
    rows.append(("Uniform", unif, unif, unif, unif, unif, unif, unif, unif, unif))
    for key, disp in ARMS:
        allm = arm_macro(key, q, lambda n: np.ones(len(n), bool))
        seen = arm_macro(key, q, lambda n: n >= 1)
        den  = arm_macro(key, q, lambda n: n == 0)
        rows.append((disp, *allm, *seen, *den))
    # sort by non-de-novo desc (uniform stays at bottom)
    body = sorted([r for r in rows if r[0] != "Uniform"], key=lambda r: -r[4])
    rows = body + [r for r in rows if r[0] == "Uniform"]
    out = f"{OUTDIR}/table_nondenovo_q{q}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "all", "all_lo", "all_hi",
                    "nondenovo", "nondenovo_lo", "nondenovo_hi",
                    "denovo", "denovo_lo", "denovo_hi"])
        w.writerows(rows)
    print(f"\n=== recall@{qpct}  (non-de-novo = seen >=1x; chance = {unif:.2f}) ===")
    hdr = f"{'method':30s} {'non-de-novo':>22s} {'all':>7s} {'de novo':>8s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:30s} {r[4]:6.3f} [{r[5]:.3f},{r[6]:.3f}] {r[1]:7.3f} {r[7]:8.3f}")
    print("wrote", out)
