#!/usr/bin/env python
"""antiGen(49) improvement over the SINGLE-TREE antiGen, stratified by how many
times each test mutation's exact (pos,alt) cell occurred in its TRAINING tree.

Same machinery as stratify_antigen_vs_baselines.py, but the comparator is the
single-virus ESM-650m arm (the `single` rung) instead of the mean of the learned
count baselines. GPU-free: joint and single per-event hit weights already live in
perevent_fig13_sp4e0.npz; we recompute only the training count N[pos,alt] per test
event in the SAME order so they align element-for-element.

Stratifies the ALL-substitutions events by N; N==0 IS the de novo set. Writes
stratify_antigen_vs_single.csv for the single-tree E/F Fig-5 variant.
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
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/stratify_antigen_vs_single.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
COMPARATOR = 'single'
SUB = 'all'
BINS = [(0, 0, "0\n(de novo)"), (1, 1, "1"), (2, 5, "2-5"), (6, 10**9, "6+")]

z = np.load(NPZ)
ncounts, joint, comp = [], [], []
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
    j = z[f"{tree}__{SUB}__joint"]; c = z[f"{tree}__{SUB}__{COMPARATOR}"]
    assert len(nc) == len(j) == len(c), f"{tree}: align {len(nc)} vs {len(j)} vs {len(c)}"
    ncounts.append(np.array(nc)); joint.append(j); comp.append(c)

ncounts = np.concatenate(ncounts); joint = np.concatenate(joint); comp = np.concatenate(comp)
gap = joint - comp
rng = np.random.default_rng(0); B = 5000
def ci(x):
    idx = rng.integers(0, len(x), size=(B, len(x))); bs = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["bin", "n_events", "antigen", "single", "gap", "gap_lo", "gap_hi"])
    for lo, hi, lab in BINS:
        m = (ncounts >= lo) & (ncounts <= hi)
        if not m.any():
            continue
        ga, glo, ghi = ci(gap[m])
        w.writerow([lab, int(m.sum()), float(joint[m].mean()), float(comp[m].mean()), ga, glo, ghi])
        print(f"{lab:12s} n={int(m.sum()):4d}  antiGen={joint[m].mean():.3f}  "
              f"single={comp[m].mean():.3f}  gap={ga:+.3f} [{glo:+.3f},{ghi:+.3f}]")
print("wrote", OUT)
