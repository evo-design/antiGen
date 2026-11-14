#!/usr/bin/env python
"""antiGen(49) improvement over the averaged learned baselines, stratified by how
many times each test mutation's exact (pos,alt) cell occurred in its TRAINING tree.

GPU-free: the per-event hit weights for joint and the five learned baselines already
live in perevent_fig13_sp4e0.npz, in event order. We recompute only the training
count N[pos,alt] per test event in that SAME order (same load_examples / events_of /
MASK-skip as the scorer), so the counts align element-for-element with the hit arrays.

Baselines averaged: PSSM, Event Counts, Site x Substitution, Smoothed Counts,
Where x which  (Uniform / ESM-2 zero-shot / Codon-aware BLOSUM62 are not learned).

Stratifies the ALL-substitutions events by N into bins; N==0 IS the de novo set.
For each bin (pooled over the 13 viruses): antiGen recall, mean-baseline recall,
the gap, and an event bootstrap CI. Writes stratify_antigen_vs_baselines.csv.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, E, ALPHA_SET, D, MASK

NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_fig13_sp4e0.npz"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/stratify_antigen_vs_baselines.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
BASE = ['pssm','frequency','site_prop','cpssm_dir','where_times_which']
SUB = 'all'

# Bins on the training event count N[pos,alt]: (lo, hi_inclusive, label).
# The signal is all at the low end (the gap is spent by N=1), so the upper
# tail is collapsed into one 6+ bin rather than split at arbitrary boundaries.
BINS = [(0, 0, "0\n(de novo)"), (1, 1, "1"), (2, 5, "2-5"), (6, 10**9, "6+")]

z = np.load(NPZ)

ncounts, joint, bavg = [], [], []
for tree in TREES:
    tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
    L = len(tr[0].aa_idx)
    st = count_baselines.fit(tr, L, E)
    N = st["N"]                                   # (L,E) training event counts
    nc = []
    for ex in load_examples(f"{NA}/examples_{tree}@{D}_{'test'}.txt", ALPHA_SET, None, source=tree):
        ev = events_of(ex)
        if not ev:
            continue
        for (pos, a) in ev:
            if a in MASK:
                continue
            nc.append(float(N[pos, a]))
    j = z[f"{tree}__{SUB}__joint"]
    b = np.mean(np.stack([z[f"{tree}__{SUB}__{r}"] for r in BASE], 0), 0)
    assert len(nc) == len(j) == len(b), f"{tree}: align {len(nc)} vs {len(j)} vs {len(b)}"
    ncounts.append(np.array(nc)); joint.append(j); bavg.append(b)

ncounts = np.concatenate(ncounts); joint = np.concatenate(joint); bavg = np.concatenate(bavg)
gap = joint - bavg
print(f"total events: {len(ncounts)}")

rng = np.random.default_rng(0); B = 5000
def ci(x):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["bin", "n_events", "antigen", "baseline_avg",
                "gap", "gap_lo", "gap_hi"])
    for lo, hi, lab in BINS:
        m = (ncounts >= lo) & (ncounts <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        ga, glo, ghi = ci(gap[m])
        w.writerow([lab, n, float(joint[m].mean()), float(bavg[m].mean()), ga, glo, ghi])
        print(f"{lab:12s} n={n:4d}  antiGen={joint[m].mean():.3f}  "
              f"base={bavg[m].mean():.3f}  gap={ga:+.3f} [{glo:+.3f},{ghi:+.3f}]")
print("wrote", OUT)
