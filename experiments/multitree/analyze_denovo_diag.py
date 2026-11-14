#!/usr/bin/env python
"""Analyze where antiGen(49)'s de-novo misses cluster, from denovo_diag.csv."""
import numpy as np, pandas as pd
CSV = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/denovo_diag.csv"
d = pd.read_csv(CSV)
n = len(d); rec = d.caught10.mean()
print(f"de-novo events: {n}   overall antiGen recall@10% = {rec:.3f}\n")

def block(title, grp):
    print(f"== {title} ==")
    g = d.groupby(grp).agg(n=("caught10", "size"), recall=("caught10", "mean"),
                           med_pct=("pct", "median")).reset_index()
    for _, r in g.iterrows():
        print(f"  {str(r[grp]):>14}  n={int(r['n']):4d}  recall={r['recall']:.3f}  med_pct={r['med_pct']:.3f}")
    print()

# per virus
block("by virus", "tree")

# positional mutability: is the de-novo at a conserved vs hypervariable site?
d["pm_bin"] = pd.cut(d.pm_pos, [-1, 0, 1, 5, 20, 1e9],
                     labels=["0 (never)", "1", "2-5", "6-20", "21+"])
block("by positional mutability (training events at that site)", "pm_bin")

# was the position ever mutated in training?
block("by position ever mutated in training", "pos_mut")

# substitution radicalness
d["blo_bin"] = pd.cut(d.blosum, [-100, -1, 0, 100],
                      labels=["radical (<0)", "neutral (=0)", "conservative (>0)"])
block("by BLOSUM62 substitution type", "blo_bin")

# distance to nearest mutated position
d["dist_bin"] = pd.cut(d.dist_mut, [-1, 0, 2, 5, 1e9],
                       labels=["0 (at mut site)", "1-2", "3-5", "6+"])
block("by distance to nearest training-mutated position", "dist_bin")

# how deep are the misses?
miss = d[d.caught10 == 0]
print(f"== miss depth (of {len(miss)} missed de-novo events) ==")
for lo, hi, lab in [(0.10, 0.20, "near-miss 0.10-0.20"), (0.20, 0.35, "0.20-0.35"),
                    (0.35, 0.60, "0.35-0.60"), (0.60, 1.01, "deep 0.60+")]:
    m = (miss.pct >= lo) & (miss.pct < hi)
    print(f"  {lab:20s}  n={int(m.sum()):4d}  ({m.mean()*100:4.1f}% of misses)")
print(f"  median miss percentile = {miss.pct.median():.3f}")
