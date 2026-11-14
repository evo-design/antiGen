#!/usr/bin/env python
"""Append the `vs 13-tree model` contrast (joint - joint13) to the sp4e0 deltas.csv.

Non-destructive, same pattern as append_zeroshot_ladder.py: the collector would
regenerate deltas.csv and drop this contrast (its CONTR dict has only single /
where x which / PSSM+Dirichlet), so we compute the per-tree and POOLED(=macro)
deltas exactly as collect_fig13_noattach_sp4e0.py does and splice the rows in.

joint = antiGen (49 trees); joint13 = antiGen (13 eval trees). Per-tree CI is the
event-level paired bootstrap; POOLED is the macro (per-tree mean) with a tree-level
bootstrap, matching the ladder's macro pooling.
"""
import numpy as np, csv
R = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface"
NPZ = f"{R}/perevent_fig13_sp4e0.npz"
DEL = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/deltas.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
UNIV, MET, RULE, EVAL, B = "standard", "hit10", "pooled", "Temporal holdout", 2000
CNAME = "vs 13-tree model"
Z = np.load(NPZ)
rng = np.random.default_rng(0)

def get(t, sub, rung):
    k = f"{t}__{sub}__{rung}"
    return Z[k] if k in Z.files else None

def paired_ci(a, b):
    d = a - b
    if len(d) == 0: return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, len(d), size=(B, len(d))); bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def macro_ci(per):
    per = np.asarray([p for p in per if p == p])
    if len(per) == 0: return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, len(per), size=(B, len(per))); bs = per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# keep everything except a stale copy of this contrast
rows, header = [], None
with open(DEL) as fh:
    r = csv.reader(fh); header = next(r)
    for row in r:
        if row[1] != CNAME:
            rows.append(row)

for sub in ("all", "novel"):
    per_tree = []
    for t in TREES:
        j, c = get(t, sub, "joint"), get(t, sub, "joint13")
        if j is None or c is None or len(j) == 0: continue
        est, lo, hi = paired_ci(j, c)
        rows.append([UNIV, CNAME, EVAL, RULE, MET, t, sub, est, lo, hi, 0])
        per_tree.append(float((j - c).mean()))
    if per_tree:
        est, lo, hi = macro_ci(per_tree)
        rows.append([UNIV, CNAME, EVAL, RULE, MET, "POOLED", sub, est, lo, hi, 1])
        print(f"{sub:5s} POOLED joint-joint13 = {est:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n_trees={len(per_tree)})")

with open(DEL, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(header); w.writerows(rows)
print(f"wrote {DEL}  ({len(rows)} data rows)")
