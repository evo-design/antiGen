#!/usr/bin/env python
"""Append the ESM zero-shot rung to the sp4e0 Fig-5 ladder.csv.

Non-destructive: the collector (collect_fig13_noattach_sp4e0.py) would regenerate
ladder.csv and drop the manually-added joint13 rows, so instead we bootstrap the
zeroshot macro recall exactly as the collector does (per-tree means -> tree
bootstrap CI) and append the two rows (all, novel) in place.
"""
import numpy as np, csv, os
R = "/large_storage/hielab/aakarsh/multitree"
NPZ = f"{R}/fc13_core_surface/perevent_fig13_sp4e0.npz"
LAD = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/ladder.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
UNIV, MET, RULE, EVAL, RUNG, B = "standard", "hit10", "pooled", "Temporal holdout", "zeroshot", 2000

Z = np.load(NPZ)
rng = np.random.default_rng(0)

def macro_ci(sub):
    per = [Z[f"{t}__{sub}__{RUNG}"].mean() for t in TREES
           if f"{t}__{sub}__{RUNG}" in Z.files and len(Z[f"{t}__{sub}__{RUNG}"]) > 0]
    per = np.asarray(per)
    idx = rng.integers(0, len(per), size=(B, len(per)))
    bs = per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# Drop any stale zeroshot rows, keep everything else.
rows, header = [], None
with open(LAD) as fh:
    r = csv.reader(fh); header = next(r)
    for row in r:
        if not (row[5] == RUNG and row[4] in ("all", "novel")):
            rows.append(row)

for sub in ("all", "novel"):
    m, lo, hi = macro_ci(sub)
    rows.append([UNIV, MET, RULE, EVAL, sub, RUNG, m, lo, hi])
    print(f"zeroshot {sub:5s}  macro recall@10% = {m:.4f}  [{lo:.4f}, {hi:.4f}]")

with open(LAD, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(header); w.writerows(rows)
print(f"wrote {LAD}  ({len(rows)} data rows)")
