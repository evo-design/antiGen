#!/usr/bin/env python
"""Per-virus gap: antiGen (49 trees) minus the mean of the LEARNED count baselines.

The learned baselines are the five rungs in Fig-5 panels A/B that are fit on the
training trees -- PSSM, Event Counts, Site x Substitution, Smoothed Counts and
Where x which. Excluded by construction: Uniform, ESM-2 zero-shot and Codon-aware
BLOSUM62, none of which is learned from these trees.

The gap is formed at the EVENT level so the CI is paired, exactly like the forests
in panels C/D: for each observed test mutation, antiGen's hit weight minus the mean
of the five baselines' hit weights; the per-virus gap is the mean over its events
(event bootstrap CI), and POOLED is the mean of the per-virus gaps (tree bootstrap CI).
Writes antigen_vs_baselines.csv for fig_antigen_vs_learned_baselines.R.
"""
import numpy as np, os, csv
NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_fig13_sp4e0.npz"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/antigen_vs_baselines.csv"
TREES = ['ncov','h5nx','h3n2_ha','h1n1pdm_ha','vic_ha','rsv_a','rsv_b','measles',
         'hmpv_a_g','denv1','denv2','denv3','denv4']
NAME = {'ncov':'SARS-CoV-2 S','h5nx':'H5Nx HA','h3n2_ha':'H3N2 HA','h1n1pdm_ha':'H1N1pdm HA',
        'vic_ha':'B/Victoria HA','rsv_a':'RSV-A F','rsv_b':'RSV-B F','measles':'Measles H',
        'hmpv_a_g':'hMPV-A G','denv1':'DENV-1 E','denv2':'DENV-2 E','denv3':'DENV-3 E','denv4':'DENV-4 E'}
BASE = ['pssm','frequency','site_prop','cpssm_dir','where_times_which']   # the learned baselines
B = 2000
z = np.load(NPZ)
rng = np.random.default_rng(0)

def event_ci(d):
    idx = rng.integers(0, len(d), size=(B, len(d))); bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def macro_ci(per):
    per = np.asarray(per); idx = rng.integers(0, len(per), size=(B, len(per))); bs = per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

rows = []
for sub in ('all', 'novel'):
    per_tree = []
    for t in TREES:
        jk = f'{t}__{sub}__joint'
        if jk not in z.files or len(z[jk]) == 0:
            continue
        j = z[jk]
        bmaps = [z[f'{t}__{sub}__{b}'] for b in BASE if f'{t}__{sub}__{b}' in z.files]
        bavg = np.mean(np.stack(bmaps, 0), 0)         # per-event mean over the 5 baselines
        d = j - bavg                                  # paired per-event gap
        m, lo, hi = event_ci(d)
        rows.append([sub, t, NAME[t], m, lo, hi, 0, int(lo > 0 or hi < 0)])
        per_tree.append(float(d.mean()))
    m, lo, hi = macro_ci(per_tree)
    rows.append([sub, 'POOLED', 'Pooled', m, lo, hi, 1, int(lo > 0 or hi < 0)])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['subset','tree','tree_lab','est','lo','hi','pooled','clears'])
    w.writerows(rows)
print(f"wrote {OUT}  ({len(rows)} rows)")
for r in rows:
    if r[0] == 'all':
        print(f"  {r[2]:15s} all   {r[3]:+.3f} [{r[4]:+.3f},{r[5]:+.3f}]{' *' if r[7] else ''}")
