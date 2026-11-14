#!/usr/bin/env python
"""Model-free recurrence: for each virus, among the (pos -> alt) substitutions that
occur >=1 time in the TRAINING tree, what fraction also occur at least once in the
forecast TEST window -- overall, and stratified by training multiplicity k.

A "mutation" is a unique (position, alternate-residue) cell; gap/stop alts excluded.
N_train[pos,alt] = event count over the train split (count_baselines.fit); N_test is
the same sum over the test split. P_recur(k) = P(N_test>=1 | N_train == k or in bin).
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import E, ALPHA_SET, D, MASK

NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUTDIR = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
ABBR = {'ncov':'SC2','h5nx':'H5Nx','h3n2_ha':'H3N2','h1n1pdm_ha':'H1N1','vic_ha':'Vic',
        'rsv_a':'RSV-A','rsv_b':'RSV-B','measles':'MeV','hmpv_a_g':'hMPV',
        'denv1':'D1','denv2':'D2','denv3':'D3','denv4':'D4'}
BINS = [(1,1,"1"),(2,3,"2-3"),(4,9,"4-9"),(10,29,"10-29"),(30,10**9,"30+")]

def mask_cols(M):
    M = M.clone()
    for c in MASK: M[:, c] = 0.0
    return M

per_virus = {}   # tree -> (Ntr flat, Nte flat) over non-mask cells with Ntr>=1
for t in TREES:
    tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
    L = len(tr[0].aa_idx)
    Ntr = mask_cols(count_baselines.fit(tr, L, E)["N"])
    Nte = torch.zeros(L, E)
    te = load_examples(f"{NA}/examples_{t}@{D}_test.txt", ALPHA_SET, None, source=t)
    for ex in te:
        Nte += ex.mat
    Nte = mask_cols(Nte)
    sel = Ntr >= 1
    per_virus[t] = (Ntr[sel].numpy().astype(int), Nte[sel].numpy().astype(int), len(te))

import csv
# ---- per-virus overall recurrence ----
print(f"{'virus':6s} {'n_train_cells':>13s} {'n_test_ex':>9s} {'P(recur in test)':>16s}")
print("-" * 50)
ov = open(f"{OUTDIR}/recurrence_overall.csv", "w", newline=""); ow = csv.writer(ov)
ow.writerow(["tree","abbr","n_train_cells","n_test_examples","p_recur"])
for t in TREES:
    tr, te, nex = per_virus[t]
    p = float((te >= 1).mean()) if len(tr) else float('nan')
    ow.writerow([t, ABBR[t], len(tr), nex, p])
    print(f"{ABBR[t]:6s} {len(tr):13d} {nex:9d} {p:16.3f}")
ov.close()

# pooled overall
allt = np.concatenate([per_virus[t][0] for t in TREES])
alle = np.concatenate([per_virus[t][1] for t in TREES])
print("-" * 50)
print(f"{'POOL':6s} {len(allt):13d} {'':9s} {float((alle>=1).mean()):16.3f}")

# ---- stratified by training multiplicity (pooled across viruses) ----
print("\nPooled, by training multiplicity k:")
print(f"{'bin':8s} {'n_cells':>8s} {'P(recur in test)':>16s} {'mean N_test':>12s}")
sv = open(f"{OUTDIR}/recurrence_by_count.csv", "w", newline=""); sw = csv.writer(sv)
sw.writerow(["bin","n_cells","p_recur","mean_n_test"])
for lo, hi, lab in BINS:
    m = (allt >= lo) & (allt <= hi)
    if not m.any(): continue
    p = float((alle[m] >= 1).mean()); mn = float(alle[m].mean())
    sw.writerow([lab, int(m.sum()), p, mn])
    print(f"{lab:8s} {int(m.sum()):8d} {p:16.3f} {mn:12.2f}")
sv.close()
print(f"\nwrote {OUTDIR}/recurrence_overall.csv  and  recurrence_by_count.csv")
