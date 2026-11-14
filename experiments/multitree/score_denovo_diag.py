#!/usr/bin/env python
"""De-novo miss diagnostic: for every DE-NOVO test event (cell with training N==0,
gap/stop excluded), record antiGen(49)'s rank-percentile of that cell among valid cells
plus per-event features, so we can see WHERE antiGen's de-novo misses cluster.

Percentile = (# valid cells ranked strictly above this cell) / (# valid cells); <=0.10
means caught at recall@10%. Features:
  pm_pos   = total training mutation events at this POSITION (positional mutability)
  pos_mut  = was this position mutated at all in training (pm_pos>0)
  dist_mut = sequence distance to the nearest training-mutated position
  blosum   = BLOSUM62(wt, alt)  (radical<0 vs conservative>=0)
  wt, alt  = residues
Writes denovo_diag.csv (one row per de-novo test event).
"""
import os, sys, glob, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, D, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/denovo_diag.csv"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']

# BLOSUM62 via Biopython if available, else a minimal fallback (0 for all).
try:
    from Bio.Align import substitution_matrices
    _BL = substitution_matrices.load("BLOSUM62")
    def blosum(wt, al):
        try: return float(_BL[wt, al])
        except Exception: return float('nan')
except Exception:
    def blosum(wt, al): return float('nan')

def main():
    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck)
    rows = []
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        N = count_baselines.fit(tr, L, E)["N"]
        pm = N.sum(1)                                   # (L,) positional mutability
        mutated_pos = torch.nonzero(pm > 0).flatten().tolist()
        mut_set = np.array(mutated_pos) if mutated_pos else np.array([-10**6])
        p = f"{NA}/examples_{tree}@{D}_test.txt"
        if not os.path.exists(p) or os.path.getsize(p) == 0: continue
        for ex in load_examples(p, ALPHA_SET, None, source=tree):
            ev = events_of(ex)
            if not ev: continue
            aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
            with torch.no_grad():
                js = jm(ab).float().squeeze(0).cpu()
            s = js.clone()
            for c in MASK: s[:, c] = float('nan')
            s[torch.arange(L), aa] = float('nan')
            flat = s.flatten(); valid = ~torch.isnan(flat)
            nv = int(valid.sum())
            vflat = flat[valid]
            for (pos, al) in ev:
                if al in MASK or al == int(aa[pos]): continue
                if float(N[pos, al]) != 0.0: continue   # DE NOVO only
                sv = float(js[pos, al])
                pct = float((vflat > sv).sum()) / nv
                dist = int(np.min(np.abs(mut_set - pos)))
                rows.append([tree, int(pos), ALPHA_SET[int(aa[pos])], ALPHA_SET[al],
                             float(pm[pos]), int(pm[pos] > 0), dist,
                             blosum(ALPHA_SET[int(aa[pos])], ALPHA_SET[al]),
                             sv, pct, int(pct < 0.10)])
        print(f"done {tree}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tree","pos","wt","alt","pm_pos","pos_mut","dist_mut","blosum",
                    "score","pct","caught10"])
        w.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} de-novo events)")

if __name__ == '__main__':
    main()
