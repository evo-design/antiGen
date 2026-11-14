#!/usr/bin/env python
"""Whole-spike accessibility gate on SC2 (ncov): corr = antiGen_logit + beta * RSA_centered.
RSA is per-position surface exposure (6VXX trimer), centered at its median so exposed sites
are boosted, buried demoted, and structurally-missing residues (mostly flexible exposed
loops) are left unchanged (0). Scored WHOLE-SPIKE at recall@10%, stratified by region
(RBD/non-RBD) and radicalness (BLOSUM62<0), to test whether a surface gate recovers the
non-RBD radical de-novo misses that escape maps can't reach. Prints the table directly.
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from Bio.Align import substitution_matrices
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, per_event_weights, E, ALPHA_SET, D, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
ACC = f"{R}/fc13_core_surface/accessibility_sc2.npz"
TREE = "ncov"; RBD_LO, RBD_HI = 330, 530
BETAS = {"b0": 0.0, "b1": 1.0, "b2": 2.0, "b4": 4.0, "b8": 8.0}
_BL = substitution_matrices.load("BLOSUM62")

def main():
    tr = load_examples(f"{NA}/examples_{TREE}@{D}_train.txt", ALPHA_SET, None, source=TREE)
    L = len(tr[0].aa_idx); N = count_baselines.fit(tr, L, E)["N"]
    z = np.load(ACC); acc = z["acc"].astype(np.float32)
    med = np.nanmedian(acc)
    accc = np.where(np.isnan(acc), 0.0, acc - med)            # centered; missing -> 0
    accc = torch.tensor(accc)
    jm = load_esm(glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0])
    te = load_examples(f"{NA}/examples_{TREE}@{D}_test.txt", ALPHA_SET, None, source=TREE)
    W = {bk: [] for bk in BETAS}
    m_n, m_rbd, m_rad = [], [], []
    for ex in te:
        ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
        if not ev: continue
        aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
        with torch.no_grad():
            js = jm(ab).float().squeeze(0).cpu()
        for bk, beta in BETAS.items():
            W[bk] += per_event_weights(js + beta * accc.view(L, 1), aa, ev)
        for (p, a) in ev:
            m_n.append(float(N[p, a])); m_rbd.append(RBD_LO <= p <= RBD_HI)
            try: m_rad.append(float(_BL[ALPHA_SET[int(aa[p])], ALPHA_SET[a]]) < 0)
            except Exception: m_rad.append(False)
    n = np.array(m_n); rbd = np.array(m_rbd); rad = np.array(m_rad); dn = (n == 0)
    def rec(bk, mask):
        w = np.array(W[bk]); return float(w[mask].mean()) if mask.sum() else float('nan')
    print(f"ncov whole-spike: events={len(n)} de-novo={dn.sum()} "
          f"(RBD de-novo={int((dn&rbd).sum())}, non-RBD radical de-novo={int((dn&~rbd&rad).sum())})\n")
    hdr = ["beta","denovo_all","dn_RBD","dn_nonRBD","dn_nonRBD_radical","dn_conserv","all_events"]
    print("  ".join(f"{h:>17}" for h in hdr))
    import csv
    OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/accessibility_sc2.csv"
    with open(OUT, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(hdr)
        for bk, beta in BETAS.items():
            row = [beta, rec(bk, dn), rec(bk, dn&rbd), rec(bk, dn&~rbd),
                   rec(bk, dn&~rbd&rad), rec(bk, dn&~rad), rec(bk, np.ones(len(n),bool))]
            wr.writerow(row)
            print("  ".join(f"{row[0]:>17.1f}" if i==0 else f"{v:>17.3f}" for i,v in enumerate(row)))
    print("\nwrote", OUT)

if __name__ == '__main__':
    main()
