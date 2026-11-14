#!/usr/bin/env python
"""Hypervariability-GATED anti-conservatism correction.

The global correction (score_denovo_correct.py) failed: a blanket -beta*BLOSUM
penalty demotes the many correctly-ranked conservative de-novo cells more than it
rescues the radical ones. Diagnosis: antiGen's conservatism reflects a TRUE base
rate; a radical change is only escape-relevant at an antigenically SELECTED site.

Here we gate the boost by POSITIONAL MUTABILITY pm_i = sum_a N[i,a] (total training
events at position i) -- a structure-free proxy for "site under diversifying/antigenic
selection." The boost applies only where pm_i >= tau; elsewhere antiGen is untouched:
  corr(i,a) = antiGen_logit(i,a) - beta * BLOSUM62(wt_i, a) * 1[pm_i >= tau]
Radical de-novo subs (BLOSUM<0) at hypervariable sites get +beta*|BLOSUM|; radical
subs at conserved sites keep antiGen's (correctly conservative) ranking.

Scored at recall@10% (macro over 13 trees), two ways:
  A) standalone corrected antiGen over ALL cells   -> keys  corr_<tau>_<beta>
  B) router de-novo pool (N<1), phi=0, corrected    -> keys  rdn_<tau>_<beta>
tau=0 reproduces the global (ungated) correction; beta=0 reproduces plain antiGen.
Writes perevent_denovo_gated.npz keyed <tree>__<sub>__<key>__q10.
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
OUT = f"{R}/fc13_core_surface/perevent_denovo_gated.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10
TAUS = {"t0": 0, "t2": 2, "t5": 5, "t10": 10}      # pm>=tau gate (t0 = global/ungated)
BETAS = {"b02": 0.2, "b05": 0.5, "b10": 1.0, "b20": 2.0}

# BLOSUM62 as an (E,E) tensor over ALPHA_SET; * and - (MASK) rows/cols = 0.
_BL = substitution_matrices.load("BLOSUM62")
B = torch.zeros(E, E)
for i, ai in enumerate(ALPHA_SET):
    for j, aj in enumerate(ALPHA_SET):
        try: B[i, j] = float(_BL[ai, aj])
        except Exception: B[i, j] = 0.0

def topk_weight(scoremap, pool_mask, kf):
    W = torch.zeros_like(scoremap)
    if kf <= 0 or not bool(pool_mask.any()): return W
    vals = scoremap[pool_mask]; n = vals.numel()
    if kf >= n: W[pool_mask] = 1.0; return W
    thr = torch.quantile(vals, 1.0 - kf / n)
    above = (scoremap > thr) & pool_mask; tie = (scoremap == thr) & pool_mask
    na = int(above.sum()); nt = int(tie.sum())
    frac = max(0.0, (kf - na)) / nt if nt > 0 else 0.0
    W[above] = 1.0; W[tie] = frac
    return W

def main():
    out = {}
    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck)
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx); N = count_baselines.fit(tr, L, E)["N"]
        pm = N.sum(1)                                   # (L,) positional mutability
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            keys = ([f"corr_{tk}_{bk}" for tk in TAUS for bk in BETAS]
                    + [f"rdn_{tk}_{bk}" for tk in TAUS for bk in BETAS]
                    + ["corr_base", "rdn_base"])        # beta=0 baselines (gate-independent)
            rows = {k: [] for k in keys}
            for ex in te:
                ev = [(pp, aa) for (pp, aa) in events_of(ex) if aa not in MASK]
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js = jm(ab).float().squeeze(0).cpu()
                Bmap = B[aa]                                    # (L,E) = BLOSUM(wt_i, a)
                valid = torch.ones(L, E, dtype=torch.bool)
                valid[torch.arange(L), aa] = False
                for c in MASK: valid[:, c] = False
                unseen = valid & (N < 1)
                nv = int(valid.sum()); budget = Q * nv
                # beta=0 baselines once
                rows["corr_base"] += per_event_weights(js, aa, ev)
                Wu0 = topk_weight(js, unseen, budget)
                rows["rdn_base"] += [float(Wu0[pp, a]) for (pp, a) in ev]
                for tk, tau in TAUS.items():
                    gate = (pm >= tau).float().unsqueeze(1)     # (L,1); tau=0 -> all ones
                    for bk, beta in BETAS.items():
                        corr = js - beta * Bmap * gate
                        rows[f"corr_{tk}_{bk}"] += per_event_weights(corr, aa, ev)
                        Wu = topk_weight(corr, unseen, budget)
                        rows[f"rdn_{tk}_{bk}"] += [float(Wu[pp, a]) for (pp, a) in ev]
            for k, v in rows.items():
                out[f"{tree}__{sub}__{k}__q10"] = np.array(v, dtype=np.float32)
        print(f"done {tree}", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

if __name__ == '__main__':
    main()
