#!/usr/bin/env python
"""Anti-conservatism correction to antiGen's de-novo ranking.

Diagnosis: antiGen under-ranks biochemically RADICAL substitutions (BLOSUM<0) -- its
recall on those de-novo events is 0.49 vs 0.74 for conservative ones. Correct it by
  corrected(i,a) = antiGen_logit(i,a) - beta * BLOSUM62(wt_i, a)
so radical subs (BLOSUM<0) get +beta*|BLOSUM| and conservative subs get penalized.

Scored two ways at recall@10% (macro):
  A) standalone corrected antiGen over ALL cells          -> keys  corr_b<NN>
  B) hard router de-novo pool ranked by corrected antiGen, all budget to unseen (phi=0)
     -> keys  rdn_b<NN>   (does the de-novo CEILING rise above antiGen's 0.80?)
beta=0 reproduces plain antiGen / the plain-router ceiling.
Writes perevent_denovo_correct.npz keyed <tree>__<sub>__<key>__q10.
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
OUT = f"{R}/fc13_core_surface/perevent_denovo_correct.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10
BETAS = {"b00": 0.0, "b02": 0.2, "b05": 0.5, "b10": 1.0, "b20": 2.0, "b40": 4.0, "b80": 8.0}

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
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            keys = [f"corr_{bk}" for bk in BETAS] + [f"rdn_{bk}" for bk in BETAS]
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
                for bk, beta in BETAS.items():
                    corr = js - beta * Bmap
                    rows[f"corr_{bk}"] += per_event_weights(corr, aa, ev)       # standalone, global top-10%
                    Wu = topk_weight(corr, unseen, budget)                       # router de-novo pool, phi=0
                    rows[f"rdn_{bk}"] += [float(Wu[pp, a]) for (pp, a) in ev]
            for k, v in rows.items():
                out[f"{tree}__{sub}__{k}__q10"] = np.array(v, dtype=np.float32)
        print(f"done {tree}", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

if __name__ == '__main__':
    main()
