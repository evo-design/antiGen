#!/usr/bin/env python
"""DMS-escape re-ranking of antiGen(49)'s de-novo predictions on SC2 (ncov) ONLY.

Replaces the positional-mutability proxy gate with a REAL escape signal: the Bloom
escape-calculator polyclonal map (dms_escape_assayed_cells.csv), max escape over ~300
antibodies per (site, mutant residue), JN.1 background (closest to the 2025-01-01 cutoff).
The most favorable form of the hypothesis -- inject measured escape directly:
  corr(i,a) = antiGen_logit(i,a) + beta * escape_jn1(i,a)      (escape in [0,1])
A cell with no measurement gets +0 (unchanged). Map covers RBD sites 331-531 only.

Scored at recall@10% on ncov, two ways, with per-event metadata (N count, RBD-covered)
so emit can stratify all / RBD-covered / RBD-radical:
  A) standalone corrected antiGen over all cells   -> corr_b<beta>
  B) router de-novo pool (N<1), phi=0, corrected    -> rdn_b<beta>
beta=0 reproduces plain antiGen. Writes perevent_escape_sc2.npz.
"""
import os, sys, glob
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, per_event_weights, E, ALPHA_SET, D, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
ESC = "/home/aakarsh/projects/antiGen/examples/analysis/data/dms_escape_assayed_cells.csv"
OUT = f"{R}/fc13_core_surface/perevent_escape_sc2.npz"
TREE = "ncov"
Q = 0.10
BETAS = {"b00": 0.0, "b1": 1.0, "b2": 2.0, "b5": 5.0, "b10": 10.0, "b20": 20.0}
ESC_COL = "max_esc_jn1"

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

def build_escape(L):
    """(L,E) escape grid: grid[site-1][alt] = escape; 0 where unmeasured. Also a
    boolean (L,E) 'covered' mask of measured cells."""
    esc = pd.read_csv(ESC)
    grid = torch.zeros(L, E); cov = torch.zeros(L, E, dtype=torch.bool)
    placed = 0
    for _, r in esc.iterrows():
        i = int(r["site"]) - 1                       # 1-indexed spike -> 0-indexed
        aa = str(r["residue"])
        if aa not in ALPHA_SET: continue
        j = ALPHA_SET.index(aa)
        if not (0 <= i < L): continue
        v = float(r[ESC_COL])
        if np.isnan(v): continue
        grid[i, j] = v; cov[i, j] = True; placed += 1
    print(f"escape grid: placed {placed} measured cells "
          f"(rows {int(cov.any(1).nonzero().min())}-{int(cov.any(1).nonzero().max())})")
    return grid, cov

def main():
    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck)
    tr = load_examples(f"{NA}/examples_{TREE}@{D}_train.txt", ALPHA_SET, None, source=TREE)
    L = len(tr[0].aa_idx); N = count_baselines.fit(tr, L, E)["N"]
    Emap, Cov = build_escape(L)
    te = load_examples(f"{NA}/examples_{TREE}@{D}_test.txt", ALPHA_SET, None, source=TREE)
    out = {}
    keys = [f"corr_{bk}" for bk in BETAS] + [f"rdn_{bk}" for bk in BETAS]
    rows = {k: [] for k in keys}
    meta_n, meta_cov, meta_rbd, meta_blo = [], [], [], []   # per-event, aligned to ev order
    for ex in te:
        ev = [(pp, aa) for (pp, aa) in events_of(ex) if aa not in MASK]
        if not ev: continue
        aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
        with torch.no_grad():
            js = jm(ab).float().squeeze(0).cpu()
        valid = torch.ones(L, E, dtype=torch.bool)
        valid[torch.arange(L), aa] = False
        for c in MASK: valid[:, c] = False
        unseen = valid & (N < 1)
        nv = int(valid.sum()); budget = Q * nv
        for bk, beta in BETAS.items():
            corr = js + beta * Emap
            rows[f"corr_{bk}"] += per_event_weights(corr, aa, ev)
            Wu = topk_weight(corr, unseen, budget)
            rows[f"rdn_{bk}"] += [float(Wu[pp, a]) for (pp, a) in ev]
        for (pp, a) in ev:
            meta_n.append(float(N[pp, a]))
            meta_cov.append(bool(Cov[pp, a]))
            meta_rbd.append(330 <= pp <= 530)
    for k, v in rows.items():
        out[f"{TREE}__{k}"] = np.array(v, dtype=np.float32)
    out[f"{TREE}__meta_n"] = np.array(meta_n, dtype=np.float32)
    out[f"{TREE}__meta_cov"] = np.array(meta_cov, dtype=bool)
    out[f"{TREE}__meta_rbd"] = np.array(meta_rbd, dtype=bool)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, "  events:", len(meta_n))

if __name__ == '__main__':
    main()
