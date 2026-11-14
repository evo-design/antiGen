#!/usr/bin/env python
"""RANK-FUSION router: one merged shortlist, seen cells ranked by Smoothed Counts and
de-novo cells ranked by antiGen(49), interleaved by a single scalar alpha.

Per genotype, over valid cells (non-WT, non-gap/stop):
  p_SC(i,a) = percentile of cell (i,a) under Smoothed Counts, among valid cells  [0,1)
  p_AG(i,a) = percentile under antiGen(49), among valid cells                    [0,1)
  combined(i,a) = p_SC                 if seen   (N>=1)
                = p_AG + alpha          if de-novo (N==0)
Take the global top-10% of combined (same budget as every method). alpha slides the whole
de-novo sub-list up/down the single list: alpha<<0 => seen-first; alpha>>0 => de-novo-first.
Sweeping alpha traces the frontier; a single alpha is LEARNED post-hoc by leave-one-virus-out
(emit_router_interp.py). alpha=... chosen on a dense grid.

Stores per-event hit weights for every alpha + the event's training count N (to stratify
seen/de-novo), plus joint/cpssm_dir reference arms. Writes perevent_router_interp.npz.
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, D, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = f"{R}/fc13_core_surface/perevent_router_interp.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10
ALPHAS = [-1.0, -0.4, -0.25, -0.18, -0.12, -0.08, -0.05, -0.03, -0.015, 0.0,
          0.015, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 1.0]
def akey(a): return f"a{a:+.3f}".replace('.', 'p').replace('+', 'P').replace('-', 'M')

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

def pct_map(scoremap, valid):
    """(L,E) percentile (fraction of valid cells strictly less) on valid cells, else 0."""
    p = torch.zeros_like(scoremap)
    v = scoremap[valid]
    sv, _ = torch.sort(v)
    idx = torch.searchsorted(sv, v, right=False).float()   # # strictly less
    p[valid] = idx / v.numel()
    return p

def main():
    out = {}
    jm = load_esm(glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0])
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx); st = count_baselines.fit(tr, L, E); N = st["N"]
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            keys = ["joint", "cpssm_dir"] + [akey(a) for a in ALPHAS]
            rows = {k: [] for k in keys}; meta_n = []
            for ex in te:
                ev = [(pp, aa) for (pp, aa) in events_of(ex) if aa not in MASK]
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js = jm(ab).float().squeeze(0).cpu()
                sc = count_baselines.score_map('cpssm_dir', st, aa.unsqueeze(0)).squeeze(0)
                valid = torch.ones(L, E, dtype=torch.bool)
                valid[torch.arange(L), aa] = False
                for c in MASK: valid[:, c] = False
                seen = valid & (N >= 1); unseen = valid & (N < 1)
                nv = int(valid.sum()); budget = Q * nv
                for name, mp in (("joint", js), ("cpssm_dir", sc)):
                    Wr = topk_weight(mp, valid, budget)
                    rows[name] += [float(Wr[pp, a]) for (pp, a) in ev]
                p_sc = pct_map(sc, valid); p_ag = pct_map(js, valid)
                for a in ALPHAS:
                    comb = p_sc.clone()
                    comb[unseen] = p_ag[unseen] + a      # seen keep p_sc; de-novo get p_ag+alpha
                    W = topk_weight(comb, valid, budget)
                    rows[akey(a)] += [float(W[pp, aa_]) for (pp, aa_) in ev]
                meta_n += [float(N[pp, aa_]) for (pp, aa_) in ev]
            for k, v in rows.items():
                out[f"{tree}__{sub}__{k}__q10"] = np.array(v, dtype=np.float32)
            out[f"{tree}__{sub}__N"] = np.array(meta_n, dtype=np.float32)
        print(f"done {tree}", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)
    Z = np.load(f"{R}/fc13_core_surface/perevent_fig13_sp4e0_budgets.npz")
    worst = 0.0
    for a in ("joint", "cpssm_dir"):
        for t in TREES:
            for sub in ("all", "novel"):
                k = f"{t}__{sub}__{a}__q10"
                if k in out and k in Z.files and len(out[k]) == len(Z[k]) and len(out[k]):
                    worst = max(worst, float(np.abs(out[k] - Z[k]).max()))
    print(f"ref q10 reproduction (joint,cpssm_dir): worst |diff| = {worst:.3g}", flush=True)

if __name__ == '__main__':
    main()
