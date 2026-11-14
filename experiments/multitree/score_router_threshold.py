#!/usr/bin/env python
"""Reliability router: route by how TRUSTWORTHY the count is, not by seen-vs-unseen.

Per genotype, among valid cells (non-WT, non-gap/stop):
  reliable R = cells with training count N >= tau   -> ranked by Smoothed Counts
  rest     P = cells with N < tau (de-novo AND low-count seen) -> ranked by antiGen(49)
Fill the 10% budget: take all of R (they are few and high-recurrence), then fill the
remainder from the top of P by antiGen. (If |R| exceeds budget, take the top-budget of R
by counts.) The point: low-count seen cells no longer get count-priority, so they stop
crowding the de-novo pool, while the genuinely recurrent high-count cells are caught cheaply.

tau=1 reproduces "all seen are reliable"; tau->inf reduces to antiGen over everything.
Writes perevent_router_thr.npz keyed <tree>__<sub>__tau<NN>__q10, plus joint/cpssm_dir refs.
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
OUT = f"{R}/fc13_core_surface/perevent_router_thr.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10
TAUS = {"t01": 1, "t02": 2, "t03": 3, "t05": 5, "t08": 8, "t15": 15, "t30": 30, "tinf": 10**9}

def topk_weight(scoremap, pool_mask, kf):
    W = torch.zeros_like(scoremap)
    if kf <= 0 or not bool(pool_mask.any()):
        return W
    vals = scoremap[pool_mask]; n = vals.numel()
    if kf >= n:
        W[pool_mask] = 1.0; return W
    thr = torch.quantile(vals, 1.0 - kf / n)
    above = (scoremap > thr) & pool_mask
    tie = (scoremap == thr) & pool_mask
    n_above = int(above.sum()); n_tie = int(tie.sum())
    frac = max(0.0, (kf - n_above)) / n_tie if n_tie > 0 else 0.0
    W[above] = 1.0; W[tie] = frac
    return W

def main():
    out = {}
    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck)
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        st = count_baselines.fit(tr, L, E); N = st["N"]
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            keys = ["joint", "cpssm_dir"] + [f"tau{tk}" for tk in TAUS]
            rows = {k: [] for k in keys}
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
                nv = int(valid.sum()); budget = Q * nv
                for name, mp in (("joint", js), ("cpssm_dir", sc)):
                    Wr = topk_weight(mp, valid, budget)
                    rows[name] += [float(Wr[pp, a]) for (pp, a) in ev]
                for tk, tau in TAUS.items():
                    reliable = valid & (N >= tau)
                    rest = valid & (N < tau)
                    n_rel = float(int(reliable.sum()))
                    if n_rel <= budget:
                        Wr = reliable.float().clone()                 # include all reliable
                        Wp = topk_weight(js, rest, budget - n_rel)    # fill rest by antiGen
                        W = Wr + Wp
                    else:
                        W = topk_weight(sc, reliable, budget)         # reliable overflows budget
                    rows[f"tau{tk}"] += [float(W[pp, a]) for (pp, a) in ev]
            for k, v in rows.items():
                out[f"{tree}__{sub}__{k}__q10"] = np.array(v, dtype=np.float32)
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
    print(f"ref q10 reproduction: worst |diff| = {worst:.3g}", flush=True)

if __name__ == '__main__':
    main()
