#!/usr/bin/env python
"""HARD router (not a score blend): partition the candidate cells and let a
DIFFERENT model rank each subset, then union the two shortlists.

Per genotype, among valid cells (non-WT, non-gap/stop):
  seen   S = cells with training count N>=1   -> ranked by Smoothed Counts (cpssm_dir)
  unseen U = cells with N==0                  -> ranked by antiGen(49)
Total shortlist = the SAME 10% budget as every other method, split between pools:
  s_slots = round(phi * budget)  (capped at |S|; spare budget spills to the other pool)
  d_slots = round(budget) - s_slots (capped at |U|)
A test event is a hit if its cell is within its pool's allotted slots (fractional at the
pool boundary, matching per_event_weights). Sweeping phi in [0,1] traces the router's
seen<->de-novo frontier. phi=1 => all budget to seen (=> Smoothed Counts on seen, 0 de novo);
phi=0 => all budget to unseen antiGen (de-novo-max, and seen cells removed from its competition).

Writes perevent_router_hard.npz keyed <tree>__<sub>__phi<NNN>__q10, plus joint/cpssm_dir refs.
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
OUT = f"{R}/fc13_core_surface/perevent_router_hard.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10
PHIS = {"p000": 0.0, "p002": 0.02, "p005": 0.05, "p008": 0.08, "p012": 0.12,
        "p020": 0.20, "p035": 0.35, "p050": 0.50, "p075": 0.75, "p100": 1.0}

def topk_weight(scoremap, pool_mask, kf):
    """(L,E) membership weight for a FLOAT budget kf cells out of pool_mask: 1 above the
    boundary, fractional at it, 0 else (and 0 off-pool). Uses the exact quantile +
    fractional rule as score_perevent.per_event_weights, so with kf = Q*n_valid over the
    whole valid set it reproduces the standard global top-Q% hit weights."""
    W = torch.zeros_like(scoremap)
    if kf <= 0 or not bool(pool_mask.any()):
        return W
    vals = scoremap[pool_mask]
    n = vals.numel()
    if kf >= n:
        W[pool_mask] = 1.0
        return W
    thr = torch.quantile(vals, 1.0 - kf / n)          # matches nanquantile(flat, 1-Q)
    above = (scoremap > thr) & pool_mask
    tie = (scoremap == thr) & pool_mask
    n_above = int(above.sum()); n_tie = int(tie.sum())
    frac = max(0.0, (kf - n_above)) / n_tie if n_tie > 0 else 0.0
    W[above] = 1.0
    W[tie] = frac
    return W

def main():
    out = {}
    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck)
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        st = count_baselines.fit(tr, L, E)
        N = st["N"]
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            keys = ["joint", "cpssm_dir"] + [f"phi{pk}" for pk in PHIS]
            rows = {k: [] for k in keys}
            for ex in te:
                ev = [(pp, aa) for (pp, aa) in events_of(ex) if aa not in MASK]
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js = jm(ab).float().squeeze(0).cpu()
                sc = count_baselines.score_map('cpssm_dir', st, aa.unsqueeze(0)).squeeze(0)
                # valid = non-WT, non-MASK
                valid = torch.ones(L, E, dtype=torch.bool)
                valid[torch.arange(L), aa] = False
                for c in MASK: valid[:, c] = False
                seen = valid & (N >= 1)
                unseen = valid & (N < 1)
                nv = int(valid.sum()); budget = Q * nv
                n_s = float(int(seen.sum())); n_u = float(int(unseen.sum()))
                # reference arms (standard global top-10%, float budget -> reproduces per_event_weights)
                for name, mp in (("joint", js), ("cpssm_dir", sc)):
                    Wr = topk_weight(mp, valid, budget)
                    rows[name] += [float(Wr[pp, a]) for (pp, a) in ev]
                for pk, phi in PHIS.items():
                    s_budget = min(phi * budget, n_s)
                    d_budget = min(budget - s_budget, n_u)
                    extra = budget - s_budget - d_budget
                    if extra > 1e-9:                    # a pool was exhausted; spill to the other
                        if s_budget < n_s: s_budget = min(s_budget + extra, n_s)
                        elif d_budget < n_u: d_budget = min(d_budget + extra, n_u)
                    Ws = topk_weight(sc, seen, s_budget)
                    Wu = topk_weight(js, unseen, d_budget)
                    W = Ws + Wu
                    rows[f"phi{pk}"] += [float(W[pp, a]) for (pp, a) in ev]
            for k, v in rows.items():
                out[f"{tree}__{sub}__{k}__q10"] = np.array(v, dtype=np.float32)
        print(f"done {tree}", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)
    # sanity: router reference cpssm_dir/joint should match the stored budget npz
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
