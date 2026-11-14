#!/usr/bin/env python
"""Router v2: saturating-count + antiGen-prior blends, scored at recall@10%.

Motivation: cpssm_ag ranks seen cells by RAW N (unbounded), so a handful of very
high-count cells hog top-shortlist slots though recurrence saturates (~0.64 at 30+).
Replace N with a bounded transform h(N) in [0,1], blend with the (per-example
max-normalized) antiGen distribution, and rank by:
    q(i,a) = h(N(i,a)) + lambda * pa(i,a),   pa = p_antigen / max(p_antigen).
Unseen cells get h=0, so they are ranked purely by lambda*pa. lambda is the
dimensionless crossover (max antiGen bonus relative to a fully-saturated seen cell).

Transforms: bin = 1[N>=1]; sat2 = N/(N+2); sat5 = N/(N+5); log30 = log1p(N)/log1p(30) (capped 1).
Writes perevent_router.npz keyed <tree>__<sub>__<transform>_l<NNN>__q10, plus joint/cpssm_dir refs.
"""
import os, sys, glob, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, D, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = f"{R}/fc13_core_surface/perevent_router.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10
TRANSFORMS = {
    "bin":   lambda N: (N >= 1).float(),
    "sat2":  lambda N: N / (N + 2.0),
    "sat5":  lambda N: N / (N + 5.0),
    "log30": lambda N: torch.clamp(torch.log1p(N) / math.log1p(30.0), max=1.0),
}
LAMBDAS = {"l025": 0.25, "l1": 1.0, "l4": 4.0, "l10": 10.0, "l25": 25.0, "l50": 50.0, "l100": 100.0}

def pew(scores, aa_idx, events):
    s = scores.clone().float()
    for c in MASK: s[:, c] = float('nan')
    s[torch.arange(len(aa_idx)), aa_idx] = float('nan')
    flat = s.flatten(); valid = ~torch.isnan(flat)
    nv = int(valid.sum())
    if nv == 0: return [0.0] * len([e for e in events if e[1] not in MASK])
    vb = torch.nanquantile(flat, 1.0 - Q)
    above = (flat > vb) & valid; n_above = int(above.sum())
    tie = (flat == vb) & valid; n_tie = int(tie.sum())
    budget = Q * nv
    frac = max(0.0, (budget - n_above)) / n_tie if n_tie > 0 else 0.0
    w = []
    for (pos, aa) in events:
        if aa in MASK: continue
        v = s[pos, aa]
        if torch.isnan(v): w.append(0.0)
        elif v > vb: w.append(1.0)
        elif v == vb: w.append(frac)
        else: w.append(0.0)
    return w

def antigen_prob(js, aa):
    s = js.clone().float()
    for c in MASK: s[:, c] = float('-inf')
    s[torch.arange(len(aa)), aa] = float('-inf')
    L, Ei = s.shape
    p = torch.softmax(s.flatten(), 0).reshape(L, Ei)
    return torch.nan_to_num(p, nan=0.0)

def main():
    out = {}
    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck)
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        st = count_baselines.fit(tr, L, E)
        N = st["N"]
        hmaps = {tn: fn(N) for tn, fn in TRANSFORMS.items()}      # (L,E) per transform
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            keys = (["joint", "cpssm_dir"] +
                    [f"{tn}_{lk}" for tn in TRANSFORMS for lk in LAMBDAS])
            rows = {k: [] for k in keys}
            cpssm_map_cache = None
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js = jm(ab).float().squeeze(0).cpu()
                pa = antigen_prob(js, aa)
                pm = pa.max()
                pa = pa / pm if pm > 0 else pa
                rows["joint"] += pew(js, aa, ev)
                rows["cpssm_dir"] += pew(
                    count_baselines.score_map('cpssm_dir', st, aa.unsqueeze(0)).squeeze(0), aa, ev)
                for tn in TRANSFORMS:
                    h = hmaps[tn]
                    for lk, lam in LAMBDAS.items():
                        rows[f"{tn}_{lk}"] += pew(h + lam * pa, aa, ev)
            for k, v in rows.items():
                out[f"{tree}__{sub}__{k}__q10"] = np.array(v, dtype=np.float32)
        print(f"done {tree}", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)
    # sanity: joint reproduces budget npz
    Z = np.load(f"{R}/fc13_core_surface/perevent_fig13_sp4e0_budgets.npz")
    worst = 0.0
    for t in TREES:
        for sub in ("all", "novel"):
            k = f"{t}__{sub}__joint__q10"
            if k in out and k in Z.files and len(out[k]) == len(Z[k]) and len(out[k]):
                worst = max(worst, float(np.abs(out[k] - Z[k]).max()))
    print(f"joint q10 reproduction: worst |diff| = {worst:.3g}", flush=True)

if __name__ == '__main__':
    main()
