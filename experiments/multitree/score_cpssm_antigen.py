#!/usr/bin/env python
"""New baseline: Smoothed Counts with antiGen(49) as the Dirichlet prior.

Standard Smoothed Counts (cpssm_dir) is the posterior mean of a Dirichlet-multinomial
whose base measure is the factorized Site x Substitution prior p0:
    q(i,a) = (N(i,a) + kappa * p0(i,a)) / (N_total + kappa),   kappa = N_total.
Here we replace p0 with antiGen(49)'s own whole-sequence softmax distribution
p_ag (WT + gap/stop masked), i.e. the exact finetuned model's output used as the prior:
    q(i,a) = (N(i,a) + kappa * p_ag(i,a)) / (N_total + kappa).
Ranking is scale-invariant, so we rank by  N + kappa * p_ag  directly.

Scored per-event (fractional hit) at q in {10,5,1}% and for kappa in {0.25,1,4}*N_total,
alongside plain antiGen(49) and plain Smoothed Counts for a self-contained comparison.
Writes perevent_cpssm_antigen.npz keyed <tree>__<sub>__<arm>__q{NN}.
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
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
OUT = f"{R}/fc13_core_surface/perevent_cpssm_antigen.npz"
BUDGETS = {"10": 0.10, "05": 0.05, "01": 0.01}
KMULT = {"k1": 1.0, "k4": 4.0, "k5": 5.0, "k8": 8.0, "k10": 10.0,
         "k16": 16.0, "k32": 32.0, "k100": 100.0}

def pew(scores, aa_idx, events, Q):
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
    """antiGen whole-sequence softmax over valid cells (WT + MASK -> 0)."""
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
        N = st["N"]; Ntot = float(N.sum())
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            arms = ['joint', 'cpssm_dir'] + [f"cpssm_ag_{k}" for k in KMULT]
            rows = {(a, q): [] for a in arms for q in BUDGETS}
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js = jm(ab).float().squeeze(0).cpu()
                p_ag = antigen_prob(js, aa)                                  # (L,E) prior
                cpssm = count_baselines.score_map('cpssm_dir', st, aa.unsqueeze(0)).squeeze(0)
                maps = {'joint': js, 'cpssm_dir': cpssm}
                for kname, km in KMULT.items():
                    maps[f"cpssm_ag_{kname}"] = N + (km * Ntot) * p_ag        # rank by N + kappa*p_ag
                for a, mp in maps.items():
                    for qk, qv in BUDGETS.items():
                        rows[(a, qk)] += pew(mp, aa, ev, qv)
            for (a, qk), v in rows.items():
                out[f"{tree}__{sub}__{a}__q{qk}"] = np.array(v, dtype=np.float32)
        print(f"done {tree}  N_total={Ntot:.0f}", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

    # sanity: joint & cpssm_dir q10 must reproduce the stored budget npz
    Z = np.load(f"{R}/fc13_core_surface/perevent_fig13_sp4e0_budgets.npz")
    worst = 0.0
    for a in ("joint", "cpssm_dir"):
        for t in TREES:
            for sub in ("all", "novel"):
                k1 = f"{t}__{sub}__{a}__q10"
                if k1 in out and k1 in Z.files and len(out[k1]) == len(Z[k1]) and len(out[k1]):
                    worst = max(worst, float(np.abs(out[k1] - Z[k1]).max()))
    print(f"q10 reproduction (joint, cpssm_dir) vs budget npz: worst |diff| = {worst:.3g}", flush=True)

if __name__ == '__main__':
    main()
