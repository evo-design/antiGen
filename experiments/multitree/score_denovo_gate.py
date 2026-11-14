#!/usr/bin/env python
"""PER-EXAMPLE de-novo gate for the rank-fusion router.

The single-scalar router (score_router_interp.py) slides the whole de-novo sub-list by one
global alpha; a per-VIRUS alpha was shown not to help (oracle 0.908, honest 0.895 < 0.900).
The 0.903->0.939 headroom to the per-event oracle is strictly per-EXAMPLE: it needs a gate that,
for each genotype, predicts whether its next realized substitution will be de-novo (unseen in
training) and routes accordingly -- de-novo-predicted examples get a high alpha (float de-novo
cells up), seen-predicted examples a low/negative alpha (counts first).

This script records, per test example, CAUSAL per-example features (known at the forecast cutoff,
NO tree identity / no global de-novo base rate -- so any gain is genuine per-example signal, not the
failed per-virus knob) plus, for a dense alpha grid, the per-event top-10% hit weight, so emit_
denovo_gate.py can fit the gate leave-one-virus-out and evaluate any per-example alpha by snapping
to the grid. Writes perevent_denovo_gate.npz.

Features per example (all from the input genotype + training counts, never the target events):
  parent_mut_denovo   : the mutation that CREATED this genotype was itself unseen in training (0/1)
  hamming_consensus   : frac positions where genotype differs from the training consensus
  n_private_lowcount  : # of those differing residues whose (pos,res) training count is 0
  ag_top10_dn_frac    : frac of antiGen's own top-10% valid cells that are unseen (N==0)
  ag_max_dn_minus_seen: max antiGen logit over unseen cells - max over seen cells
  ag_mean_dn_minus_seen: mean antiGen logit over unseen - mean over seen
  sc_top_logcount     : log1p(max training count over valid cells) -- dominance of best seen move
  frac_unseen_cells   : frac of valid cells that are unseen
  ag_topsite_saturation: log1p(row-sum training count) at antiGen's top-1 position
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, D, dev, MASK

R  = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
OUT = f"{R}/fc13_core_surface/perevent_denovo_gate.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10
# dense alpha grid spanning pure-seen (-1) to de-novo-floated (+0.2)
ALPHAS = [-1.0, -0.3, -0.1, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03,
          0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
FEATS = ["parent_mut_denovo","hamming_consensus","n_private_lowcount","ag_top10_dn_frac",
         "ag_max_dn_minus_seen","ag_mean_dn_minus_seen","sc_top_logcount",
         "frac_unseen_cells","ag_topsite_saturation"]
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
    p = torch.zeros_like(scoremap)
    v = scoremap[valid]
    sv, _ = torch.sort(v)
    idx = torch.searchsorted(sv, v, right=False).float()
    p[valid] = idx / v.numel()
    return p

def main():
    out = {}
    jm = load_esm(glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0])
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx); st = count_baselines.fit(tr, L, E); N = st["N"]
        rowN = N.sum(dim=1)                                  # per-site total training count
        # training consensus (most common residue per position over training genotypes)
        stack = torch.stack([t.aa_idx for t in tr])          # (n_tr, L)
        consensus = torch.mode(stack, dim=0).values          # (L,)

        p = f"{NA}/examples_{tree}@{D}_test.txt"
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"skip {tree} (no test)", flush=True); continue
        te = load_examples(p, ALPHA_SET, None, source=tree)
        if not te: continue

        ev_w = {akey(a): [] for a in ALPHAS}
        ev_N, ev_ex = [], []
        ex_feat, ex_y = [], []
        exi = 0
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

            # ---- per-event top-10% hit weight at every grid alpha (merged shortlist) ----
            p_sc = pct_map(sc, valid); p_ag = pct_map(js, valid)
            for a in ALPHAS:
                comb = p_sc.clone(); comb[unseen] = p_ag[unseen] + a
                W = topk_weight(comb, valid, budget)
                ev_w[akey(a)] += [float(W[pp, al]) for (pp, al) in ev]
            ev_N  += [float(N[pp, al]) for (pp, al) in ev]
            ev_ex += [exi] * len(ev)

            # ---- causal per-example features ----
            js_v = js.clone(); js_v[~valid] = float('nan')
            ag_un = js[unseen]; ag_se = js[seen]
            # antiGen's own top-10% cells, and the unseen fraction among them
            flat = js_v.flatten(); vv = flat[~torch.isnan(flat)]
            thr = torch.quantile(vv, 1.0 - Q) if vv.numel() else torch.tensor(0.0)
            ag_top = valid & (js >= thr)
            ag_top10_dn_frac = (float((ag_top & unseen).sum()) / max(1, int(ag_top.sum())))
            # parent mutation de-novo?
            pm = 0.0
            if ex.parent_mutation is not None:
                ppos, palt = ex.parent_mutation
                try:
                    pidx = ALPHA_SET.index(palt) if isinstance(palt, str) else int(palt)
                    if 0 <= ppos < L: pm = 1.0 if float(N[ppos, pidx]) < 1 else 0.0
                except Exception: pm = 0.0
            diff = (aa != consensus)
            n_priv = 0
            if diff.any():
                dpos = torch.nonzero(diff, as_tuple=False).flatten()
                n_priv = int(sum(1 for pp in dpos.tolist() if float(N[pp, int(aa[pp])]) < 1))
            topsite = int(torch.nan_to_num(js_v, nan=-1e9).max(dim=1).values.argmax())
            feat = [
                pm,
                float(diff.float().mean()),
                float(n_priv),
                ag_top10_dn_frac,
                float(ag_un.max() - ag_se.max()) if ag_un.numel() and ag_se.numel() else 0.0,
                float(ag_un.mean() - ag_se.mean()) if ag_un.numel() and ag_se.numel() else 0.0,
                float(torch.log1p(N[valid].max())),
                float(unseen.sum()) / max(1, nv),
                float(torch.log1p(rowN[topsite])),
            ]
            ex_feat.append(feat)
            # label: majority of this example's events are de-novo
            yl = np.mean([1.0 if float(N[pp, al]) < 1 else 0.0 for (pp, al) in ev])
            ex_y.append(1.0 if yl > 0.5 else 0.0)
            exi += 1

        for a in ALPHAS:
            out[f"{tree}__ev_w__{akey(a)}"] = np.array(ev_w[akey(a)], dtype=np.float32)
        out[f"{tree}__ev_N"]   = np.array(ev_N, dtype=np.float32)
        out[f"{tree}__ev_ex"]  = np.array(ev_ex, dtype=np.int32)
        out[f"{tree}__ex_feat"] = np.array(ex_feat, dtype=np.float32)
        out[f"{tree}__ex_y"]   = np.array(ex_y, dtype=np.float32)
        print(f"done {tree}: {exi} examples, {len(ev_N)} events, "
              f"de-novo examples={int(np.sum(ex_y))}", flush=True)

    out["__feats__"] = np.array(FEATS)
    out["__alphas__"] = np.array(ALPHAS, dtype=np.float32)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

if __name__ == '__main__':
    main()
