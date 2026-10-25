#!/usr/bin/env python
"""Per-event fractional hit weights at q=10% for every rung, for the Fig-5 bootstrap.

For each tree and subset (all/novel), each observed post-D mutation event gets a weight in
[0,1]: 1 if its (pos,alt) cell is strictly inside the model's top-10%, the tie fraction if
on the boundary, else 0 -- exactly compute_hit_rate's fractional tie-break, but per event.
Gap/stop columns masked, sentinel target events dropped.

Rungs: joint (ESM-2 650M ep6), single (ESM-2 650M per tree), site_prop, cpssm_dir,
pssm (residue occurrence), frequency (event PSSM), where_times_which, substitution,
where_only. Writes one npz of ragged weight arrays, consumed by collect_fig13.py.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for count_baselines
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
import torch
from antiGen.load import load_examples
from antiGen.eval_utils import (load_checkpoint, fit_substitution_baseline,
    positional_marginal_counts, frequency_baseline_scores, where_only_baseline_scores,
    WhereTimesWhichPredictor)
from antiGen.models.factory import create_model

R="/large_storage/hielab/aakarsh/multitree"; D="2025-01-01"
FDATA="/home/ivan.specht/antiGen/experiments/multitree/data/forecast"
ALPHA_SET="ACDEFGHIKLMNPQRSTVWY*-"; E=len(ALPHA_SET); Q=0.10; ALPHA=0.5
GAP,STOP=ALPHA_SET.index('-'),ALPHA_SET.index('*'); MASK=[GAP,STOP]
TREES=['ncov','h5nx','measles','rsv_a','rsv_b','denv1','denv2','denv3','denv4','h3n2_ha','h1n1pdm_ha','hmpv_b_g','vic_ha']
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def per_event_weights(scores, aa_idx, events):
    """scores (L,E) float; aa_idx (L,); events list of (pos,aa). Returns weights list."""
    s=scores.clone().float()
    for c in MASK: s[:,c]=float('nan')
    s[torch.arange(len(aa_idx)), aa_idx]=float('nan')     # mask WT
    flat=s.flatten(); valid=~torch.isnan(flat)
    nv=int(valid.sum())
    if nv==0: return [0.0]*len(events)
    vb=torch.nanquantile(flat, 1.0-Q)                     # boundary value
    above=(flat>vb)&valid; n_above=int(above.sum())
    tie=(flat==vb)&valid; n_tie=int(tie.sum())
    budget=Q*nv
    frac=max(0.0,(budget-n_above))/n_tie if n_tie>0 else 0.0
    w=[]
    for (pos,aa) in events:
        if aa in MASK: continue
        v=s[pos,aa]
        if torch.isnan(v): w.append(0.0)
        elif v>vb: w.append(1.0)
        elif v==vb: w.append(frac)
        else: w.append(0.0)
    return w

import count_baselines   # single source of truth for the pssm / site_prop / cpssm maps

def events_of(ex):
    # observed child mutations (pos,aa) from the (L,E) count matrix
    nz=torch.nonzero(ex.mat>0, as_tuple=False)
    out=[]
    for pos,aa in nz.tolist():
        out+= [(pos,aa)]*int(round(float(ex.mat[pos,aa])))
    return out

def load_esm(ckpt):
    cap,cfg=load_checkpoint(ckpt, map_location='cpu')
    m=create_model(cfg, torch.device('cpu'))
    m.load_state_dict(cap['model_state_dict'], strict=False)
    del cap
    return m.to(dev).eval()

def main():
    out={}
    # joint ep6 snapshot is the SAME checkpoint for every tree -> load once
    j_ck=glob.glob(f"{R}/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0/*_best_denv1.pt")[0]
    jm=load_esm(j_ck)
    for tree in TREES:
        tr=load_examples(f"{FDATA}/examples_{tree}@{D}_train.txt",ALPHA_SET,None,source=tree)
        L=len(tr[0].aa_idx)
        st=count_baselines.fit(tr,L,E)
        # fitted substitution (for where_times_which + substitution rung)
        fitted=fit_substitution_baseline(tr,dev,ALPHA_SET)
        sub_mat=torch.log_softmax(fitted.score_matrix().detach().cpu(),dim=-1)  # (E,E) log P(a|x)
        pos_counts=positional_marginal_counts(tr,L,E)
        freq=frequency_baseline_scores(tr,L,E); wonly=where_only_baseline_scores(tr,L,E)
        # single ESM for this tree (load, use, free)
        s_ck=[p for p in glob.glob(f"{R}/forecast_single_650m/single_{tree}_s0/*_best.pt") if "_best_" not in p][-1]
        sm=load_esm(s_ck)
        for sub,suf in [('all','test'),('novel','novel')]:
            p=f"{FDATA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p)==0: continue
            te=load_examples(p,ALPHA_SET,None,source=tree)
            if not te: continue
            rows={r:[] for r in ['joint','single','site_prop','cpssm_dir','pssm','frequency','where_times_which','substitution','where_only']}
            for ex in te:
                ev=events_of(ex)
                if not ev: continue
                aa=ex.aa_idx; ab=aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js=jm(ab).float().squeeze(0).cpu(); ss=sm(ab).float().squeeze(0).cpu()
                ab1=aa.unsqueeze(0)
                maps={
                  'joint':js,'single':ss,
                  'site_prop':count_baselines.score_map('site_prop',st,ab1).squeeze(0),
                  'cpssm_dir':count_baselines.score_map('cpssm_dir',st,ab1).squeeze(0),
                  'pssm':count_baselines.score_map('pssm',st,ab1).squeeze(0),
                  'frequency':torch.log(freq),'where_only':torch.log(wonly),
                  'where_times_which':sub_mat[aa]+torch.log(pos_counts.clamp(min=1e-12)).view(L,1),
                  'substitution':sub_mat[aa],
                }
                for r,mp in maps.items(): rows[r]+=per_event_weights(mp,aa,ev)
            out[f"{tree}|{sub}"]={r:np.array(v,dtype=np.float32) for r,v in rows.items()}
        del sm
        if dev.type=='cuda': torch.cuda.empty_cache()
        print(f"done {tree}", flush=True)
    np.savez_compressed(f"{R}/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0/perevent_fig13.npz",
                        **{k.replace("|","__")+"__"+r:v for k,d in out.items() for r,v in d.items()})
    print("wrote perevent_fig13.npz")

if __name__=='__main__': main()
