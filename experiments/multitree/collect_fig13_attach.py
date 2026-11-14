#!/usr/bin/env python
"""Bootstrap the per-event weights into Ivan's Fig-5 CSV schema (panel 13, temporal, hit10).

Reads perevent_fig13.npz (score_perevent.py) and writes ladder.csv, deltas.csv,
agreement.csv into antiGen-figures/data/multitree13/ for fig5_multitree.R (MULTITREE_PANEL=13).
Single-seed, so CIs are event-level bootstraps (resample observed events with replacement).
"""
import numpy as np, os, csv
R="/large_storage/hielab/aakarsh/multitree"
NPZ=f"{R}/fc13_joint_650m/joint13_partial4_attach_recallsel_s0/perevent_fig13_attach.npz"
OUT="/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_attach"; os.makedirs(OUT,exist_ok=True)
TREES=['h5nx','rsv_a','rsv_b','measles','denv1','denv2','denv3','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
RUNGS=['pssm','substitution','frequency','where_times_which','site_prop','cpssm_dir','single','joint']
EVAL="Temporal holdout"; RULE="pooled"; UNIV="standard"; MET="hit10"; B=2000
rng=np.random.default_rng(0)
Z=np.load(NPZ)
def get(tree,sub,rung):
    k=f"{tree}__{sub}__{rung}"
    return Z[k] if k in Z.files else None

def ci(vals):  # vals: 1d weights; returns mean, lo, hi via event bootstrap
    if len(vals)==0: return float('nan'),float('nan'),float('nan')
    idx=rng.integers(0,len(vals),size=(B,len(vals)))
    bs=vals[idx].mean(1)
    return float(vals.mean()), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

def paired_ci(a,b):  # mean(a-b) with paired event bootstrap
    d=a-b
    if len(d)==0: return float('nan'),float('nan'),float('nan')
    idx=rng.integers(0,len(d),size=(B,len(d)))
    bs=d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

# ---- ladder.csv (MACRO recall per rung: per-tree recall averaged, tree-bootstrap CI) ----
import json as _json
ENS_JSON="/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/partial4_ens_ep1_attachtest.json"
_ens=_json.load(open(ENS_JSON)) if os.path.exists(ENS_JSON) else {}
def macro_ci(per):
    per=np.asarray([p for p in per if p==p])
    if len(per)==0: return float('nan'),float('nan'),float('nan')
    idx=rng.integers(0,len(per),size=(B,len(per))); bs=per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))
with open(f"{OUT}/ladder.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","metric","rule","eval","subset","rung","value","lo","hi"])
    for sub in ("all","novel"):
        w.writerow([UNIV,MET,RULE,EVAL,sub,"uniform",0.10,0.10,0.10])
        for rung in RUNGS:
            per=[get(t,sub,rung).mean() for t in TREES
                 if get(t,sub,rung) is not None and len(get(t,sub,rung))>0]
            m,lo,hi=macro_ci(per)
            w.writerow([UNIV,MET,RULE,EVAL,sub,rung,m,lo,hi])
        # 3-seed logit ensemble rung (per-tree recalls from ensemble_logit.py)
        er=_ens.get(sub,{})
        per=[er[t]['ensemble']['recall10'] for t in TREES
             if t in er and 'ensemble' in er[t] and er[t]['ensemble']['n']>0]
        m,lo,hi=macro_ci(per)
        w.writerow([UNIV,MET,RULE,EVAL,sub,"ensemble",m,lo,hi])

# ---- deltas.csv (joint - single, joint - where_times_which; per tree + POOLED) ----
CONTR={"vs single-tree model":"single","vs where x which":"where_times_which",
       "vs PSSM + Dirichlet":"cpssm_dir"}
with open(f"{OUT}/deltas.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","contrast","eval","rule","metric","tree","subset","est","lo","hi","pooled"])
    for cname,comp in CONTR.items():
        for sub in ("all","novel"):
            pj=[];pc=[]
            for t in TREES:
                j=get(t,sub,"joint"); c=get(t,sub,comp)
                if j is None or c is None or len(j)==0: continue
                est,lo,hi=paired_ci(j,c)
                w.writerow([UNIV,cname,EVAL,RULE,MET,t,sub,est,lo,hi,0])
                pj.append(j); pc.append(c)
            if pj:
                J=np.concatenate(pj); C=np.concatenate(pc); est,lo,hi=paired_ci(J,C)
                w.writerow([UNIV,cname,EVAL,RULE,MET,"POOLED",sub,est,lo,hi,1])

# ---- agreement.csv (joint vs where_times_which overlap, POOLED, Venn) ----
with open(f"{OUT}/agreement.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","eval","rule","subset","tree","budget","n_mutations","both","model_only","wxw_only","neither"])
    for sub in ("all","novel"):
        J=np.concatenate([get(t,sub,"joint") for t in TREES if get(t,sub,"joint") is not None])
        W=np.concatenate([get(t,sub,"where_times_which") for t in TREES if get(t,sub,"where_times_which") is not None])
        jc=J>=0.5; wc=W>=0.5   # "caught" = event inside top-10% (majority for tie cells)
        n=len(J); both=int((jc&wc).sum()); mo=int((jc&~wc).sum()); wo=int((~jc&wc).sum()); nei=int((~jc&~wc).sum())
        w.writerow([UNIV,EVAL,RULE,sub,"POOLED",MET,n,both,mo,wo,nei])

# ---- agreement_cpssm.csv (joint vs PSSM+Dirichlet overlap, for the D/E/F variant) ----
with open(f"{OUT}/agreement_cpssm.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","eval","rule","subset","tree","budget","n_mutations","both","model_only","wxw_only","neither"])
    for sub in ("all","novel"):
        J=np.concatenate([get(t,sub,"joint") for t in TREES if get(t,sub,"joint") is not None])
        C=np.concatenate([get(t,sub,"cpssm_dir") for t in TREES if get(t,sub,"cpssm_dir") is not None])
        jc=J>=0.5; cc=C>=0.5
        n=len(J); both=int((jc&cc).sum()); mo=int((jc&~cc).sum()); co=int((~jc&cc).sum()); nei=int((~jc&~cc).sum())
        w.writerow([UNIV,EVAL,RULE,sub,"POOLED",MET,n,both,mo,co,nei])   # wwx_only col = cpssm_only here
print("wrote ladder/deltas/agreement(.csv/_cpssm.csv) to",OUT)
