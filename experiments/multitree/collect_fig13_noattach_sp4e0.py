#!/usr/bin/env python
"""Bootstrap the per-event weights into Ivan's Fig-5 CSV schema, NO-ATTACH panel.

Reads perevent_fig13_noattach.npz (score_perevent_noattach.py) and writes ladder.csv,
deltas.csv, agreement.csv into antiGen-figures/data/multitree_noattach/ for the no-attach
driver of fig5_multitree. The `joint` rung IS the 5-seed epoch-3 logit ensemble (the
antiGen rung the figure shows), so there is no separate single-seed/ensemble split here.
Ladder values are MACRO (per-tree recall averaged, tree-bootstrap CI).
"""
import numpy as np, os, csv
R="/large_storage/hielab/aakarsh/multitree"
NPZ="/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_fig13_sp4e0.npz"
OUT="/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0"; os.makedirs(OUT,exist_ok=True)
TREES=['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
RUNGS=['pssm','codon_blosum','frequency','where_times_which','site_prop','cpssm_dir','single','joint']
EVAL="Temporal holdout"; RULE="pooled"; UNIV="standard"; MET="hit10"; B=2000
rng=np.random.default_rng(0)
Z=np.load(NPZ)
def get(tree,sub,rung):
    k=f"{tree}__{sub}__{rung}"
    return Z[k] if k in Z.files else None

def paired_ci(a,b):
    d=a-b
    if len(d)==0: return float('nan'),float('nan'),float('nan')
    idx=rng.integers(0,len(d),size=(B,len(d)))
    bs=d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

def macro_ci(per):
    per=np.asarray([p for p in per if p==p])
    if len(per)==0: return float('nan'),float('nan'),float('nan')
    idx=rng.integers(0,len(per),size=(B,len(per))); bs=per[idx].mean(1)
    return float(per.mean()), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

# ---- ladder.csv (MACRO recall per rung) ----
with open(f"{OUT}/ladder.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","metric","rule","eval","subset","rung","value","lo","hi"])
    for sub in ("all","novel"):
        w.writerow([UNIV,MET,RULE,EVAL,sub,"uniform",0.10,0.10,0.10])
        for rung in RUNGS:
            per=[get(t,sub,rung).mean() for t in TREES
                 if get(t,sub,rung) is not None and len(get(t,sub,rung))>0]
            m,lo,hi=macro_ci(per)
            w.writerow([UNIV,MET,RULE,EVAL,sub,rung,m,lo,hi])

# ---- deltas.csv (joint - single, joint - where_times_which, joint - cpssm_dir; per tree + POOLED) ----
CONTR={"vs single-tree model":"single","vs where x which":"where_times_which",
       "vs PSSM + Dirichlet":"cpssm_dir"}
with open(f"{OUT}/deltas.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","contrast","eval","rule","metric","tree","subset","est","lo","hi","pooled"])
    for cname,comp in CONTR.items():
        for sub in ("all","novel"):
            per_tree=[]
            for t in TREES:
                j=get(t,sub,"joint"); c=get(t,sub,comp)
                if j is None or c is None or len(j)==0: continue
                est,lo,hi=paired_ci(j,c)           # per-tree delta (event-level CI)
                w.writerow([UNIV,cname,EVAL,RULE,MET,t,sub,est,lo,hi,0])
                per_tree.append(float((j-c).mean()))
            if per_tree:
                # POOLED = MACRO: mean of the per-tree deltas, tree-level bootstrap CI
                # (every tree equal weight, matching the ladder's macro pooling).
                est,lo,hi=macro_ci(per_tree)
                w.writerow([UNIV,cname,EVAL,RULE,MET,"POOLED",sub,est,lo,hi,1])

# ---- agreement.csv (joint vs where_times_which overlap, POOLED, Venn) ----
with open(f"{OUT}/agreement.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","eval","rule","subset","tree","budget","n_mutations","both","model_only","wxw_only","neither"])
    for sub in ("all","novel"):
        J=np.concatenate([get(t,sub,"joint") for t in TREES if get(t,sub,"joint") is not None])
        W=np.concatenate([get(t,sub,"where_times_which") for t in TREES if get(t,sub,"where_times_which") is not None])
        jc=J>=0.5; wc=W>=0.5
        n=len(J); both=int((jc&wc).sum()); mo=int((jc&~wc).sum()); wo=int((~jc&wc).sum()); nei=int((~jc&~wc).sum())
        w.writerow([UNIV,EVAL,RULE,sub,"POOLED",MET,n,both,mo,wo,nei])

# ---- agreement_cpssm.csv (joint vs Smoothed Counts overlap) ----
with open(f"{OUT}/agreement_cpssm.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["universe","eval","rule","subset","tree","budget","n_mutations","both","model_only","wxw_only","neither"])
    for sub in ("all","novel"):
        J=np.concatenate([get(t,sub,"joint") for t in TREES if get(t,sub,"joint") is not None])
        C=np.concatenate([get(t,sub,"cpssm_dir") for t in TREES if get(t,sub,"cpssm_dir") is not None])
        jc=J>=0.5; cc=C>=0.5
        n=len(J); both=int((jc&cc).sum()); mo=int((jc&~cc).sum()); co=int((~jc&cc).sum()); nei=int((~jc&~cc).sum())
        w.writerow([UNIV,EVAL,RULE,sub,"POOLED",MET,n,both,mo,co,nei])
print("wrote ladder/deltas/agreement(.csv/_cpssm.csv) to",OUT)
