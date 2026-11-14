#!/usr/bin/env python
"""Build macro (12-virus, denv4 dropped) panel data for fig4 H/I/J/K/L, standardized
650m/p4 (MeanLogProb+NLL) model arms + from-scratch transformer + baselines.

Outputs (in the figure data dir):
  ladder_fig4.csv   rung, subset{all,novel}, est, lo, hi            (H/I)
  forest_fig4.csv   contrast{single,smoothed}, tree, subset, est, lo, hi, summary  (J/K)
  scatter_fig4.csv  tree, est, train_events                          (L, joint49-single, all)

Macro = equal weight per virus; CI propagated from per-virus CIs (macro analog of the
figure's pool_strike): est=mean(r_v), se=sqrt(sum se_v^2)/n, se_v=(hi-lo)/(2*1.96).
Panel-K comparator (Smoothed Counts) paired deltas use per-event weights: joint49 from
HI_650mp4.npz, cpssm_dir recomputed here in the SAME event order.
"""
import os, sys, glob, csv, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
import count_baselines
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D

REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
DD = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"
DROP = {"denv1", "denv3", "denv4"}   # noisy dengue trees (<=5 de-novo events each)
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
Q = 0.10; NBOOT = 2000

def se_of(lo, hi): return (hi - lo) / (2 * 1.96)
def macro_est_ci(rv):   # rv: list of (recall, lo, hi)
    r = [x[0] for x in rv]; se = [se_of(x[1], x[2]) for x in rv]
    n = len(r); est = sum(r) / n; s = (sum(e*e for e in se) ** 0.5) / n
    return est, max(0.0, est - 1.96*s), min(1.0, est + 1.96*s)

# ---------- per-virus recall+CI sources ----------
def rows_of(path):
    return list(csv.DictReader(open(f"{DD}/{path}")))
grid = rows_of("grid_recall_650m_p4.csv")
xf   = rows_of("xformer_val_pervirus.csv")
b9   = rows_of("denovo_panels_recall.csv")

def arm_pv(arm, sub, pool):   # standardized 650m/p4 model arms (macro selection)
    out = {}
    for r in grid:
        if (r['arm']==arm and r['selection']=='macro' and r['subset']==sub and r['pool']==pool
                and abs(float(r['q'])-0.10)<1e-9 and r['recall'] not in ('nan','')):
            out[r['virus']] = (float(r['recall']), float(r['ci_lo']), float(r['ci_hi']))
    return out
def xf_pv(sub, pool):
    return {r['virus']:(float(r['recall']),float(r['ci_lo']),float(r['ci_hi']))
            for r in xf if r['subset']==sub and r['pool']==pool}
def base_pv(arm, sub, pool):
    return {r['virus']:(float(r['recall']),float(r['ci_lo']),float(r['ci_hi']))
            for r in b9 if r['arm']==arm and r['subset']==sub and r['pool']==pool
            and abs(float(r['q'])-0.10)<1e-9}

# rung key -> getter
RUNGS = [
    ("joint",     lambda s,p: arm_pv("joint49",s,p)),
    ("joint13",   lambda s,p: arm_pv("joint13",s,p)),
    ("single",    lambda s,p: arm_pv("single",s,p)),
    ("xformer",   lambda s,p: xf_pv(s,p)),
    ("cpssm_dir", lambda s,p: base_pv("cpssm_dir",s,p)),
    ("site_prop", lambda s,p: base_pv("site_prop",s,p)),
    ("where_times_which", lambda s,p: base_pv("where_times_which",s,p)),
    ("codon_blosum", lambda s,p: base_pv("codon_blosum",s,p)),
    ("pssm",      lambda s,p: base_pv("pssm",s,p)),
    ("frequency", lambda s,p: base_pv("frequency",s,p)),
    ("zeroshot",  lambda s,p: base_pv("zeroshot",s,p)),
]

def write_ladder():
    with open(f"{DD}/ladder_fig4.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["rung","subset","est","lo","hi","n_virus"])
        for sub,pool in [("all","full"),("novel","denovo")]:
            for key,getter in RUNGS:
                pv=getter(sub,pool)
                rv=[v for t,v in pv.items() if t not in DROP]
                if not rv: continue
                est,lo,hi=macro_est_ci(rv)
                w.writerow([key,sub,f"{est:.6f}",f"{lo:.6f}",f"{hi:.6f}",len(rv)])
            w.writerow(["uniform",sub,"0.100000","0.100000","0.100000",0])
    print("wrote ladder_fig4.csv")

# ---------- forests: per-virus paired deltas + CI + macro summary ----------
def paired_ci(dvec):
    a=np.asarray(dvec,float)
    if a.size==0: return (float('nan'),float('nan'))
    rng=np.random.default_rng(0)
    m=a[rng.integers(0,a.size,(NBOOT,a.size))].mean(1)
    return float(np.percentile(m,2.5)),float(np.percentile(m,97.5))

def forest_J():
    """joint49 - single, per-virus + CI from HI_650mp4_forest.csv; macro summary."""
    fr=list(csv.DictReader(open(f"{DD}/HI_650mp4_forest.csv")))
    rows={}
    for r in fr:
        if r['contrast']!='joint49-single': continue
        rows.setdefault((r['virus'],r['subset']),r)
    out=[]
    summ={}
    for (v,sub),r in rows.items():
        if v in DROP: continue
        d=float(r['delta']); lo=float(r['d_lo']); hi=float(r['d_hi'])
        out.append(("single",v,sub,d,lo,hi,0))
        summ.setdefault(sub,[]).append((d,lo,hi))
    for sub,rv in summ.items():
        out.append(("single","POOLED",sub,*macro_est_ci_delta(rv),1))
    return out

def macro_est_ci_delta(rv):
    d=[x[0] for x in rv]; se=[se_of(x[1],x[2]) for x in rv]; n=len(d)
    est=sum(d)/n; s=(sum(e*e for e in se)**0.5)/n
    return est, est-1.96*s, est+1.96*s

def forest_K():
    """joint49 - Smoothed Counts, per-virus + paired CI (per-event); macro summary."""
    W=np.load(f"{DD}/HI_650mp4.npz")
    out=[]; summ={}
    for t in TREES:
        if t in DROP: continue
        tr=load_examples(f"{NA}/examples_{t}@{D}_train.txt",ALPHA_SET,None,source=t)
        L=len(tr[0].aa_idx); seen=torch.zeros(L,E)
        for ex in tr: seen+=ex.mat
        seen=seen>0
        st=count_baselines.fit(tr,L,E)
        for sub,suf,pool in [("all","test","full"),("novel","novel","denovo")]:
            p=f"{NA}/examples_{t}@{D}_{suf}.txt"
            if not (os.path.exists(p) and os.path.getsize(p)>0): continue
            te=load_examples(p,ALPHA_SET,None,source=t)
            ent=[(ex.aa_idx,events_of(ex)) for ex in te if events_of(ex)]
            if not ent: continue
            jkey=f"{t}__{sub}__{pool}__joint49"
            if jkey not in W: continue
            jvec=W[jkey]
            smask=None if pool=="full" else seen
            cv=[]
            for aa,ev in ent:
                mp=count_baselines.score_map("cpssm_dir",st,aa.unsqueeze(0)).float().squeeze(0)
                cv+=per_event_weights(mp,aa,ev,seen_mask=smask,q=Q)
            cvec=np.asarray(cv,np.float32)
            n=min(len(jvec),len(cvec)); dvec=jvec[:n]-cvec[:n]
            lo,hi=paired_ci(dvec)
            out.append(("smoothed",t,sub,float(dvec.mean()),lo,hi,0))
            summ.setdefault(sub,[]).append((float(dvec.mean()),lo,hi))
        print(f"  K done {t}",flush=True)
    for sub,rv in summ.items():
        out.append(("smoothed","POOLED",sub,*macro_est_ci_delta(rv),1))
    return out

def write_forest():
    rows=forest_J()+forest_K()
    with open(f"{DD}/forest_fig4.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["contrast","tree","subset","est","lo","hi","summary"])
        for r in rows: w.writerow([r[0],r[1],r[2],f"{r[3]:.6f}",f"{r[4]:.6f}",f"{r[5]:.6f}",r[6]])
    print("wrote forest_fig4.csv")

def write_scatter():
    fr=list(csv.DictReader(open(f"{DD}/HI_650mp4_forest.csv")))
    ds={r['tree']:int(r['train_events']) for r in csv.DictReader(open(f"{DD}/datasize.csv"))}
    with open(f"{DD}/scatter_fig4.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["tree","est","train_events"])
        for r in fr:
            if r['contrast']=='joint49-single' and r['subset']=='all' and r['virus'] not in DROP:
                w.writerow([r['virus'],f"{float(r['delta']):.6f}",ds.get(r['virus'],'')])
    print("wrote scatter_fig4.csv")

if __name__=='__main__':
    write_ladder(); write_scatter(); write_forest()
