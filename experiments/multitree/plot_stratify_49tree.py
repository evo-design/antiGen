#!/usr/bin/env python
"""antiGen (49 trees) minus Smoothed Counts, recall@10%, stratified by how often the
exact (pos,alt) occurred in the per-tree training corpus. NO-ATTACH, 13 eval trees,
pooled over events. Top: grouped recall bars per training-count bin. Bottom: the
49-tree-minus-SmoothedCounts gap with a paired event-bootstrap 95% CI.
Reads stratify_49tree.npz written by score_stratify_49tree.py.
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R="/large_storage/hielab/aakarsh/multitree"
NPZ=f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/stratify_49tree.npz"
OUTB="/home/aakarsh/projects/antiGen/experiments/multitree/figures/recall_by_training_count_49tree"
TREES=['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']

# bins on the training count of the exact (pos,alt): 0 (de novo), 1, 2-5, 6-20, 21-100, 101+
EDGES=[0,1,2,6,21,101,np.inf]
LABELS=["0\n(de novo)","1","2-5","6-20","21-100","101+"]
MOD_BLUE="#023e8a"; SM_GREY="#7d94a3"; POS="#b23a48"; NEG="#3b6fb0"
rng=np.random.default_rng(0); B=5000

def load(sub):
    nc=[]; ew=[]; sw=[]
    Z=np.load(NPZ)
    for t in TREES:
        k=f"{t}__{sub}__ncount"
        if k not in Z.files: continue
        nc.append(Z[k]); ew.append(Z[f"{t}__{sub}__model"]); sw.append(Z[f"{t}__{sub}__smoothed"])
    return np.concatenate(nc), np.concatenate(ew), np.concatenate(sw)

# ALL events (the 'all' subset carries every count bin incl. 0 = de novo)
nc, ew, sw = load("all")
b=np.digitize(nc, EDGES[1:-1], right=False)   # 0..5

em=np.zeros(6); sm=np.zeros(6); diff=np.zeros(6); lo=np.zeros(6); hi=np.zeros(6); ns=np.zeros(6,int)
for i in range(6):
    m=b==i; ns[i]=m.sum()
    if ns[i]==0: continue
    e=ew[m]; s=sw[m]
    em[i]=e.mean(); sm[i]=s.mean(); diff[i]=em[i]-sm[i]
    d=e-s; idx=rng.integers(0,len(d),size=(B,len(d))); bs=d[idx].mean(1)
    lo[i]=np.percentile(bs,2.5); hi[i]=np.percentile(bs,97.5)

keep=ns>0
em,sm,diff,lo,hi,ns=em[keep],sm[keep],diff[keep],lo[keep],hi[keep],ns[keep]
LAB=[l for l,k in zip(LABELS,keep) if k]
NB=len(ns)

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(13,10.5),height_ratios=[1.45,1],sharex=False)
x=np.arange(NB); w=0.38
ax1.bar(x-w/2, em, w, color=MOD_BLUE, label="antiGen (49 trees)")
ax1.bar(x+w/2, sm, w, color=SM_GREY, label="Smoothed Counts")
for i in range(NB):
    ax1.text(x[i]-w/2, em[i]+0.012, f"{em[i]:.2f}", ha="center", va="bottom", fontsize=11, color=MOD_BLUE)
    ax1.text(x[i]+w/2, sm[i]+0.012, f"{sm[i]:.2f}", ha="center", va="bottom", fontsize=11, color="#4c5a63")
    ax1.text(x[i], -0.055, f"n={ns[i]:,}", ha="center", va="top", fontsize=10.5, color="grey")
ax1.set_ylim(0,1.05); ax1.set_ylabel("recall@10%", fontsize=13)
ax1.set_title("Recall@10% by how often the test substitution occurred in training\n"
              "no-attach test, 13 trees, pooled over events  (antiGen = 49-tree partial-4, epoch 0)",
              fontsize=14, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels([]); ax1.legend(fontsize=12, loc="center right", frameon=False)
ax1.grid(axis="y", alpha=0.25); [ax1.spines[s].set_visible(False) for s in ("top","right")]

cols=[POS if d>=0 else NEG for d in diff]
ax2.bar(x, diff, 0.6, color=cols, yerr=[diff-lo, hi-diff], capsize=4, error_kw=dict(ecolor="grey",lw=1.3))
ax2.axhline(0, color="grey", lw=1)
for i in range(NB):
    va="bottom" if diff[i]>=0 else "top"
    ax2.text(x[i], hi[i]+0.004 if diff[i]>=0 else lo[i]-0.004, f"{diff[i]:+.3f}", ha="center", va=va, fontsize=11)
ax2.set_ylabel("antiGen (49 trees) - Smoothed Counts\n(recall@10%)", fontsize=12)
ax2.set_xlabel("times this exact substitution (pos, alt) occurred in the training corpus", fontsize=13)
ax2.set_xticks(x); ax2.set_xticklabels(LAB, fontsize=12)
ax2.grid(axis="y", alpha=0.25); [ax2.spines[s].set_visible(False) for s in ("top","right")]

plt.tight_layout()
fig.savefig(OUTB+".png", dpi=150, bbox_inches="tight")
fig.savefig(OUTB+".pdf", bbox_inches="tight")
print("wrote",OUTB+".png")
print("bins:",LAB)
print("n   :",list(ns))
print("mod :",[round(float(v),3) for v in em])
print("smo :",[round(float(v),3) for v in sm])
print("diff:",[round(float(v),3) for v in diff])
