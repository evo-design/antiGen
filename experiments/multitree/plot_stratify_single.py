#!/usr/bin/env python
"""Recall@10 by training recurrence for a SINGLE virus: ensembled multi-tree antiGen
(5-seed epoch-3) vs Smoothed Counts. Top = grouped bars per recurrence bin; bottom =
ensemble-minus-SmoothedCounts gap with a paired event bootstrap CI. Reads
stratify_noattach.npz. Usage: plot_stratify_single.py <tree> [<tree> ...]
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R="/large_storage/hielab/aakarsh/multitree"
NPZ=f"{R}/fc13_joint_650m/joint13_lmonly_noattach_recallsel_s0/stratify_noattach.npz"
OUTDIR="/home/aakarsh/projects/antiGen/experiments/multitree/figures"
NICE={'h5nx':'H5Nx HA','hmpv_a_g':'hMPV A G','ncov':'SARS-CoV-2 S','h3n2_ha':'H3N2 HA',
      'h1n1pdm_ha':'H1N1pdm HA','vic_ha':'B/Victoria HA','measles':'Measles H',
      'rsv_a':'RSV A F','rsv_b':'RSV B F','denv1':'DENV1 E','denv2':'DENV2 E',
      'denv3':'DENV3 E','denv4':'DENV4 E'}
EDGES=[0,1,2,6,21,101,np.inf]
LABELS=["0\n(de novo)","1","2-5","6-20","21-100","101+"]
ENS_BLUE="#15497b"; SM_GREY="#7d94a3"; POS="#b23a48"; NEG="#3b6fb0"
rng=np.random.default_rng(0); B=5000
Z=np.load(NPZ)

def plot_tree(tree):
    nc=Z[f"{tree}__all__ncount"]; ew=Z[f"{tree}__all__ensemble"]; sw=Z[f"{tree}__all__smoothed"]
    b=np.digitize(nc, EDGES[1:-1], right=False)
    em=np.zeros(6); sm=np.zeros(6); diff=np.zeros(6); lo=np.zeros(6); hi=np.zeros(6); ns=np.zeros(6,int)
    for i in range(6):
        m=b==i; ns[i]=m.sum()
        if ns[i]==0: continue
        e=ew[m]; s=sw[m]; em[i]=e.mean(); sm[i]=s.mean(); diff[i]=em[i]-sm[i]
        d=e-s; idx=rng.integers(0,len(d),size=(B,len(d))); bs=d[idx].mean(1)
        lo[i]=np.percentile(bs,2.5); hi[i]=np.percentile(bs,97.5)
    keep=ns>0
    em,sm,diff,lo,hi,ns=em[keep],sm[keep],diff[keep],lo[keep],hi[keep],ns[keep]
    labs=[l for l,k in zip(LABELS,keep) if k]; NB=len(ns)

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,9),height_ratios=[1.45,1])
    x=np.arange(NB); w=0.38
    ax1.bar(x-w/2, em, w, color=ENS_BLUE, label="antiGen (multi-tree)")
    ax1.bar(x+w/2, sm, w, color=SM_GREY, label="Smoothed Counts")
    for i in range(NB):
        ax1.text(x[i]-w/2, em[i]+0.012, f"{em[i]:.2f}", ha="center", va="bottom", fontsize=11, color=ENS_BLUE)
        ax1.text(x[i]+w/2, sm[i]+0.012, f"{sm[i]:.2f}", ha="center", va="bottom", fontsize=11, color="#4c5a63")
        ax1.text(x[i], -0.055, f"n={ns[i]:,}", ha="center", va="top", fontsize=10.5, color="grey")
    ax1.set_ylim(0,1.08); ax1.set_ylabel("recall@10%", fontsize=13)
    ax1.set_title(f"Recall@10% by how often the test substitution occurred in training — {NICE[tree]}\n"
                  "no-attach, pooled over events  (antiGen = 5-seed epoch-3 ensemble)",
                  fontsize=13, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels([]); ax1.legend(fontsize=12, loc="lower right", frameon=False)
    ax1.grid(axis="y", alpha=0.25); [ax1.spines[s].set_visible(False) for s in ("top","right")]

    cols=[POS if d>=0 else NEG for d in diff]
    ax2.bar(x, diff, 0.6, color=cols, yerr=[diff-lo, hi-diff], capsize=4, error_kw=dict(ecolor="grey",lw=1.3))
    ax2.axhline(0, color="grey", lw=1)
    for i in range(NB):
        va="bottom" if diff[i]>=0 else "top"
        ax2.text(x[i], hi[i]+0.004 if diff[i]>=0 else lo[i]-0.004, f"{diff[i]:+.3f}", ha="center", va=va, fontsize=11)
    ax2.set_ylabel("antiGen - Smoothed Counts\n(recall@10%)", fontsize=12)
    ax2.set_xlabel("times this exact substitution (pos, alt) occurred in the training corpus", fontsize=12.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=12)
    ax2.grid(axis="y", alpha=0.25); [ax2.spines[s].set_visible(False) for s in ("top","right")]

    plt.tight_layout()
    out=f"{OUTDIR}/recall_by_training_recurrence_{tree}"
    fig.savefig(out+".png", dpi=150, bbox_inches="tight"); fig.savefig(out+".pdf", bbox_inches="tight")
    print(f"{tree}: n/bin={dict(zip([l.split(chr(10))[0] for l in labs],ns.tolist()))}  "
          f"ens={np.round(em,3).tolist()} sm={np.round(sm,3).tolist()} -> {out}.png")

for t in sys.argv[1:]: plot_tree(t)
