#!/usr/bin/env python
"""Does the multi-tree-minus-Smoothed-Counts gap (all substitutions) track EFFECTIVE sample
size better than raw training-event count? Two panels, same y, x = raw events vs ESS.

ESS per virus = inverse-Simpson effective number of independent training haplotypes:
  ESS = (sum c)^2 / sum c^2  over counts c of identical training-node sequences.
A near-clonal pool (one dominant haplotype) -> ESS ~ 1; a diverse pool -> ESS ~ n_seqs.
(We also print unique-haplotype count as a cross-check.)
"""
import os, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
from antiGen.load import load_examples
from score_perevent import ALPHA_SET, D

R="/large_storage/hielab/aakarsh/multitree"
NA="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
NPZ=f"{R}/fc13_joint_650m/joint13_lmonly_noattach_recallsel_s0/perevent_fig13_noattach.npz"
OUTB="/home/aakarsh/projects/antiGen/experiments/multitree/figures/gap_vs_ess_noattach"
TREES=['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
NICE={'h5nx':'H5Nx HA','ncov':'SARS-CoV-2 S','rsv_a':'RSV A F','rsv_b':'RSV B F','measles':'Measles H',
      'denv1':'DENV1 E','denv2':'DENV2 E','denv3':'DENV3 E','denv4':'DENV4 E','h3n2_ha':'H3N2 HA',
      'h1n1pdm_ha':'H1N1pdm HA','vic_ha':'B/Vic HA','hmpv_a_g':'hMPV A G'}

def ess_and_count(tree):
    tr=load_examples(f"{NA}/examples_{tree}@{D}_train.txt",ALPHA_SET,None,source=tree)
    from collections import Counter
    hap=Counter(tuple(ex.aa_idx.tolist()) for ex in tr)   # identical node sequences = haplotypes
    c=np.array(list(hap.values()),dtype=float)
    ess=(c.sum()**2)/(c**2).sum()                          # inverse-Simpson effective N
    n_events=sum(int((ex.mat>0).sum()) for ex in tr) if hasattr(tr[0],'mat') else 0
    return ess, len(hap), len(tr), n_events

import contextlib, io
stats_v={}
with contextlib.redirect_stdout(io.StringIO()):
    for t in TREES: stats_v[t]=ess_and_count(t)
print("virus         ESS   uniq  nseq")
for t in TREES:
    e,u,n,_=stats_v[t]; print(f"{t:12} {e:6.1f} {u:5d} {n:5d}")

Z=np.load(NPZ)
ORDER=sorted(TREES, key=lambda t: stats_v[t][0])
cmap=plt.get_cmap("tab20"); COL={t:cmap(i%20) for i,t in enumerate(ORDER)}

def gap(t):
    kj=f"{t}__all__joint"; kc=f"{t}__all__cpssm_dir"
    if kj not in Z.files: return None
    return float((Z[kj]-Z[kc]).mean())

def panel(ax, xof, xlabel, title):
    xs=[]; ys=[]
    for t in ORDER:
        g=gap(t)
        if g is None: continue
        x=xof(t); xs.append(x); ys.append(g)
        ax.scatter([x],[g], s=75, color=COL[t], alpha=0.9, edgecolor="white", lw=0.6, zorder=3, label=NICE[t])
    xs=np.array(xs); ys=np.array(ys); lx=np.log10(xs)
    m,b=np.polyfit(lx,ys,1); xg=np.linspace(lx.min(),lx.max(),100)
    ax.plot(10**xg, m*xg+b, color="#333", lw=1.8, ls="--", zorder=2)
    ax.axhline(0, color="grey", lw=1)
    r,pr=stats.pearsonr(lx,ys); rho,ps=stats.spearmanr(xs,ys)
    ax.text(0.03,0.04,f"Pearson r = {r:.2f}  (p = {pr:.3f})\nSpearman $\\rho$ = {rho:.2f}  (p = {ps:.3f})",
            transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.85))
    ax.set_xscale("log"); ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("multi-tree − Smoothed Counts   recall@10%", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.grid(alpha=0.25)
    [ax.spines[s].set_visible(False) for s in ("top","right")]

fig,(a1,a2)=plt.subplots(1,2,figsize=(15.5,6.4))
panel(a1, lambda t: stats_v[t][3] if stats_v[t][3] else 1, "training substitution events  (nominal, log scale)",
      "vs raw event count")
panel(a2, lambda t: stats_v[t][0], "effective sample size  (inverse-Simpson haplotypes, log scale)",
      "vs effective sample size")
h,l=a1.get_legend_handles_labels()
fig.legend(h,l,title="virus (low → high ESS)",loc="center left",bbox_to_anchor=(0.995,0.5),
           fontsize=10,title_fontsize=10,frameon=False)
fig.suptitle("Multi-tree − Smoothed Counts gap (all substitutions) vs nominal count and vs ESS\n"
             "no-attach, antiGen = 5-seed epoch-3 ensemble", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout(rect=[0,0,0.9,1])
fig.savefig(OUTB+".png", dpi=150, bbox_inches="tight"); fig.savefig(OUTB+".pdf", bbox_inches="tight")
print("wrote", OUTB+".png / .pdf")
