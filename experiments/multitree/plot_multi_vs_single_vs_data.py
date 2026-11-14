#!/usr/bin/env python
"""Multi-tree antiGen minus single-tree antiGen (recall@10), per virus, as a function of how
much training data that virus has. Two panels: all substitutions and de novo only.

multi-tree = 5-seed epoch-3 ensemble (npz 'joint' arm); single-tree = single-virus ESM-650m
lm-only (npz 'single' arm). x = number of training substitution events for that virus (no-attach
corpus). Paired event bootstrap CI per virus. Reads perevent_fig13_noattach.npz.
"""
import os, sys, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available=False; _iu._torchvision_version="N/A"
from antiGen.load import load_examples
from score_perevent import events_of, ALPHA_SET, D

R="/large_storage/hielab/aakarsh/multitree"
NA="/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
NPZ=f"{R}/fc13_joint_650m/joint13_lmonly_noattach_recallsel_s0/perevent_fig13_noattach.npz"
OUTB="/home/aakarsh/projects/antiGen/experiments/multitree/figures/multi_vs_single_by_datasize_noattach"
TREES=['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4','h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
NICE={'h5nx':'H5Nx HA','ncov':'SARS-CoV-2 S','rsv_a':'RSV A F','rsv_b':'RSV B F','measles':'Measles H',
      'denv1':'DENV1 E','denv2':'DENV2 E','denv3':'DENV3 E','denv4':'DENV4 E','h3n2_ha':'H3N2 HA',
      'h1n1pdm_ha':'H1N1pdm HA','vic_ha':'B/Vic HA','hmpv_a_g':'hMPV A G'}
rng=np.random.default_rng(0); B=5000

def train_events(tree):
    tr=load_examples(f"{NA}/examples_{tree}@{D}_train.txt",ALPHA_SET,None,source=tree)
    return sum(len(events_of(ex)) for ex in tr)

Z=np.load(NPZ)
sizes={t:train_events(t) for t in TREES}
print("training events per virus:", {t:sizes[t] for t in TREES})

# one stable colour per virus (ordered by training size so the legend reads small->large)
ORDER=sorted(TREES, key=lambda t: sizes[t])
cmap=plt.get_cmap("tab20")
COL={t:cmap(i % 20) for i,t in enumerate(ORDER)}

def panel(ax, sub, comp, title, ylabel):
    """comp = the comparator arm in the npz ('single' or 'cpssm_dir'); y = joint - comp."""
    ax.axhline(0, color="grey", lw=1)
    xs=[]; ys=[]
    for t in ORDER:
        kj=f"{t}__{sub}__joint"; kc=f"{t}__{sub}__{comp}"
        if kj not in Z.files or kc not in Z.files: continue
        j=Z[kj]; c=Z[kc]
        if len(j)==0: continue
        x=sizes[t]; y=float((j-c).mean())
        xs.append(x); ys.append(y)
        ax.scatter([x],[y], s=70, color=COL[t], alpha=0.9, edgecolor="white",
                   linewidth=0.6, zorder=3, label=NICE[t])
    xs=np.array(xs); ys=np.array(ys); lx=np.log10(xs)
    # line of best fit (y ~ log10 training events)
    m,b=np.polyfit(lx,ys,1)
    xg=np.linspace(lx.min(),lx.max(),100)
    ax.plot(10**xg, m*xg+b, color="#333", lw=1.8, ls="--", zorder=2)
    # correlation + p (Pearson on log x, and Spearman on raw)
    r,pr=stats.pearsonr(lx,ys); rho,ps=stats.spearmanr(xs,ys)
    ax.text(0.03,0.04,
            f"Pearson r = {r:.2f}  (p = {pr:.3f})\nSpearman $\\rho$ = {rho:.2f}  (p = {ps:.3f})",
            transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.85))
    ax.set_xscale("log")
    ax.set_xlabel("training substitution events for that virus  (no-attach, log scale)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    [ax.spines[s].set_visible(False) for s in ("top","right")]

fig,(a1,a2)=plt.subplots(1,2,figsize=(15.5,6.4))
panel(a1,"all","single","vs single-tree antiGen  (all substitutions)",
      "multi-tree − single-tree   recall@10%")
panel(a2,"all","cpssm_dir","vs Smoothed Counts  (all substitutions)",
      "multi-tree − Smoothed Counts   recall@10%")
# one shared legend (virus order = ascending training size), to the right
h,l=a2.get_legend_handles_labels()
fig.legend(h,l,title="virus (small → large data)", loc="center left",
           bbox_to_anchor=(0.995,0.5), fontsize=10, title_fontsize=10, frameon=False)
fig.suptitle("Multi-tree antiGen advantage by how much training data a virus has  (all substitutions)\n"
             "no-attach, antiGen = 5-seed epoch-3 ensemble",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout(rect=[0,0,0.9,1])
fig.savefig(OUTB+".png", dpi=150, bbox_inches="tight")
fig.savefig(OUTB+".pdf", bbox_inches="tight")
print("wrote", OUTB+".png / .pdf")
