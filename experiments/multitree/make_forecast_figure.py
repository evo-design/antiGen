#!/usr/bin/env python
"""Headline forecast figure + table, epoch-6 joint as canonical.

Cleveland dot plot (magnitude-by-identity): per tree, recall@10 of the joint-650M
(fixed epoch 6, the early-plateau checkpoint) vs the single-virus 650M, with the best
pre-D baseline drawn as a reference tick. Two panels: ALL post-D substitutions and de
novo. Absolute values with the baseline as a tick (not deltas). Okabe-Ito CVD-safe hues.

Reads the recall JSONs already written by score_forecast_recall.py.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/large_storage/hielab/aakarsh/multitree"
D = "2025-01-01"
OUT = "/home/aakarsh/projects/antiGen/experiments/multitree/figures"
os.makedirs(OUT, exist_ok=True)
PANEL = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','hmpv_a_g','hmpv_b_g','vic_ha']
# Okabe-Ito (colorblind-safe by construction)
C_JOINT, C_SINGLE, C_BASE = "#0072B2", "#D55E00", "#666666"

def load(p): return json.load(open(p))["results"] if os.path.exists(p) else None
ep6 = load(f"{R}/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0/recall_epoch6_alltrees@{D}.json")
def single(t, sub):
    r = load(f"{R}/forecast_single_650m/single_{t}_s0/forecast_recall@{D}.json")
    x = r.get(sub,{}).get(t) if r else None
    return x['model']['0.1'] if x else None
def joint(t, sub):
    x = ep6.get(sub,{}).get(t); return x['model']['0.1'] if x else None
def base(t, sub):
    x = ep6.get(sub,{}).get(t); return x['best_baseline']['0.1'] if x else None
def nmut(t, sub):
    x = ep6.get(sub,{}).get(t); return x['n_mutations'] if x else None

# ---- table / CSV ----
rows = []
for sub in ('all','novel'):
    for t in PANEL:
        rows.append((sub,t,nmut(t,sub),joint(t,sub),single(t,sub),base(t,sub)))
csv = f"{OUT}/forecast_epoch6_recall10.csv"
with open(csv,"w") as fh:
    fh.write("subset,tree,n_mutations,joint_ep6_recall10,single650_recall10,best_baseline_recall10\n")
    for sub,t,n,j,s,b in rows:
        fh.write(f"{sub},{t},{'' if n is None else int(n)},"
                 f"{'' if j is None else f'{j:.4f}'},{'' if s is None else f'{s:.4f}'},"
                 f"{'' if b is None else f'{b:.4f}'}\n")
def macro(sub, which):
    f={'joint':joint,'single':single,'base':base}[which]
    v=[f(t,sub) for t in PANEL]; v=[x for x in v if x is not None]; return sum(v)/len(v)

# ---- figure ----
order = sorted(PANEL, key=lambda t: joint(t,'all') or 0)
fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
for ax, sub, title in zip(axes, ('all','novel'),
                          ('ALL post-D substitutions', 'de novo (novel) substitutions')):
    y = range(len(order))
    for i, t in enumerate(order):
        j, s, b = joint(t,sub), single(t,sub), base(t,sub)
        if j is not None and s is not None:
            ax.plot([min(j,s),max(j,s)], [i,i], color="#cccccc", lw=1.5, zorder=1)
        if b is not None:
            ax.plot([b,b],[i-0.28,i+0.28], color=C_BASE, lw=2, zorder=2)  # baseline tick
        if s is not None: ax.scatter(s,i, s=70, color=C_SINGLE, zorder=3, marker='s')
        if j is not None: ax.scatter(j,i, s=80, color=C_JOINT, zorder=4, marker='o')
    mj, ms = macro(sub,'joint'), macro(sub,'single')
    ax.axvline(mj, color=C_JOINT, ls='--', lw=1, alpha=0.7, zorder=1)
    ax.axvline(ms, color=C_SINGLE, ls=':', lw=1, alpha=0.7, zorder=1)
    ax.set_title(f"{title}\nmacro: joint {mj:.3f} · single {ms:.3f}", fontsize=10)
    ax.set_xlabel("recall@10%  (next-mutation recall, gaps/stops masked)")
    ax.set_xlim(0.45, 1.02); ax.grid(axis='x', color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=9)
from matplotlib.lines import Line2D
leg = [Line2D([],[],marker='o',color='w',markerfacecolor=C_JOINT,markersize=9,label='joint-650M (epoch 6)'),
       Line2D([],[],marker='s',color='w',markerfacecolor=C_SINGLE,markersize=9,label='single-virus 650M'),
       Line2D([],[],color=C_BASE,lw=2,label='best pre-D baseline')]
axes[1].legend(handles=leg, loc='lower right', fontsize=9, frameon=False)
fig.suptitle("Forecast @2025-01-01: joint vs single-virus ESM-2 650M (head-only), recall@10%",
             fontsize=12, y=0.99)
fig.tight_layout(rect=[0,0,1,0.96])
png=f"{OUT}/forecast_epoch6_recall10.png"
fig.savefig(png, dpi=150, bbox_inches='tight'); fig.savefig(png.replace('.png','.pdf'), bbox_inches='tight')
print("wrote", png)
print("wrote", csv)
print(f"\nMACRO recall@10  ALL:   joint {macro('all','joint'):.3f}  single {macro('all','single'):.3f}  base {macro('all','base'):.3f}")
print(f"MACRO recall@10  NOVEL: joint {macro('novel','joint'):.3f}  single {macro('novel','single'):.3f}  base {macro('novel','base'):.3f}")
