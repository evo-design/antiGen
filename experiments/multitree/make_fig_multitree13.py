#!/usr/bin/env python
"""High-res multi-tree forecast figure, Ivan-fig5 visual language, our spec:
13-tree panel (Ivan's MULTITREE_PANEL=13 set), MICRO-pooled recall@10, joint = the
650M head-only model at epoch 6, both subsets (ALL + de novo).

Top row: micro-pooled recall@10 ladder (rungs: count/where baselines -> single -> joint),
with the q=10% chance line. Bottom row: per-tree joint-ep6 vs single dot plot. Left
column ALL, right column de novo. Reads the recall JSONs already on disk.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R="/large_storage/hielab/aakarsh/multitree"; D="2025-01-01"
OUT="/home/aakarsh/projects/antiGen/experiments/multitree/figures"; os.makedirs(OUT, exist_ok=True)
# Ivan panel-13 tree order + labels
TREES=['ncov','h5nx','measles','rsv_a','rsv_b','denv1','denv2','denv3','denv4',
       'h3n2_ha','h1n1pdm_ha','hmpv_b_g','vic_ha']
TLAB={'ncov':'SARS-CoV-2 S','h5nx':'H5Nx HA','measles':'Measles H','rsv_a':'RSV A F',
      'rsv_b':'RSV B F','denv1':'DENV1 E','denv2':'DENV2 E','denv3':'DENV3 E','denv4':'DENV4 E',
      'h3n2_ha':'H3N2 HA','h1n1pdm_ha':'H1N1pdm HA','hmpv_b_g':'hMPV B G','vic_ha':'B/Victoria HA'}
ANTIGEN_BLUE="#023e8a"; GATE_BLUE="#0077b6"
# rungs sourced from ep6 json (baselines + ESM) and from the user-models json
USER={'occ_pssm','site_prop','cpssm_dir'}
RUNGS=['occ_pssm','where_only','substitution','frequency','where_times_which',
       'site_prop','cpssm_dir','single','joint']
RLAB={'occ_pssm':'PSSM (residue occurrence)','where_only':'Where only','substitution':'Substitution matrix',
      'frequency':'PSSM / freq. (events)','where_times_which':'Where × which (codon×BLOSUM)',
      'site_prop':'Site-propensity (rank-1)','cpssm_dir':'PSSM + Dirichlet (counted)',
      'single':'antiGen single-tree','joint':'antiGen multi-tree (ep6)'}
RCOL={'occ_pssm':'#c6bbbb','where_only':'#b3aaaa','substitution':'#9aa39a',
      'frequency':'#888f88','where_times_which':'#555560',
      'site_prop':'#b5651d','cpssm_dir':'#8c4a12','single':GATE_BLUE,'joint':ANTIGEN_BLUE}
_um=None
def um_val(rung,t,sub):
    global _um
    if _um is None:
        import json as _j
        _um=_j.load(open(f"{R}/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0/usermodels_recall@{D}.json"))
    c=_um.get(sub,{}).get(t)
    return c[rung]['recall10'] if c else None

def load(p): return json.load(open(p))["results"] if os.path.exists(p) else None
ep6=load(f"{R}/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0/recall_epoch6_alltrees@{D}.json")
def single(t,sub):
    r=load(f"{R}/forecast_single_650m/single_{t}_s0/forecast_recall@{D}.json")
    x=r.get(sub,{}).get(t) if r else None
    return x['model']['0.1'] if x else None
def cell(t,sub): return ep6.get(sub,{}).get(t)
def rung_val(rung,t,sub):
    c=cell(t,sub)
    if not c: return None
    if rung in USER: return um_val(rung,t,sub)
    if rung=='joint': return c['model']['0.1']
    if rung=='single': return single(t,sub)
    return c['baselines'][rung]['0.1']
def micro(rung,sub):
    num=den=0
    for t in TREES:
        c=cell(t,sub); v=rung_val(rung,t,sub)
        if c and v is not None: num+=v*c['n_mutations']; den+=c['n_mutations']
    return num/den if den else float('nan')

fig=plt.figure(figsize=(15,12)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.5],hspace=0.30,wspace=0.42)
for col,(sub,title) in enumerate([('all','ALL post-D substitutions'),('novel','de novo (novel) substitutions')]):
    # --- ladder (micro-pooled) ---
    axl=fig.add_subplot(gs[0,col])
    vals=[(r,micro(r,sub)) for r in RUNGS]; vals.sort(key=lambda x:x[1])
    y=range(len(vals))
    axl.barh(list(y),[v for _,v in vals],color=[RCOL[r] for r,_ in vals],height=0.7)
    for i,(r,v) in enumerate(vals):
        axl.text(v+0.005,i,f"{v:.3f}",va='center',fontsize=8,color='#333')
    axl.set_yticks(list(y)); axl.set_yticklabels([RLAB[r] for r,_ in vals],fontsize=9)
    axl.axvline(0.10,ls=':',color='grey',lw=1)  # q=10% chance
    axl.set_xlim(0,1.08); axl.set_xlabel("micro-pooled recall@10%"); axl.set_title(title,fontsize=11,fontweight='bold')
    axl.grid(axis='x',color='#eee'); axl.set_axisbelow(True)
    # --- per-tree forest (joint ep6 vs single) ---
    axf=fig.add_subplot(gs[1,col])
    order=list(reversed(TREES))
    for i,t in enumerate(order):
        j=rung_val('joint',t,sub); s=single(t,sub); b=cell(t,sub)['best_baseline']['0.1'] if cell(t,sub) else None
        if j is not None and s is not None: axf.plot([min(j,s),max(j,s)],[i,i],color="#ccc",lw=1.5,zorder=1)
        if b is not None: axf.plot([b,b],[i-0.28,i+0.28],color="#888",lw=2,zorder=2)
        if s is not None: axf.scatter(s,i,s=70,color=GATE_BLUE,marker='s',zorder=3)
        if j is not None: axf.scatter(j,i,s=85,color=ANTIGEN_BLUE,marker='o',zorder=4)
    mj,ms=micro('joint',sub),micro('single',sub)
    axf.axvline(mj,ls='--',color=ANTIGEN_BLUE,lw=1,alpha=0.7); axf.axvline(ms,ls=':',color=GATE_BLUE,lw=1,alpha=0.7)
    axf.set_yticks(range(len(order))); axf.set_yticklabels([TLAB[t] for t in order],fontsize=9)
    axf.set_xlim(0.3,1.02); axf.set_xlabel("recall@10%  (per tree)")
    axf.set_title(f"per-tree · micro-pooled: joint {mj:.3f} · single {ms:.3f}",fontsize=10)
    axf.grid(axis='x',color='#eee'); axf.set_axisbelow(True)
leg=[Line2D([],[],marker='o',color='w',markerfacecolor=ANTIGEN_BLUE,markersize=10,label='antiGen multi-tree (650M, ep6)'),
     Line2D([],[],marker='s',color='w',markerfacecolor=GATE_BLUE,markersize=9,label='antiGen single-tree (650M)'),
     Line2D([],[],color='#888',lw=2,label='best pre-D baseline')]
fig.legend(handles=leg,loc='lower center',ncol=3,fontsize=10,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.suptitle("Forecast @2025-01-01 · 13-tree panel · ESM-2 650M head-only · recall@10% (micro-pooled)",
             fontsize=13,fontweight='bold',y=0.995)
fig.tight_layout(rect=[0,0.02,1,0.98])
for ext,dpi in [('png',300),('pdf',None)]:
    fig.savefig(f"{OUT}/fig_multitree13_recall10.{ext}", dpi=dpi, bbox_inches='tight')
print("wrote", f"{OUT}/fig_multitree13_recall10.png (300 dpi) + .pdf")
print(f"\nMICRO-pooled recall@10 (13 trees):")
for sub in ('all','novel'):
    print(f"  {sub:5}: joint {micro('joint',sub):.3f}  single {micro('single',sub):.3f}  "
          f"wxw {micro('where_times_which',sub):.3f}  freq {micro('frequency',sub):.3f}")
